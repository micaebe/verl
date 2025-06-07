import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import random


def calc_perplexity(logits: torch.Tensor,
                    targets: torch.Tensor,
                    mask: torch.Tensor | None = None) -> float:
    log_probs = F.log_softmax(logits, dim=-1)                     # (B, T, V)

    tgt_log_probs = log_probs.gather(-1, targets.unsqueeze(-1))   # (B, T, 1)
    tgt_log_probs = tgt_log_probs.squeeze(-1)                     # (B, T)

    if mask is not None:
        tgt_log_probs = tgt_log_probs * mask
        token_count = mask.sum()
    else:
        token_count = targets.numel()

    nll = -tgt_log_probs.sum() / token_count

    return torch.exp(nll).item()

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)


def extract_answer(model_output: str) -> str:
    # Extract answer after </think>
    think_index = model_output.find("</think>")
    if think_index != -1:
        answer = model_output[think_index + len("</think>"):].strip()
        return answer
    else:
        return None

def compute_score(model_output: str, ground_truth: str, extra_info) -> float:
    print("Computing score...")
    print(f"Model Output: {model_output}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Extra Info: {extra_info}")
    summary = extract_answer(model_output)
    if summary is None:
        return 0.0
    input_text = extra_info.get("input_text", "")
    if not input_text:
        # print warning if input_text is not provided
        print("Warning: input_text is not provided in extra_info.")
        return 0.0
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    summary_ids = tokenizer(summary, return_tensors="pt").input_ids
    if summary_ids.shape[1] > 64:
        # if the summary is longer than 64 tokens, return 0.0
        return 0.0
    completion_ids = tokenizer(ground_truth, return_tensors="pt").input_ids
    # we now compute the logits of input_ids + completion_ids
    gold_ids = torch.cat([input_ids, completion_ids], dim=-1)
    summarizer_ids = torch.cat([summary_ids, completion_ids], dim=-1)
    with torch.no_grad():
        outputs = model(gold_ids)
        gold_logits = outputs.logits
        outputs = model(summarizer_ids)
        summarizer_logits = outputs.logits
    gold_perplexity = calc_perplexity(gold_logits, gold_ids)
    summarizer_perplexity = calc_perplexity(summarizer_logits, summarizer_ids)

    # reward is 1 - summarizer_perplexity / gold_perplexity 
    reward = 1 - summarizer_perplexity / gold_perplexity
    if reward < 0:
        reward = 0.0
    # add a little bonus if the summary length is less than 128 tokens
    # if its 128 the bonus is 0.0, going down to 0 it gets 0.1
    summary_length = len(summary_ids[0])
    bonus = max(0.0, 0.1 - (summary_length / 64) * 0.1)
    reward += bonus
    print(f"Gold Perplexity: {gold_perplexity}, Summarizer Perplexity: {summarizer_perplexity}, Reward: {reward}, Bonus: {bonus}")
    return reward


# lets test the compute_score function

model_output = "The quick brown fox jumps over the lazy dog. </think> The quick brown fox jumps over the lazy dog."
ground_truth = "The quick brown fox jumps over the lazy dog."
extra_info = {"input_text": "The quick brown fox jumps over the lazy dog."}

score = compute_score(model_output, ground_truth, extra_info)
print(f"Score: {score}")  # Expected output: Score: 1.0