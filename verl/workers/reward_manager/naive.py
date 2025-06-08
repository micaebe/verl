# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
import ray

@ray.remote(num_gpus=1)
class Scorer:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading model on GPU...")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        self.max_summary_length = 192
        self.summary_length_bonus = 0.01

    def score(self, data_source, solution_str, ground_truth, extra_info):
        import torch.nn.functional as F
        def calc_perplexity(logits: torch.Tensor,
                    targets: torch.Tensor,
                    mask: torch.Tensor | None = None) -> float:
            logits = logits[:, :-1, :]
            targets = targets[:, 1:]
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
            summary = extract_answer(model_output)
            if summary is None:
                return 0.0
            input_text = extra_info.get("input_text", "")
            if not input_text:
                return 0.0
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
            summary_ids = self.tokenizer(summary, return_tensors="pt").input_ids
            if summary_ids.shape[1] > self.max_summary_length:
                return 0.0
            completion_ids = self.tokenizer(ground_truth, return_tensors="pt").input_ids
            gold_ids = torch.cat([input_ids, completion_ids], dim=-1).cuda()
            summarizer_ids = torch.cat([summary_ids, completion_ids], dim=-1).cuda()
            with torch.no_grad():
                outputs = self.model(gold_ids)
                gold_logits = outputs.logits
                outputs = self.model(summarizer_ids)
                summarizer_logits = outputs.logits
            gold_perplexity = calc_perplexity(gold_logits, gold_ids)
            summarizer_perplexity = calc_perplexity(summarizer_logits, summarizer_ids)

            shift = max(gold_perplexity / 4.0, 3.0)
            reward = 1 - summarizer_perplexity / (gold_perplexity + shift)
            if reward < 0:
                reward = 0.0
                bonus = 0.0
            else:
                summary_length = len(summary_ids[0])
                bonus = max(0.0, self.summary_length_bonus - (summary_length / self.max_summary_length) * self.summary_length_bonus)
            reward += bonus
            print(f"Gold Perplexity: {gold_perplexity}, Summarizer Perplexity: {summarizer_perplexity}, Reward: {reward}, Bonus: {bonus}")
            return reward
        return compute_score(
            model_output=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info
        ) 

    def shutdown(self):
        del self.model
        torch.cuda.empty_cache()
        ray.actor.exit_actor()


class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        scorer = Scorer.remote()
        futures = []
        call_meta = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            ref = scorer.score.remote(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            futures.append(ref)
            call_meta.append((i, valid_response_length - 1))

        scores = ray.get(futures)
        scorer.shutdown.remote()

        for (i, pos), reward in zip(call_meta, scores):
            reward_tensor[i, pos] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
