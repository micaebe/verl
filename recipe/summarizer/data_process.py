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
"""
Preprocess the dataset to parquet format
"""

import argparse
import os
from functools import partial

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs



def map_fn(example, idx, process_fn, data_source, ability, split, tokenizer):
    text = process_fn(example)
    tokens = tokenizer(text, return_tensors="pt")
    # TODO: add to config
    n_prefix_tokens = 1024
    n_completion_tokens = 1024
    if len(tokens["input_ids"][0]) < n_prefix_tokens + n_completion_tokens:
        # adjust n_prefix_tokens and n_completion_tokens
        n_prefix_tokens = len(tokens["input_ids"][0]) - n_completion_tokens
        if n_prefix_tokens <= 0:
            n_completion_tokens = len(tokens["input_ids"][0]) // 2
            n_prefix_tokens = len(tokens["input_ids"][0]) - n_completion_tokens
        
    if len(tokens["input_ids"][0]) < n_prefix_tokens + n_completion_tokens:
        raise ValueError(f"Input text is too short: {len(tokens['input_ids'][0])} tokens, "
                         f"expected at least {n_prefix_tokens + n_completion_tokens} tokens.")
    input_text = tokenizer.decode(tokens["input_ids"][0][:n_prefix_tokens], skip_special_tokens=True)
    answer_text = tokenizer.decode(tokens["input_ids"][0][n_prefix_tokens:n_prefix_tokens + n_completion_tokens], skip_special_tokens=True)

    prompt = f"Summarize the following text so that it preserves all the information, but is as short as possible.\n\n{input_text}"
    solution = answer_text

    data = {
        "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx, "input_text": input_text},
    }
    return data

def build_book_dataset():
    def process_book(example):
        return example["text"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    data_source = "ubaada/booksum-complete-cleaned"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "chapters", split="train")
    map_fn_train = partial(map_fn, process_fn=process_book, data_source=data_source, ability="English", split="train", tokenizer=tokenizer)
    dataset = dataset.map(map_fn_train, with_indices=True, remove_columns=dataset.column_names)
    test_dataset = load_dataset(data_source, "chapters", split="test")
    map_fn_test = partial(map_fn, process_fn=process_book, data_source=data_source, ability="English", split="test", tokenizer=tokenizer)
    test_dataset = test_dataset.map(map_fn_test, with_indices=True, remove_columns=test_dataset.column_names)
    return dataset, test_dataset


def build_arxiv_dataset():
    def process_book(example):
        return example["article"]
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    data_source = "ccdv/arxiv-summarization"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "document", split="train")
    dataset = dataset.select(range(8000))
    map_fn_train = partial(map_fn, process_fn=process_book, data_source=data_source, ability="English", split="train", tokenizer=tokenizer)
    dataset = dataset.map(map_fn_train, with_indices=True, remove_columns=dataset.column_names)
    test_dataset = load_dataset(data_source, "document", split="test")
    test_dataset = test_dataset.select(range(1000))
    map_fn_test = partial(map_fn, process_fn=process_book, data_source=data_source, ability="English", split="test", tokenizer=tokenizer)
    test_dataset = test_dataset.map(map_fn_test, with_indices=True, remove_columns=test_dataset.column_names)
    return dataset, test_dataset



TASK2DATA = {
    "book": build_book_dataset,
    "arxiv": build_arxiv_dataset,
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/summarizer")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS:
                raise NotImplementedError(f"{task} has not been supported.")

    datasets = []
    test_datasets = []
    for task in args.tasks:
        train_dataset, test_dataset = TASK2DATA[task]()
        datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    train_dataset = concatenate_datasets(datasets)
    test_dataset = concatenate_datasets(test_datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
