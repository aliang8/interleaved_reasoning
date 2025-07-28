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
Generalized preprocessing script for multiple datasets.
Outputs a single large parquet file with standardized format.

python3 data_gen/create_parquets.py --config_file=data_gen/dataset_config.yaml
"""

import argparse
import os
import yaml
import datasets
from datasets import concatenate_datasets
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from generate_concat_interleaved_code import test_list_to_unittest
from helpers import StandardizedRewardModel, save_to_parquet_all, combine_examples

ADDITIONAL_INSTRUCTION = """
First, outline the solution in a markdown format.
Then, write the code to implement the solution.
Finally, generate unit tests to test the code. Format the unit tests as a python function with a docstring. Use this exact format:
```python
import unittest
from task_func import task_func

class Test(unittest.TestCase):
    def test_case_1(self):
        # Test case 1 description
        result = task_func(...)
        self.assertEqual(result, expected_value)
```
"""


def process_knights_and_knaves(local_dir, subsets):
    data_source = "K-and-K/knights-and-knaves"
    instruction_following = "You must infer the identity of each character. At the end of your answer, you must clearly state the identity of each character by following the format:\n\nCONCLUSION:\n(1) ...\n(2) ...\n(3) ..."

    train_datasets = []
    test_datasets = []
    for subset in subsets:
        print(f"Loading subset: {subset}")
        train_dataset = datasets.load_dataset(data_source, "train", split=subset)
        test_dataset = datasets.load_dataset(data_source, "test", split=subset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)

    if len(train_datasets) > 1:
        combined_train_dataset = concatenate_datasets(train_datasets)
        combined_test_dataset = concatenate_datasets(test_datasets)
    else:
        combined_train_dataset = train_datasets[0]
        combined_test_dataset = test_datasets[0]

    def extract_solution(solution_text):
        return solution_text.strip()

    def make_map_fn(split):
        def process_fn(example, idx):
            quiz_raw = example.pop("quiz")
            question = quiz_raw + "\n\n" + instruction_following
            solution_text_raw = example.pop("solution_text")
            solution = extract_solution(solution_text_raw)

            # Clear the example to only keep what we need
            example.clear()

            # Create standardized reward model
            reward_model = StandardizedRewardModel(
                ground_truth=[solution], style="rule"
            )

            data = {
                "data_source": "k&k",
                "prompt": question,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": [solution_text_raw],
                    "question": [quiz_raw],
                },
            }
            return data

        return process_fn

    train_dataset = combined_train_dataset.map(
        function=make_map_fn("train"), with_indices=True
    )
    test_dataset = combined_test_dataset.map(
        function=make_map_fn("test"), with_indices=True
    )
    return train_dataset, test_dataset


def process_simpleqa(local_dir):
    data_source = "SimpleQA"
    print("Loading SimpleQA from HuggingFace...")
    ds = datasets.load_dataset("basicv8vc/SimpleQA")
    test_dataset = ds["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example["problem"]
            answer = example["answer"]

            # Clear the example to only keep what we need
            example.clear()

            # Create standardized reward model
            reward_model = StandardizedRewardModel(ground_truth=[answer], style="rule")

            data = {
                "data_source": data_source,
                "prompt": question,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": [question],
                    "answer": [answer],
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return test_dataset


def process_mbpp(local_dir):
    data_source = "code_mbpp"
    print("Loading MBPP from HuggingFace...")
    ds = datasets.load_dataset("mbpp")
    test_dataset = ds["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example["text"]
            # Add instructional text
            instr = "\n\nYou should write self-contained code starting with:\n```\ndef task_func(args):\n```"
            prompt += instr
            answer = example["code"]
            # Generate unit tests with function renaming
            unit_tests = test_list_to_unittest(
                example["test_list"], func_name="task_func"
            )
            answer = re.sub(r"def\s+\w+\(", "def task_func(", answer)
            answer = answer.replace("solution(", "task_func(")

            # Clear the example to only keep what we need
            example.clear()

            # Create standardized reward model with unit tests
            reward_model = StandardizedRewardModel(
                ground_truth=[answer], style="code", unit_tests=[unit_tests], libs=[]
            )

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": [prompt],
                    "answer": [answer],
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return test_dataset


def process_mbpp_combined(
    local_dir,
    n=2,
    sep=" ",
    prompt_combine_mode="space",
    llm_model_name="Qwen/Qwen3-8B",
    llm_device_map="auto",
    prompt_prefix=None,
):
    data_source = f"code_mbpp_combined_{prompt_combine_mode}_{n}"
    print("Loading MBPP from HuggingFace...")
    ds = datasets.load_dataset("mbpp")
    test_dataset = ds["test"]

    def combine_dataset(dataset, split):
        combined = []
        for i in range(0, len(dataset), n):
            group = dataset[i : i + n]
            if len(group) < n:
                continue
            # For MBPP, use 'text' as prompt, 'code' as answer
            group_dict = {
                "prompt": group["text"],
                "code": group["code"],
                "test_list": group["test_list"],
            }
            # Combine unit tests for all problems in the group
            combined_unit_tests = [
                test_list_to_unittest(tests, func_name="task_func")
                for tests in group_dict["test_list"]
            ]
            # Rename function in code
            group_dict["code"] = [
                re.sub(r"def\s+\w+\(", "def task_func(", c).replace(
                    "solution(", "task_func("
                )
                for c in group_dict["code"]
            ]

            combined_ex = combine_examples(
                group_dict,
                prompt_key="prompt",
                answer_key="code",
                prompt_combine_mode="numbered_list",
                prompt_prefix="Solve the following coding problems:",
            )

            # Create standardized reward model with unit tests
            reward_model = StandardizedRewardModel(
                ground_truth=combined_ex["answers"],
                style="code",
                unit_tests=combined_unit_tests,
                libs=[],
            )

            # Add instructional text to combined prompt
            prompt = combined_ex["prompt"]
            instr = "\n\nYou should write self-contained code starting with:\n```\ndef task_func(args):\n```"
            prompt = prompt + instr

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": i // n,
                    "questions": group_dict["prompt"],
                    "answers": group_dict["code"],
                },
            }
            combined.append(data)
        return combined

    test_combined = combine_dataset(test_dataset, "test")
    return test_combined


def process_simpleqa_combined(
    local_dir,
    n=2,
    sep=" ",
    prompt_combine_mode="space",
    llm_model_name="Qwen/Qwen3-8B",
    llm_device_map="auto",
    prompt_prefix=None,
):
    """
    Loads simpleqa, combines every n examples into one, and saves to parquet.
    prompt_combine_mode: 'space', 'and', or 'llm'
    """
    data_source = f"simpleqa_combined_{prompt_combine_mode}_{n}"
    print("Loading SimpleQA from HuggingFace...")
    ds = datasets.load_dataset("basicv8vc/SimpleQA")
    test_dataset = ds["test"]
    print(f"Loaded {len(test_dataset)} examples from SimpleQA")

    def combine_dataset(dataset, split):
        combined = []
        for i in range(0, len(dataset), n):
            group = dataset[i : i + n]

            combined_ex = combine_examples(
                group,
                prompt_key="problem",
                answer_key="answer",
                sep=sep,
                prompt_combine_mode=prompt_combine_mode,
                llm_model_name=llm_model_name,
                llm_device_map=llm_device_map,
                prompt_prefix=prompt_prefix,
            )

            # Create standardized reward model
            reward_model = StandardizedRewardModel(
                ground_truth=combined_ex["answers"], style="rule"
            )

            data = {
                "data_source": data_source,
                "prompt": combined_ex["prompt"],
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": i // n,
                    "questions": group["problem"],
                    "answers": group["answer"],
                },
            }
            combined.append(data)
        return combined

    test_combined = combine_dataset(test_dataset, "test")
    return test_combined


def process_math500(local_dir):
    data_source = "math500"
    print("Loading Math500 from HuggingFace...")
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500")
    test_dataset = ds["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example["problem"]
            answer = example["answer"]

            # Clear the example to only keep what we need
            example.clear()

            # Create standardized reward model
            reward_model = StandardizedRewardModel(ground_truth=[answer], style="rule")

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": [prompt],
                    "answer": [answer],
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return test_dataset


def process_math500_combined(local_dir, n=2):
    data_source = f"math500_combined_{n}"
    print("Loading Math500 from HuggingFace...")
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500")
    test_dataset = ds["test"]

    training_prompt = test_dataset.select(range(400))
    test_prompt = test_dataset.select(range(400, 500))

    def combine_dataset(dataset, split):
        combined = []
        for i in range(0, len(dataset), n):
            group = dataset[i : i + n]
            if len(group) < n:
                continue
            problems = group["problem"]
            answers = group["answer"]
            combined_ex = combine_examples(
                group,
                prompt_key="problem",
                answer_key="answer",
                prompt_combine_mode="numbered_list",
                prompt_prefix="Solve the following math problems:",
            )

            # Create standardized reward model
            reward_model = StandardizedRewardModel(ground_truth=answers, style="rule")

            data = {
                "data_source": data_source,
                "prompt": combined_ex["prompt"],
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": i // n,
                    "questions": problems,
                    "answers": answers,
                },
            }
            combined.append(data)
        return combined

    test_combined = combine_dataset(test_prompt, "test")
    train_combined = combine_dataset(training_prompt, "train")
    return train_combined, test_combined


def process_bcb(local_dir):
    data_source = "bcb_outline_code_test_interleave"
    print("Loading BigCodeBench from HuggingFace...")

    ds = datasets.load_dataset("bigcode/bigcodebench", split="v0.1.4")

    # Split into train (first 500) and test (rest)
    train_dataset = ds.select(range(500))
    test_dataset = ds.select(range(500, len(ds)))

    print(f"BCB dataset split: {len(train_dataset)} train, {len(test_dataset)} test")

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example["instruct_prompt"] + "\n" + ADDITIONAL_INSTRUCTION
            answer = example["canonical_solution"]
            unit_tests = example["test"]
            libs = example["libs"]

            # Clear the example to only keep what we need
            example.clear()

            # Create standardized reward model with unit tests
            reward_model = StandardizedRewardModel(
                ground_truth=[answer],
                style="code",
                unit_tests=[unit_tests],
                libs=[libs],
            )

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": [prompt],
                    "answer": [answer],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


def process_bcb_hard(local_dir):
    data_source = "bcb_outline_code_test_interleave"
    print("Loading BigCodeBench from HuggingFace...")

    ds = datasets.load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")

    # Split into train (first 100) and test (rest)
    train_dataset = ds.select(range(100))
    test_dataset = ds.select(range(100, len(ds)))

    print(f"BCB dataset split: {len(train_dataset)} train, {len(test_dataset)} test")

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example["instruct_prompt"] + "\n" + ADDITIONAL_INSTRUCTION
            answer = example["canonical_solution"]
            unit_tests = example["test"]
            libs = example["libs"]

            # Clear the example to only keep what we need
            example.clear()

            # Create standardized reward model with unit tests
            reward_model = StandardizedRewardModel(
                ground_truth=[answer],
                style="code",
                unit_tests=[unit_tests],
                libs=[libs],
            )

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "reward_model": reward_model.to_dict(),
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": [prompt],
                    "answer": [answer],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset


def process_datasets_from_config(config):
    """Process all datasets specified in the config."""
    all_datasets = {}

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")

        if dataset_name == "knights_and_knaves":
            subsets = dataset_config.get("subsets", ["2ppl"])
            train_dataset, test_dataset = process_knights_and_knaves(
                config["local_dir"], subsets
            )
            all_datasets[dataset_name] = (train_dataset, test_dataset)

        elif dataset_name == "simpleqa":
            combine_n = dataset_config.get("combine_n", 1)
            if combine_n > 1:
                prompt_combine_mode = dataset_config.get("prompt_combine_mode", "space")
                llm_model_name = dataset_config.get("llm_model_name", "Qwen/Qwen3-8B")
                llm_device_map = dataset_config.get("llm_device_map", "auto")
                prompt_prefix = dataset_config.get("prompt_prefix", None)
                test_dataset = process_simpleqa_combined(
                    config["local_dir"],
                    n=combine_n,
                    prompt_combine_mode=prompt_combine_mode,
                    llm_model_name=llm_model_name,
                    llm_device_map=llm_device_map,
                    prompt_prefix=prompt_prefix,
                )
                all_datasets[
                    f"{dataset_name}_combined_{prompt_combine_mode}_{combine_n}"
                ] = (None, test_dataset)
            else:
                test_dataset = process_simpleqa(config["local_dir"])
                all_datasets[dataset_name] = (None, test_dataset)

        elif dataset_name == "mbpp":
            combine_n = dataset_config.get("combine_n", 1)
            if combine_n > 1:
                prompt_combine_mode = dataset_config.get("prompt_combine_mode", "space")
                llm_model_name = dataset_config.get("llm_model_name", "Qwen/Qwen3-8B")
                llm_device_map = dataset_config.get("llm_device_map", "auto")
                prompt_prefix = dataset_config.get("prompt_prefix", None)
                test_dataset = process_mbpp_combined(
                    config["local_dir"],
                    n=combine_n,
                    prompt_combine_mode=prompt_combine_mode,
                    llm_model_name=llm_model_name,
                    llm_device_map=llm_device_map,
                    prompt_prefix=prompt_prefix,
                )
                all_datasets[
                    f"{dataset_name}_combined_{prompt_combine_mode}_{combine_n}"
                ] = (None, test_dataset)
            else:
                test_dataset = process_mbpp(config["local_dir"])
                all_datasets[dataset_name] = (None, test_dataset)

        elif dataset_name == "math500":
            combine_n = dataset_config.get("combine_n", 1)
            if combine_n > 1:
                train_dataset, test_dataset = process_math500_combined(
                    config["local_dir"], n=combine_n
                )
                all_datasets[f"{dataset_name}_combined_{combine_n}"] = (
                    train_dataset,
                    test_dataset,
                )
            else:
                test_dataset = process_math500(config["local_dir"])
                all_datasets[dataset_name] = (None, test_dataset)
        elif dataset_name == "bcb_hard":
            train_dataset, test_dataset = process_bcb_hard(config["local_dir"])
            all_datasets[dataset_name] = (train_dataset, test_dataset)
        elif dataset_name == "bcb":
            train_dataset, test_dataset = process_bcb(config["local_dir"])
            all_datasets[dataset_name] = (train_dataset, test_dataset)
        else:
            print(f"Warning: Unknown dataset {dataset_name}, skipping...")

    return all_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="YAML config file specifying datasets to process",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    print(f"Processing datasets from config: {args.config_file}")
    print(f"Output directory: {config['local_dir']}")
    print(f"Output filename: {config['output_filename']}")

    # Process all datasets
    all_datasets = process_datasets_from_config(config)

    # Save combined dataset
    save_to_parquet_all(all_datasets, config["local_dir"], config["output_filename"])


if __name__ == "__main__":
    main()
