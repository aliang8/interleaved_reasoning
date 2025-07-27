#!/usr/bin/env python3
"""
Generate paired interleaved reasoning traces for coding prompts.

For each sample, randomly select two distinct prompts from MBPP.
For each, generate:
  <think>...</think> about the solution approach
  <answer>...</answer> with the code solution
for both problems, then concatenate as:
  <think>...1</think>\n<answer>code1</answer>\n<think>...2</think>\n<answer>code2</answer>

Saves the reasoning traces to parquet and JSONL files for SFT training.
"""

import json
import argparse
import random
import os
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import re
import datasets
from interleave_generator import InterleavedResponsesGenerator
from helpers import StandardizedRewardModel, save_to_parquet, save_jsonl, combine_examples

# Prompt configurations for each step
INTERLEAVED_PROMPTS = {
    "think": "Think step by step about how to solve this problem. Begin with <think> and end with </think>.",
    "code": "Now provide the code solution in <answer></answer> tags.",
}

def generate_paired_interleaved_trace(
    generator: InterleavedResponsesGenerator,
    problem1: Dict[str, Any],
    problem2: Dict[str, Any],
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
    indices: list = [],
    use_canonical: bool = False
) -> Dict[str, Any]:
    """Generate a paired interleaved trace for two MBPP problems."""
    prompt1 = problem1["text"]
    prompt2 = problem2["text"]
    
    # Replace function name in canonical solutions and test_list
    canonical_solution_1 = re.sub(r'def\s+\w+\(', 'def task_func(', problem1["code"])
    canonical_solution_1 = canonical_solution_1.replace('solution(', 'task_func(')
    canonical_solution_2 = re.sub(r'def\s+\w+\(', 'def task_func(', problem2["code"])
    canonical_solution_2 = canonical_solution_2.replace('solution(', 'task_func(')
    test_1 = test_list_to_unittest(problem1["test_list"], func_name="task_func")
    test_2 = test_list_to_unittest(problem2["test_list"], func_name="task_func")
    task_id1 = problem1.get("task_id", indices[0] if indices else "Unknown1")
    task_id2 = problem2.get("task_id", indices[1] if indices else "Unknown2")

    # --- Problem 1 ---
    messages1 = [{"role": "user", "content": prompt1}]
    
    # Generate thinking for problem 1
    messages1.append({"role": "user", "content": INTERLEAVED_PROMPTS["think"]})
    think1 = generator.generate_thoughts(
        messages1,
        max_new_tokens=max_new_tokens_per_turn,
        temperature=temperature,
        top_p=top_p
    )
    messages1.append({"role": "assistant", "content": think1})

    # Generate code for problem 1
    if use_canonical:
        code1 = f"<answer>{canonical_solution_1}</answer>"
    else:
        messages1.append({"role": "user", "content": INTERLEAVED_PROMPTS["code"]})
        code1 = generator.generate_answer(
            messages1,
            max_new_tokens=max_new_tokens_per_turn * 2,
            temperature=temperature,
            top_p=top_p
        )
        if '<answer>' not in code1:
            code1 = f"<answer>{code1}</answer>"
    messages1.append({"role": "assistant", "content": code1})

    # --- Problem 2 ---
    messages2 = [{"role": "user", "content": prompt2}]
    
    # Generate thinking for problem 2
    messages2.append({"role": "user", "content": INTERLEAVED_PROMPTS["think"]})
    think2 = generator.generate_thoughts(
        messages2,
        max_new_tokens=max_new_tokens_per_turn,
        temperature=temperature,
        top_p=top_p
    )
    messages2.append({"role": "assistant", "content": think2})

    # Generate code for problem 2
    if use_canonical:
        code2 = f"<answer>{canonical_solution_2}</answer>"
    else:
        messages2.append({"role": "user", "content": INTERLEAVED_PROMPTS["code"]})
        code2 = generator.generate_answer(
            messages2,
            max_new_tokens=max_new_tokens_per_turn * 2,
            temperature=temperature,
            top_p=top_p
        )
        if '<answer>' not in code2:
            code2 = f"<answer>{code2}</answer>"
    messages2.append({"role": "assistant", "content": code2})

    # Build combined prompt and interleaved answer
    combined_ex = combine_examples([problem1, problem2], prompt_key="text", answer_key="code", prompt_combine_mode="numbered_list", prompt_prefix="Solve the following coding problems:")
    combined_prompt = combined_ex["prompt"]
    combined_prompt += "\n\nYou should write self-contained code for each problem starting with:\n```\ndef task_func(args):\n```"
    interleaved_answer = f"{think1}\n{code1}\n{think2}\n{code2}"

    # Validate that we have exactly two <answer></answer> blocks
    answer_blocks = re.findall(r'<answer>.*?</answer>', interleaved_answer, re.DOTALL)
    if len(answer_blocks) != 2:
        import ipdb; ipdb.set_trace()

    return {
        "prompt": combined_prompt,
        "answer": interleaved_answer.strip(),
        "prompt_1": prompt1,
        "prompt_2": prompt2,
        "indices": indices,
        "combined_answers": combined_ex["answers"],
        "reward_model": {
            "test_1": test_1,
            "test_2": test_2,
            "libs_1": "",
            "libs_2": "",
            "ground_truth": ""
        },
        "data_source": "mbpp_INTERLEAVED_PROMPTS",
        "task_id_1": task_id1,
        "task_id_2": task_id2
    }

def load_mbpp_data(limit=None):
    ds = datasets.load_dataset('mbpp', split='train')
    data = list(ds)
    if limit:
        data = data[:limit]
    return data

def test_list_to_unittest(test_list, func_name='solution'):
    # Converts a list of asserts to a unittest.TestCase class string
    lines = ["import unittest", "", "class Test(unittest.TestCase):"]
    for i, test in enumerate(test_list):
        lines.append(f"    def test_case_{i+1}(self):")
        for t in test.split('\n'):
            # Replace function call to match func_name in asserts
            t_mod = re.sub(r'assert\s+\w+\(', f'assert {func_name}(', t)
            lines.append(f"        {t_mod.strip()}")
    return '\n'.join(lines)

def generate_concat_dataset(
    generator: InterleavedResponsesGenerator,
    num_samples: int = 50,
    use_canonical: bool = False,
    max_new_tokens_per_turn: int = 512
) -> List[Dict[str, Any]]:
    """Generate paired interleaved coding traces for multiple MBPP problems."""
    print(f"Generating concat dataset with {num_samples} samples from MBPP...")

    problems = load_mbpp_data(limit=None)

    if len(problems) < 2:
        print("Not enough problems to generate pairs.")
        return []

    entries = []
    used_pairs = set()
    
    for i in tqdm(range(num_samples), desc="Generating paired coding traces"):
        idx1, idx2 = random.sample(range(len(problems)), 2)
        while (idx1, idx2) in used_pairs:
            idx1, idx2 = random.sample(range(len(problems)), 2)
        used_pairs.add((idx1, idx2))
        problem1 = problems[idx1]
        problem2 = problems[idx2]

        # Generate interleaved trace
        trace_data = generate_paired_interleaved_trace(
            generator,
            problem1,
            problem2,
            max_new_tokens_per_turn=max_new_tokens_per_turn,
            temperature=0.2,
            top_p=0.7,
            indices=[idx1, idx2],
            use_canonical=use_canonical
        )

        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth=trace_data["combined_answers"],  # List of individual answers
            style="code",
            unit_tests=[trace_data['reward_model']['test_1'], trace_data['reward_model']['test_2']],
            libs=[trace_data['reward_model']['libs_1'], trace_data['reward_model']['libs_2']]
        )

        # Create result entry for training (following create_parquets format)
        result_entry = {
            "data_source": "mbpp_concat_interleaved",
            "prompt": trace_data["prompt"],
            "answer": trace_data["answer"],
            "reward_model": reward_model.to_dict(),
            "extra_info": {
                "split": "train",
                "index": i,
                "question": [trace_data["prompt_1"], trace_data["prompt_2"]],
                "answer": [trace_data["answer"]],
                "task_id_1": trace_data["task_id_1"],
                "task_id_2": trace_data["task_id_2"],
                "indices": trace_data["indices"]
            },
        }

        entries.append(result_entry)

    print(f"\n✅ Successfully generated {len(entries)} concat traces")
    return entries

def main():
    parser = argparse.ArgumentParser(description="Generate paired interleaved coding traces for MBPP")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B", help="Model name for generation")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for generated data")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of paired samples to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature")
    parser.add_argument("--max_tokens_per_turn", type=int, default=512, help="Max tokens per reasoning turn")
    parser.add_argument("--device_map", type=str, default="auto", help="Device mapping for model")
    parser.add_argument("--use_canonical", action="store_true", help="Use canonical solutions instead of generating new code solutions")
    args = parser.parse_args()

    print(f"\n🚀 CONCAT INTERLEAVED REASONING GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Dataset: MBPP")
    print(f"Paired Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Solution type: {'Canonical' if args.use_canonical else 'Generated'}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Initialize generator
    generator = InterleavedResponsesGenerator(
        model_name=args.model_name,
        device_map=args.device_map
    )

    # Generate concat dataset
    concat_data = generate_concat_dataset(
        generator=generator,
        num_samples=args.num_samples,
        use_canonical=args.use_canonical,
        max_new_tokens_per_turn=args.max_tokens_per_turn
    )

    if concat_data:
        # Save as parquet files
        filename_prefix = f"sft/mbpp_concat_interleaved"
        save_to_parquet(concat_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL for debugging
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(concat_data, jsonl_file)

        print(f"\n{'='*60}")
        print("CONCAT CODING DATASET GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: MBPP")
        print(f"Total paired problems processed: {len(concat_data)}")

        # Show example
        if concat_data:
            example = concat_data[0]
            print(f"\n📋 Example concat trace structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Task ID 1: {example['extra_info']['task_id_1']}")
            print(f"  Task ID 2: {example['extra_info']['task_id_2']}")
            print(f"  Question: {example['prompt'][:100]}...")
            print(f"  Reward Model Style: {example['reward_model']['style']}")
    else:
        print("❌ No concat data generated")

if __name__ == "__main__":
    main() 