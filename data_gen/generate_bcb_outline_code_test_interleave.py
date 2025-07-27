#!/usr/bin/env python3
"""
Generate interleaved reasoning traces for BigCodeBench coding prompts.
Saves the reasoning traces to parquet files for SFT training.

Usage: python generate_bcb_outline_code_test_interleave.py --output_dir bigcodebench_data --num_samples 50
"""

import argparse
from typing import List, Dict, Any
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
from datasets import load_dataset
from interleave_generator import InterleavedResponsesGenerator
from helpers import StandardizedRewardModel, save_to_parquet, save_jsonl

# Prompt configurations for each step
CODING_PROMPTS = {
    "outline_thought": "Let me start by thinking about this problem. What are the requirements, constraints, and what approach should I take? Begin with <think> and end with </think>.",
    "outline_answer": """Now provide a clear description of your solution approach in <answer></answer> tags. Format your response as a numbered outline with bolded steps. Each step should be numbered and the main action/topic should be in bold, followed by a colon and explanation. For example:

1. **Remove URLs from the text**: Use regular expressions to identify and remove all URLs...
2. **Check for remaining words**: After removing URLs, verify if any words are left...
3. **Generate the word cloud**: Using the WordCloud class...
4. **Return the result**: Return the final word cloud object...

Provide a comprehensive step-by-step breakdown of your solution approach.""",
    "code_thought": "Now let me think about the specific implementation details, data structures, and code structure I'll need. Begin with <think> and end with </think>.",
    "code_answer": "Now implement the solution in <answer></answer> tags. Provide clean, well-commented code.",
    "unit_test_thought": "Now let me think about comprehensive test cases to validate my solution. What edge cases, normal cases, and boundary conditions should I test? I need to create unittest test cases that import from task_func. Begin with <think> and end with </think>.",
    "unit_test_answer": """Now create exactly 4 test cases in <answer></answer> tags. Use this exact format:

import unittest
from task_func import task_func

class Test(unittest.TestCase):
    def test_case_1(self):
        # Test case 1 description
        result = task_func(...)
        self.assertEqual(result, expected_value)
        self.assertEqual(result, expected_value)
    
    def test_case_2(self):
        # Test case 2 description
        result = task_func(...)
        self.assertEqual(result, expected_value)
    
    def test_case_3(self):
        # Test case 3 description
        result = task_func(...)
        self.assertEqual(result, expected_value)
    
    def test_case_4(self):
        # Test case 4 description
        result = task_func(...)
        self.assertEqual(result, expected_value)

""",
}


@contextmanager
def timer(name: str, verbose: bool = True):
    """Context manager for timing operations."""
    start_time = time.time()
    if verbose:
        print(f"    Starting {name}...")
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print(f"    {name} completed in {duration:.2f}s")


def generate_interleaved_coding_trace(
    generator: InterleavedResponsesGenerator,
    problem: Dict[str, Any],
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> Dict[str, Any]:
    """Generate a complete interleaved coding trace following the 6-step pattern."""

    task_id = problem.get("task_id", "Unknown")
    prompt = problem.get(
        "instruct_prompt", problem.get("question", "No prompt available")
    )

    with timer(f"Complete trace generation for {task_id}", verbose=True):
        # Initialize conversation
        messages = [{"role": "user", "content": f"{prompt}"}]

        # Step 1: Think about the prompt/problem
        messages.append({"role": "user", "content": CODING_PROMPTS["outline_thought"]})

        thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": thinking_response})

        # Step 2: Provide solution description
        messages.append({"role": "user", "content": CODING_PROMPTS["outline_answer"]})

        description_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": description_response})

        # Step 3: Think about the code implementation
        messages.append({"role": "user", "content": CODING_PROMPTS["code_thought"]})

        code_thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": code_thinking_response})

        # Step 4: Write the actual code
        messages.append({"role": "user", "content": CODING_PROMPTS["code_answer"]})

        code_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens_per_turn * 2,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": code_response})

        # Step 5: Think about test cases
        messages.append(
            {"role": "user", "content": CODING_PROMPTS["unit_test_thought"]}
        )

        test_thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": test_thinking_response})

        # Step 6: Create test cases
        messages.append({"role": "user", "content": CODING_PROMPTS["unit_test_answer"]})

        test_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": test_response})

        # Build complete interleaved response
        full_interleaved = f"{thinking_response}\n\n{description_response}\n\n{code_thinking_response}\n\n{code_response}\n\n{test_thinking_response}\n\n{test_response}"

        return {
            "task_id": task_id,
            "prompt": prompt,
            "full_response": full_interleaved,
        }


def generate_coding_dataset(
    generator: InterleavedResponsesGenerator,
    num_samples: int = 50,
    max_new_tokens_per_turn: int = 512,
) -> List[Dict[str, Any]]:
    """Generate interleaved coding traces for multiple problems."""
    print(f"Generating coding dataset with {num_samples} samples from BigCodeBench...")

    # Load BigCodeBench problems
    problems = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")
    problems = problems.select(range(num_samples))

    entries = []

    for i, problem in enumerate(tqdm(problems, desc=f"Generating coding traces")):

        print(f"\n  Problem {i+1}/{len(problems)}: {problem.get('task_id', 'Unknown')}")

        # Generate interleaved trace
        trace_data = generate_interleaved_coding_trace(
            generator, problem, max_new_tokens_per_turn=max_new_tokens_per_turn
        )

        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth=problem["canonical_solution"],
            style="code",
            unit_tests=[problem["test"]],
            libs=[problem["libs"]],
        )

        result_entry = {
            "data_source": "bcb_outline_code_test_interleave",
            "prompt": trace_data["prompt"],
            "answer": trace_data["full_response"],
            "reward_model": reward_model.to_dict(),
            "extra_info": {
                "split": "train",
                "index": i,
                "question": [trace_data["prompt"]],
                "answer": [trace_data["full_response"]],
            },
        }

        entries.append(result_entry)

    print(f"\n✅ Successfully generated {len(entries)} coding traces")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate interleaved coding traces for BigCodeBench"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_samples", type=int, default=25, help="Number of problems to process"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens_per_turn",
        type=int,
        default=256,
        help="Max tokens per reasoning turn",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device mapping for model"
    )

    args = parser.parse_args()

    print(f"🚀 BIGCODEBENCH INTERLEAVED REASONING GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Initialize generator
    generator = InterleavedResponsesGenerator(
        model_name=args.model_name, device_map=args.device_map
    )

    coding_data = generate_coding_dataset(
        generator,
        num_samples=args.num_samples,
        max_new_tokens_per_turn=args.max_tokens_per_turn,
    )

    if coding_data:
        # Save as parquet files
        filename_prefix = f"sft/bcb_outline_code_test_interleave"
        save_to_parquet(coding_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL for debuggin
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(coding_data, jsonl_file)

        print(f"\n{'='*60}")
        print("CODING DATASET GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: BigCodeBench")
        print(f"Total problems processed: {len(coding_data)}")

        # Show example
        if coding_data:
            example = coding_data[0]
            print(f"\n📋 Example trace structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Task ID: {example['extra_info']['task_id']}")
            print(f"  Question: {example['prompt'][:100]}...")
            print(f"  Steps: {example['extra_info']['step_labels']}")
            print(
                f"  Total reasoning steps: {example['extra_info']['reasoning_steps']}"
            )
            print(f"  Reward Model Style: {example['reward_model']['style']}")
    else:
        print("❌ No data generated")


if __name__ == "__main__":
    main()
