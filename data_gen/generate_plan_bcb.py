#!/usr/bin/env python3
"""
Generate interleaved reasoning traces for BigCodeBench coding prompts with a plan-first approach.
The model first generates a high-level plan, then implements each part of the plan.
Saves the reasoning traces to parquet files for SFT training.

Usage: python data_gen/generate_plan_bcb.py --output_dir data --num_samples 50
"""

import argparse
from sys import implementation
from typing import List, Dict, Any
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
from datasets import load_dataset
from interleave_generator import InterleavedResponsesGenerator
from helpers import StandardizedRewardModel, save_to_parquet, save_jsonl
from create_parquets import ADDITIONAL_INSTRUCTION


# Prompt configurations for plan-first approach
PLAN_CODING_PROMPTS = {
    "plan_thought": "Start by thinking about this problem. What are the requirements, constraints, and what approach should I take? Begin with <think> and end with </think>. Only think about the plan / approach for the problem.",
    "plan_answer": """Now create a high-level plan for solving this problem in <answer></answer> tags. Your plan should:

1. Break down the problem into clear, sequential steps
2. Identify the key components and functions needed
3. Consider edge cases and error handling
4. Outline the overall structure and flow

Provide a numbered list of high-level steps to solve this problem. Keep it simple and concise. Do not include any other text.""",
    "implementation_thought": "Think about implementing the plan to solve this problem. Think about how to translate each step into actual code? What data structures, algorithms, and implementation details do you need? Begin with <think> and end with </think>.",
    "implementation_answer": """Now implement the complete solution based on the plan and the reasoning. Follow the plan step by step and provide clean, well-commented code that implements each part of the plan. Make sure the code is well-formatted. Wrap the code in <answer></answer> tags.""",
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


def generate_plan_interleaved_coding_trace(
    generator: InterleavedResponsesGenerator,
    problem: Dict[str, Any],
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> Dict[str, Any]:
    """Generate a complete interleaved coding trace following the plan-first pattern."""

    task_id = problem.get("task_id", "Unknown")
    prompt = problem.get(
        "instruct_prompt", problem.get("question", "No prompt available")
    )

    with timer(f"Complete plan-first trace generation for {task_id}", verbose=True):
        # Initialize conversation
        messages = [{"role": "user", "content": f"{prompt}"}]

        # Step 1: Think about the prompt/problem
        messages.append(
            {"role": "user", "content": PLAN_CODING_PROMPTS["plan_thought"]}
        )

        thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # Sanity check: ensure thinking response is > 20 tokens
        thinking_tokens = len(thinking_response.split())
        if thinking_tokens <= 20:
            print(f"⚠️  Warning: Thinking response too short ({thinking_tokens} tokens) for {task_id}")
            import ipdb; ipdb.set_trace()
        else:
            print(f"✅ Thinking response: {thinking_tokens} tokens")

        messages.append({"role": "user", "content": thinking_response})

        # Step 2: Generate high-level plan
        messages.append({"role": "user", "content": PLAN_CODING_PROMPTS["plan_answer"]})

        plan_response = generator.generate_answer(
            messages,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # Sanity check: ensure plan response is > 20 tokens
        plan_tokens = len(plan_response.split())
        if plan_tokens <= 20:
            print(f"⚠️  Warning: Plan response too short ({plan_tokens} tokens) for {task_id}")
            import ipdb; ipdb.set_trace()
        else:
            print(f"✅ Plan response: {plan_tokens} tokens")

        messages.append({"role": "user", "content": plan_response})

        # Step 3: Think about implementation
        messages.append(
            {"role": "user", "content": PLAN_CODING_PROMPTS["implementation_thought"]}
        )

        implementation_thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=max_new_tokens_per_turn * 2,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # Sanity check: ensure implementation thinking response is > 20 tokens
        impl_thinking_tokens = len(implementation_thinking_response.split())
        if impl_thinking_tokens <= 20:
            print(f"⚠️  Warning: Implementation thinking response too short ({impl_thinking_tokens} tokens) for {task_id}")
            import ipdb; ipdb.set_trace()
        else:
            print(f"✅ Implementation thinking response: {impl_thinking_tokens} tokens")

        messages.append(
            {"role": "user", "content": implementation_thinking_response}
        )

        # Step 4: Implement the solution based on the plan
        messages.append(
            {"role": "user", "content": PLAN_CODING_PROMPTS["implementation_answer"]}
        )

        implementation_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # make sure the implementation response is wrapped in <answer></answer> tags
        if not implementation_response.startswith("<answer>"):
            implementation_response = "<answer>" + implementation_response + "</answer>"
        
        implementation_response = implementation_response.replace("```python", "")
        implementation_response = implementation_response.replace("```", "")
        code = implementation_response.split("<answer>")[1].split("</answer>")[0]
        implementation_response = f"<answer>```python\n{code}\n```</answer>"

        messages.append({"role": "user", "content": implementation_response})
        print(f"✅ Implementation response: {len(implementation_response.split())} tokens")


        # Build complete interleaved response
        full_interleaved = f"{thinking_response}\n\n{plan_response}\n\n{implementation_thinking_response}\n\n{implementation_response}"

        print("Problem:")
        print(prompt)
        print("Plan:")
        print(plan_response)
        print("Implementation:")
        print(implementation_response)

        return {
            "task_id": task_id,
            "prompt": prompt,
            "full_response": full_interleaved,
        }


def generate_plan_coding_dataset(
    generator: InterleavedResponsesGenerator,
    num_samples: int = 50,
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.6,
) -> List[Dict[str, Any]]:
    """Generate plan-first interleaved coding traces for multiple problems."""
    print(
        f"Generating plan-first coding dataset with {num_samples} samples from BigCodeBench..."
    )

    # Load BigCodeBench problems
    problems = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")
    problems = problems.select(range(num_samples))

    entries = []

    for i, problem in enumerate(
        tqdm(problems, desc=f"Generating plan-first coding traces")
    ):
        print(
            f"\n  Problem {i + 1}/{len(problems)}: {problem.get('task_id', 'Unknown')}"
        )

        # Generate interleaved trace
        trace_data = generate_plan_interleaved_coding_trace(
            generator, problem, max_new_tokens_per_turn=max_new_tokens_per_turn, temperature=temperature        
        )

        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth=[problem.get("canonical_solution", "")],
            style="code",
            unit_tests=[problem.get("test", "")],
            libs=[problem.get("libs", [])],
        )

        # Standardized entry structure (aligned with math plan-first generator)
        result_entry = {
            "data_source": "bcb_plan_code_interleave",
            "prompt": trace_data["prompt"],
            "answer": trace_data["full_response"],
            "reward_model": reward_model.to_dict(),
            "system_instruction_type": "plan_first",
            "extra_info": {
                "split": "train",
                "index": i,
                "question": [trace_data["prompt"]],
                "answer": [trace_data["full_response"]],
                "template_type": trace_data.get("template_type", "plan_first"),
                "interleaved": trace_data.get("interleaved", True),
                "task_id": trace_data.get("task_id", "Unknown"),
            },
        }

        entries.append(result_entry)

    print(f"\n✅ Successfully generated {len(entries)} plan-first coding traces")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate plan-first interleaved coding traces for BigCodeBench"
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
        default=1024,
        help="Max tokens per reasoning turn",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device mapping for model"
    )

    args = parser.parse_args()

    print(f"🚀 BIGCODEBENCH PLAN-FIRST INTERLEAVED REASONING GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Initialize generator
    generator = InterleavedResponsesGenerator(
        model_name=args.model_name, device_map=args.device_map
    )

    coding_data = generate_plan_coding_dataset(
        generator,
        num_samples=args.num_samples,
        max_new_tokens_per_turn=args.max_tokens_per_turn,
        temperature=args.temperature,
    )

    if coding_data:
        # Save as parquet files
        filename_prefix = f"sft/bcb_plan_code_interleave"
        save_to_parquet(coding_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(coding_data, jsonl_file)

        print(f"\n{'=' * 60}")
        print("PLAN-FIRST CODING DATASET GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Dataset: BigCodeBench (Plan-First Approach)")
        print(f"Total problems processed: {len(coding_data)}")

        # Show example
        if coding_data:
            example = coding_data[0]
            print(f"\n📋 Example trace structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Question: {example['prompt'][:20]}...")
            print(f"  Reward Model Style: {example['reward_model']['style']}")
            print(f"  Template Type: {example['extra_info'].get('template_type')}")
    else:
        print("❌ No data generated")


if __name__ == "__main__":
    main()
