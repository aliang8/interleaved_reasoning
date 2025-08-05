#!/usr/bin/env python3
"""
Generate interleaved reasoning traces for MATH500 prompts with a plan-first approach.
The model first generates a high-level plan, then implements each part of the plan.
Saves the reasoning traces to parquet files for SFT training.

Usage: python data_gen/generate_plan_math500.py --output_dir data --num_samples 50
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

# Prompt configurations for plan-first math approach
PLAN_MATH_PROMPTS = {
    "plan_thought": "Let me start by thinking about this math problem. What are the key concepts, formulas, and what approach should I take? Begin with <think> and end with </think>.",
    "plan_answer": """Now create a comprehensive high-level plan for solving this math problem in <answer></answer> tags. Your plan should:

1. Break down the problem into clear, sequential steps
2. Identify the key mathematical concepts and formulas needed
3. Consider different solution approaches and choose the best one
4. Outline the overall strategy and methodology

Format your response as a numbered outline with bolded steps. Each step should be numbered and the main action/topic should be in bold, followed by a colon and explanation. For example:

1. **Understand the problem**: Identify what is being asked and what information is given...
2. **Identify relevant concepts**: Determine which mathematical concepts and formulas apply...
3. **Set up the solution**: Write down the equations or expressions needed...
4. **Solve step by step**: Work through the calculations systematically...
5. **Verify the answer**: Check if the result makes sense and satisfies the problem...

Provide a comprehensive step-by-step plan that will guide the solution.""",
    "implementation_thought": "Now let me think about implementing the plan I just created. How will I translate each step into actual mathematical work? What calculations, formulas, and reasoning do I need? Begin with <think> and end with </think>.",
    "implementation_answer": """Now implement the complete solution based on the plan in <answer></answer> tags. Follow the plan step by step and provide clear, well-explained mathematical work that implements each part of the plan.

Make sure your implementation:
- Follows the plan structure you outlined
- Shows all key calculations and steps
- Explains the reasoning behind each step
- Uses proper mathematical notation
- Handles all the concepts mentioned in the plan
- Provides a clear final answer""",
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


def generate_plan_interleaved_math_trace(
    generator: InterleavedResponsesGenerator,
    problem: Dict[str, Any],
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> Dict[str, Any]:
    """Generate a complete interleaved math trace following the plan-first pattern."""

    problem_text = problem.get("problem", "No problem available")

    with timer(f"Complete plan-first math trace generation", verbose=True):
        # Initialize conversation
        messages = [{"role": "user", "content": f"{problem_text}"}]

        # Step 1: Think about the math problem
        messages.append({"role": "user", "content": PLAN_MATH_PROMPTS["plan_thought"]})

        thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": thinking_response})

        # Step 2: Generate high-level plan
        messages.append({"role": "user", "content": PLAN_MATH_PROMPTS["plan_answer"]})

        plan_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens_per_turn * 2,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": plan_response})

        # Step 3: Think about implementation
        messages.append({"role": "user", "content": PLAN_MATH_PROMPTS["implementation_thought"]})

        implementation_thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=max_new_tokens_per_turn,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": implementation_thinking_response})

        # Step 4: Implement the solution based on the plan
        messages.append({"role": "user", "content": PLAN_MATH_PROMPTS["implementation_answer"]})

        implementation_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens_per_turn * 2,
            temperature=temperature,
            top_p=top_p,
        )

        messages.append({"role": "assistant", "content": implementation_response})

        # Build complete interleaved response
        full_interleaved = f"{thinking_response}\n\n{plan_response}\n\n{implementation_thinking_response}\n\n{implementation_response}"

        return {
            "problem": problem_text,
            "full_response": full_interleaved,
        }


def generate_plan_math_dataset(
    generator: InterleavedResponsesGenerator,
    num_samples: int = 50,
    max_new_tokens_per_turn: int = 512,
) -> List[Dict[str, Any]]:
    """Generate plan-first interleaved math traces for multiple problems."""
    print(f"Generating plan-first math dataset with {num_samples} samples from MATH500...")

    # Load MATH500 problems
    problems = load_dataset("HuggingFaceH4/MATH-500")
    test_problems = problems["test"]
    test_problems = test_problems.select(range(num_samples))

    entries = []

    for i, problem in enumerate(tqdm(test_problems, desc=f"Generating plan-first math traces")):

        print(f"\n  Problem {i+1}/{len(test_problems)}")

        # Generate interleaved trace
        trace_data = generate_plan_interleaved_math_trace(
            generator, problem, max_new_tokens_per_turn=max_new_tokens_per_turn
        )

        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth=[problem["answer"]], style="rule"
        )

        result_entry = {
            "data_source": "math500_plan_interleave",
            "prompt": trace_data["problem"],
            "answer": trace_data["full_response"],
            "reward_model": reward_model.to_dict(),
            "system_instruction_type": "plan_first",
            "extra_info": {
                "split": "test",
                "index": i,
                "question": [trace_data["problem"]],
                "answer": [trace_data["full_response"]],
            },
        }

        entries.append(result_entry)

    print(f"\n✅ Successfully generated {len(entries)} plan-first math traces")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate plan-first interleaved math traces for MATH500"
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
        default=512,
        help="Max tokens per reasoning turn",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device mapping for model"
    )

    args = parser.parse_args()

    print(f"🚀 MATH500 PLAN-FIRST INTERLEAVED REASONING GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Initialize generator
    generator = InterleavedResponsesGenerator(
        model_name=args.model_name, device_map=args.device_map
    )

    math_data = generate_plan_math_dataset(
        generator,
        num_samples=args.num_samples,
        max_new_tokens_per_turn=args.max_tokens_per_turn,
    )

    if math_data:
        # Save as parquet files
        filename_prefix = f"sft/math500_plan_interleave"
        save_to_parquet(math_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(math_data, jsonl_file)

        print(f"\n{'='*60}")
        print("PLAN-FIRST MATH DATASET GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: MATH500 (Plan-First Approach)")
        print(f"Total problems processed: {len(math_data)}")

        # Show example
        if math_data:
            example = math_data[0]
            print(f"\n📋 Example trace structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Question: {example['prompt'][:100]}...")
            print(f"  Reward Model Style: {example['reward_model']['style']}")
    else:
        print("❌ No data generated")


if __name__ == "__main__":
    main() 