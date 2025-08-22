#!/usr/bin/env python3
"""
Generate interleaved reasoning traces for BigCodeBench coding prompts using a two-stage approach.
Stage 1: Generate complete solution (plan + code)
Stage 2: Decompose into interleaved thought/answer format
Saves the reasoning traces to parquet files for SFT training.

Usage: python data_gen/generate_decompose_bcb.py --output_dir data --num_samples 50
"""

import argparse
import re
from typing import List, Dict, Any
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
from datasets import load_dataset
from interleave_generator import InterleavedResponsesGenerator
from helpers import StandardizedRewardModel, save_to_parquet, save_jsonl
from create_parquets import ADDITIONAL_INSTRUCTION


# Prompt for generating complete solution
COMPLETE_SOLUTION_PROMPT = """You are an expert coding assistant. Solve the following problem by thinking through it step by step, creating a detailed plan, and then implementing the solution.

Please provide your response in this exact format:

**THOUGHTS:**
[Your detailed thinking process about understanding the problem, requirements, constraints, and approach]

**PLAN:**
[Your detailed step-by-step plan here]

**IMPLEMENTATION:**
[Your complete code implementation here]

Make sure your thoughts are comprehensive, your plan is detailed, and your implementation follows the plan exactly."""


# Prompt for decomposing into interleaved format
DECOMPOSE_PROMPT = """You are an expert at breaking down solutions into interleaved thought/answer format.

Given a complete solution with thoughts, plan, and implementation, decompose it into alternating <think></think> and <answer></answer> chunks.

**Rules:**
1. Each <think> chunk should contain reasoning about what to do next
2. Each <answer> chunk should contain the actual thoughts, plan step, or code implementation
3. Start with a <think> about understanding the problem
4. Follow with an <answer> containing the initial thoughts
5. Continue with <think> about planning, then <answer> with the plan
6. Continue alternating between thinking about implementation and providing code
7. Make sure the thought/answer pairs are logically connected
8. Preserve the original thinking process from the complete solution

**Input Solution:**
{complete_solution}

**Decompose this into interleaved format:**"""


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


def generate_complete_solution(
    generator: InterleavedResponsesGenerator,
    problem: Dict[str, Any],
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> str:
    """Generate a complete solution (plan + implementation) for the problem."""

    task_id = problem.get("task_id", "Unknown")
    prompt = problem.get(
        "instruct_prompt", problem.get("question", "No prompt available")
    )

    full_prompt = f"{prompt}\n\n{COMPLETE_SOLUTION_PROMPT}"

    with timer(f"Complete solution generation for {task_id}", verbose=True):
        messages = [{"role": "user", "content": full_prompt}]

        # Generate complete response with thinking
        full_response = generator.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Check if the model finished thinking (has </think>)
        if "</think>" in full_response:
            # Extract the solution part (everything after </think>)
            answer = generator.extract_solution_from_response(full_response)
        else:
            # Model didn't finish thinking, generate answer separately
            messages.append({"role": "assistant", "content": full_response})
            messages.append(
                {
                    "role": "user",
                    "content": "Now provide your complete solution with plan and implementation.",
                }
            )

            answer = generator.generate_answer(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            # Combine the incomplete thinking with the answer
        return full_response, answer


def decompose_thoughts_and_answers(
    generator: InterleavedResponsesGenerator,
    thoughts: str,
    answer: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> str:
    """Decompose thoughts and answer into interleaved format by matching reasoning chains."""

    with timer(
        "Decomposing thoughts and answers into interleaved format", verbose=True
    ):
        # Step 1: Decompose the answer into plan and implementation parts
        decompose_answer_prompt = f"""Given this answer, break it down into the plan part and implementation part:

**ANSWER:**
{answer}

Please separate it into:

**PLAN:**
[The plan part]

**IMPLEMENTATION:**
[The implementation part]"""

        messages = [{"role": "user", "content": decompose_answer_prompt}]

        answer_decomposition = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract plan and implementation from the decomposition
        plan_match = re.search(
            r"\*\*PLAN:\*\*\s*(.*?)(?=\*\*IMPLEMENTATION:\*\*|\Z)",
            answer_decomposition,
            re.DOTALL,
        )
        implementation_match = re.search(
            r"\*\*IMPLEMENTATION:\*\*\s*(.*?)(?=\Z)", answer_decomposition, re.DOTALL
        )

        plan = plan_match.group(1).strip() if plan_match else ""
        implementation = (
            implementation_match.group(1).strip() if implementation_match else ""
        )

        # Step 2: Break up thoughts into plan and implementation parts
        thoughts_decomposition_prompt = f"""Given these thoughts, plan, and implementation, break up the thoughts into two parts that correspond to each.

**THOUGHTS:**
{thoughts}

**PLAN:**
{plan}

**IMPLEMENTATION:**
{implementation}

Break up the thoughts into two non-overlapping parts:

**PLAN THOUGHTS:**
[The part of the thoughts that led to the plan. Include the complete reasoning word for word. This should be distinct from implementation thoughts.]

**IMPLEMENTATION THOUGHTS:**
[The part of the thoughts that led to the implementation. Include the complete reasoning word for word. This should be distinct from plan thoughts and should not overlap with plan thoughts.]"""

        messages = [{"role": "user", "content": thoughts_decomposition_prompt}]

        thoughts_decomposition_response = generator.generate_answer(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract plan and implementation thoughts
        plan_thoughts_match = re.search(
            r"\*\*PLAN THOUGHTS:\*\*\s*(.*?)(?=\*\*IMPLEMENTATION THOUGHTS:\*\*|\Z)",
            thoughts_decomposition_response,
            re.DOTALL,
        )
        implementation_thoughts_match = re.search(
            r"\*\*IMPLEMENTATION THOUGHTS:\*\*\s*(.*?)(?=\Z)",
            thoughts_decomposition_response,
            re.DOTALL,
        )

        plan_reasoning = (
            plan_thoughts_match.group(1).strip() if plan_thoughts_match else ""
        )
        implementation_reasoning = (
            implementation_thoughts_match.group(1).strip()
            if implementation_thoughts_match
            else ""
        )

        # Step 3: Chain them together in interleaved format
        interleaved = f"""<think>{plan_reasoning}</think>
<answer>{plan}</answer>
<think>{implementation_reasoning}</think>
<answer>{implementation}</answer>"""

        components = {
            "plan": plan,
            "implementation": implementation,
            "plan_reasoning": plan_reasoning,
            "implementation_reasoning": implementation_reasoning,
        }
        return interleaved, components


def generate_decompose_coding_trace(
    generator: InterleavedResponsesGenerator,
    problem: Dict[str, Any],
    max_new_tokens: int = 2048,
    temperature: float = 0.2,
    top_p: float = 0.7,
) -> Dict[str, Any]:
    """Generate a complete interleaved coding trace using the two-stage approach."""

    task_id = problem.get("task_id", "Unknown")
    prompt = problem.get(
        "instruct_prompt", problem.get("question", "No prompt available")
    )

    print(f"\n  Generating complete solution...")
    # Stage 1: Generate complete solution
    thoughts, answer = generate_complete_solution(
        generator,
        problem,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    print(f"  Decomposing thoughts and answers into interleaved format...")
    # Stage 2: Decompose thoughts and answers into interleaved format
    interleaved_solution, components = decompose_thoughts_and_answers(
        generator,
        thoughts,
        answer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    import ipdb

    ipdb.set_trace()
    return {
        "task_id": task_id,
        "prompt": prompt,
        "complete_solution": components,
        "full_response": interleaved_solution,
    }


def generate_decompose_coding_dataset(
    generator: InterleavedResponsesGenerator,
    num_samples: int = 50,
    max_new_tokens: int = 2048,
) -> List[Dict[str, Any]]:
    """Generate decomposed interleaved coding traces for multiple problems."""
    print(
        f"Generating decompose coding dataset with {num_samples} samples from BigCodeBench..."
    )

    # Load BigCodeBench problems
    problems = load_dataset("bigcode/bigcodebench-hard", split="v0.1.4")
    problems = problems.select(range(num_samples))

    entries = []

    for i, problem in enumerate(
        tqdm(problems, desc=f"Generating decompose coding traces")
    ):
        print(
            f"\n  Problem {i + 1}/{len(problems)}: {problem.get('task_id', 'Unknown')}"
        )

        # Generate interleaved trace using two-stage approach
        trace_data = generate_decompose_coding_trace(
            generator, problem, max_new_tokens=max_new_tokens
        )

        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth=[problem["canonical_solution"]],
            style="code",
            unit_tests=[problem["test"]],
            libs=[problem["libs"]],
        )

        result_entry = {
            "data_source": "bcb_decompose_code_interleave",
            "prompt": trace_data["prompt"] + "\n" + ADDITIONAL_INSTRUCTION,
            "answer": trace_data["full_response"],
            "reward_model": reward_model.to_dict(),
            "system_instruction_type": "plan_first",
            "extra_info": {
                "split": "train",
                "index": i,
                "question": [trace_data["prompt"]],
                "answer": [trace_data["full_response"]],
                "complete_solution": [
                    trace_data["complete_solution"]
                ],  # Store original for reference
            },
        }

        entries.append(result_entry)

    print(f"\n✅ Successfully generated {len(entries)} decompose coding traces")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate decomposed interleaved coding traces for BigCodeBench"
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
        "--max_tokens",
        type=int,
        default=4096,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device mapping for model"
    )

    args = parser.parse_args()

    print(f"🚀 BIGCODEBENCH DECOMPOSE INTERLEAVED REASONING GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Initialize generator
    generator = InterleavedResponsesGenerator(
        model_name=args.model_name, device_map=args.device_map
    )

    coding_data = generate_decompose_coding_dataset(
        generator,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
    )

    if coding_data:
        # Save as parquet files
        filename_prefix = f"sft/bcb_decompose_code_interleave"
        save_to_parquet(coding_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(coding_data, jsonl_file)

        print(f"\n{'=' * 60}")
        print("DECOMPOSE CODING DATASET GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Dataset: BigCodeBench (Decompose Approach)")
        print(f"Total problems processed: {len(coding_data)}")

        # Show example
        if coding_data:
            example = coding_data[0]
            print(f"\n📋 Example trace structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Question: {example['prompt'][:100]}...")
            print(f"  Reward Model Style: {example['reward_model']['style']}")
    else:
        print("❌ No data generated")


if __name__ == "__main__":
    main()
