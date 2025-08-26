#!/usr/bin/env python3
"""
Generate interleaved reasoning traces for BirdSQL prompts with a plan-first approach.
The model first generates a high-level plan, then implements each part of the plan.
The final answer uses the ground truth SQL command from the dataset.
Saves the reasoning traces to parquet files for SFT training.

Usage: python data_gen/generate_plan_birdsql.py --output_dir data --num_samples 50
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

# Prompt configurations for plan-first BirdSQL approach
PLAN_BIRDSQL_PROMPTS = {
    "plan_thought": "Start by thinking about a plan for solving this text-to-SQL problem. What are the key entities, relationships, and what approach should I take? Think about a high-level plan that you will implement in the next step. Begin with <think> and end with </think>. Only think about the plan / approach and do not include any other text in your reasoning, do not think about the SQL query implementation.",
    "plan_answer": """Now create a high-level plan for solving this text-to-SQL problem in <answer></answer> tags. Your plan should:

1. Break down the question into clear, sequential steps
2. Identify the key database tables and columns needed
3. Consider different SQL approaches (JOINs, subqueries, aggregations)
4. Outline the overall structure and flow of the SQL query

Provide a numbered list of high-level steps to solve this problem. Keep it simple and concise. Do not include any other text.
""",
    "implementation_thought": "Now think about implementing the plan to solve this text-to-SQL problem. Think about how to translate each step into actual SQL syntax? What tables to join, what conditions to apply, and what functions to use? Begin with <think> and end with </think>.",
    "implementation_answer": "Given the reasoning, now give me the final SQL query in <answer></answer> tags based on the reasoning. Just give me the SQL query, do not include any other text.",
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


def generate_plan_interleaved_birdsql_trace(
    generator: InterleavedResponsesGenerator,
    problem: Dict[str, Any],
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.7,
    use_ground_truth: bool = False,
) -> Dict[str, Any]:
    """Generate a complete interleaved BirdSQL trace following the plan-first pattern."""

    question_text = problem.get("question", "No question available")
    db_id = problem.get("db_id", "unknown")
    evidence = problem.get("evidence", "")
    ground_truth_sql = problem.get("SQL", "No SQL available")

    # Create the full prompt with database context
    full_prompt = f"""Database: {db_id}

Question: {question_text}"""

    if evidence and evidence.strip():
        full_prompt += f"""

External Knowledge Evidence:
{evidence}"""

    full_prompt += """

Please generate a SQL query to answer the question above. Output only the SQL query without any explanation."""

    with timer(f"Complete plan-first BirdSQL trace generation", verbose=True):
        # Initialize conversation
        messages = [{"role": "user", "content": full_prompt}]

        # Step 1: Think about the text-to-SQL problem
        messages.append({"role": "user", "content": PLAN_BIRDSQL_PROMPTS["plan_thought"]})

        thinking_response = generator.generate_thoughts(
            messages,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # Sanity check: ensure thinking response is > 20 tokens
        thinking_tokens = len(thinking_response.split())
        if thinking_tokens <= 20:
            print(f"⚠️  Warning: Thinking response too short ({thinking_tokens} tokens)")
            import ipdb; ipdb.set_trace()
        else:
            print(f"✅ Thinking response: {thinking_tokens} tokens")

        messages.append({"role": "user", "content": thinking_response})

        # Step 2: Generate high-level plan
        messages.append({"role": "user", "content": PLAN_BIRDSQL_PROMPTS["plan_answer"]})

        plan_response = generator.generate_answer(
            messages,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # Sanity check: ensure plan response is > 20 tokens
        plan_tokens = len(plan_response.split())
        if plan_tokens <= 20:
            print(f"⚠️  Warning: Plan response too short ({plan_tokens} tokens)")
            import ipdb; ipdb.set_trace()
        else:
            print(f"✅ Plan response: {plan_tokens} tokens")

        messages.append({"role": "user", "content": plan_response})

        # Step 3: Think about implementation
        messages.append(
            {"role": "user", "content": PLAN_BIRDSQL_PROMPTS["implementation_thought"]}
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
            print(f"⚠️  Warning: Implementation thinking response too short ({impl_thinking_tokens} tokens)")
            import ipdb; ipdb.set_trace()
        else:
            print(f"✅ Implementation thinking response: {impl_thinking_tokens} tokens")

        messages.append({"role": "user", "content": implementation_thinking_response})

        # Step 4: Generate SQL query or use ground truth
        if use_ground_truth:
            # Use ground truth SQL as the final answer
            implementation_response = f"<answer>{ground_truth_sql}</answer>"
            print("Using ground truth SQL for final answer")
        else:
            # Prompt the model to generate the SQL query
            messages.append({"role": "user", "content": PLAN_BIRDSQL_PROMPTS["implementation_answer"]})
            
            implementation_response = generator.generate_answer(
                messages,
                max_new_tokens=512,
                temperature=temperature,
                top_p=top_p,
            )[0]
            
            print("Model generated SQL for final answer")
    
        print("Question:")
        print(question_text)
        print("Database:", db_id)
        print("Plan:")
        print(plan_response)
        print("Implementation:")
        print(implementation_response)

        # Build complete interleaved response
        full_interleaved = f"{thinking_response}\n\n{plan_response}\n\n{implementation_thinking_response}\n\n{implementation_response}"

        return {
            "question": question_text,
            "db_id": db_id,
            "evidence": evidence,
            "full_response": full_interleaved,
            "ground_truth_sql": ground_truth_sql,
            "use_ground_truth": use_ground_truth,
        }


def generate_plan_birdsql_dataset(
    generator: InterleavedResponsesGenerator,
    num_samples: int = 50,
    max_new_tokens_per_turn: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.9,
    use_ground_truth: bool = False,
) -> List[Dict[str, Any]]:
    """Generate plan-first interleaved BirdSQL traces for multiple problems."""
    print(
        f"Generating plan-first BirdSQL dataset with {num_samples} samples from BirdSQL..."
    )
    
    if use_ground_truth:
        print("⚠️  Using ground truth SQL for final answers")
    else:
        print("✅ Model will generate SQL queries for final answers")

    # Load BirdSQL problems
    problems = load_dataset("birdsql/bird_mini_dev", split="mini_dev_sqlite")
    test_problems = problems.select(range(num_samples))

    entries = []

    for i, problem in enumerate(
        tqdm(test_problems, desc=f"Generating plan-first BirdSQL traces")
    ):
        print(f"\n  Problem {i + 1}/{len(test_problems)}")

        # Generate interleaved trace
        trace_data = generate_plan_interleaved_birdsql_trace(
            generator, problem, max_new_tokens_per_turn=max_new_tokens_per_turn, 
            temperature=temperature,
            top_p=top_p,
            use_ground_truth=use_ground_truth,
        )

        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth=[trace_data["ground_truth_sql"]], style="rule"
        )

        result_entry = {
            "data_source": "birdsql_plan_interleave",
            "prompt": trace_data["question"],
            "answer": trace_data["full_response"],
            "reward_model": reward_model.to_dict(),
            "system_instruction_type": "plan_first",
            "extra_info": {
                "split": "mini_dev_sqlite",
                "index": i,
                "question": [trace_data["question"]],
                "answer": [trace_data["full_response"]],
                "db_id": trace_data["db_id"],
                "evidence": trace_data["evidence"],
                "ground_truth_sql": trace_data["ground_truth_sql"],
                "use_ground_truth": trace_data["use_ground_truth"],
                "template_type": "plan_first",
                "interleaved": True,
                "task_id": f"birdsql_{i}",
            },
        }

        entries.append(result_entry)

    print(f"\n✅ Successfully generated {len(entries)} plan-first BirdSQL traces")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate plan-first interleaved BirdSQL traces for BirdSQL"
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
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p for generation"
    )
    parser.add_argument(
        "--use_ground_truth", action="store_true", 
        help="Use ground truth SQL instead of generating SQL queries"
    )
    args = parser.parse_args()

    print(f"🚀 BirdSQL PLAN-FIRST INTERLEAVED REASONING GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Initialize generator
    generator = InterleavedResponsesGenerator(
        model_name=args.model_name, device_map=args.device_map
    )

    birdsql_data = generate_plan_birdsql_dataset(
        generator,
        num_samples=args.num_samples,
        max_new_tokens_per_turn=args.max_tokens_per_turn,
        temperature=args.temperature,
        top_p=args.top_p,
        use_ground_truth=args.use_ground_truth,
    )

    if birdsql_data:
        # Save as parquet files
        filename_prefix = f"sft/birdsql_plan_interleave"
        save_to_parquet(birdsql_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(birdsql_data, jsonl_file)

        print(f"\n{'=' * 60}")
        print("PLAN-FIRST BirdSQL DATASET GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Dataset: BirdSQL (Plan-First Approach)")
        print(f"Total problems processed: {len(birdsql_data)}")

        # Show example
        if birdsql_data:
            example = birdsql_data[0]
            print(f"\n📋 Example trace structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Question: {example['prompt'][:20]}...")
            print(f"  Database: {example['extra_info']['db_id']}")
            print(f"  Reward Model Style: {example['reward_model']['style']}")
            print(f"  Template Type: {example['extra_info']['template_type']}")
            print(f"  Interleaved: {example['extra_info']['interleaved']}")
    else:
        print("❌ No data generated")


if __name__ == "__main__":
    main() 