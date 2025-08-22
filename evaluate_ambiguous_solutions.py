#!/usr/bin/env python3
"""
Ambiguous Prompts Generation Script

This script:
1. Loads ambiguous prompt entries from a JSONL file
2. Initializes the model with ActorRolloutRefWorker using the specified rollout (default: vllm_autorater_rollout)
3. Generates responses for each ambiguous prompt (no evaluation of canonical solutions here)
4. Saves results to JSONL/JSON and can generate an HTML visualization using the MBPP visualizer

Run with:
  python evaluate_ambiguous_solutions.py \
    --input_file data/ambiguous.jsonl \
    --model_path Qwen/Qwen3-8B \
    --rollout_name vllm_autorater_rollout \
    --template_type default \
    --batch_size 25 \
    --make_html
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from omegaconf import DictConfig

# Set environment variables for distributed setup (single GPU)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from actor_config import ACTOR_ROLLOUT_CONFIG

# Create prompts DataProto (same utility used in mbpp_evaluation.py)
from simple_vllm_rollout_demo import create_prompts_dataproto

# For optional interleaved parsing/visualization helpers
from helpers import parse_interleaved_components

# MBPP HTML visualization
from create_mbpp_html import create_mbpp_html_visualization
from mbpp_evaluation import evaluate_code_against_tests, extract_code_from_response


def load_ambiguous_data(input_file: str) -> List[Dict[str, Any]]:
    """
    Load ambiguous prompts data from JSONL file.
    """
    print(f"Loading ambiguous data from {input_file}...")
    entries: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    print(f"✅ Loaded {len(entries)} entries")
    return entries


def batched(iterable: List[Any], n: int) -> List[List[Any]]:
    """Yield successive n-sized batches from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses for ambiguous prompts using vLLM rollouts"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to JSONL file with ambiguous entries",
    )
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen3-8B", help="Model path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ambiguous_generation",
        help="Output directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=25, help="Batch size for generation"
    )
    parser.add_argument(
        "--template_type",
        type=str,
        default="default",
        help="Template type for chat formatting",
    )
    parser.add_argument(
        "--rollout_name",
        type=str,
        default="vllm_autorater_rollout",
        help="Rollout class name to use",
    )
    parser.add_argument(
        "--make_html",
        action="store_true",
        help="Also generate an HTML visualization (MBPP style)",
    )
    parser.add_argument(
        "--html_file", type=str, default=None, help="Optional explicit HTML output path"
    )
    parser.add_argument(
        "--disable_mbpp_prefix",
        action="store_true",
        help="Disable MBPP-style code prefix for prompts",
    )
    args = parser.parse_args()

    print("=== Ambiguous Prompts Generation ===\n")
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return False

    # Init distributed (single GPU fallback)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        print(f"🚀 Initializing distributed training with {world_size} GPUs")
        try:
            import torch.distributed as dist

            dist.init_process_group(backend="nccl", init_method="env://")
            torch.cuda.set_device(local_rank)
        except Exception as e:
            print(f"❌ Failed to initialize distributed training: {e}")
            world_size, rank, local_rank = 1, 0, 0

    if not torch.cuda.is_available():
        print("❌ CUDA not available! This script requires a GPU.")
        return False

    # Load data
    entries = load_ambiguous_data(args.input_file)
    if not entries:
        print("❌ No entries to process")
        return False

    # Configure rollout
    config = ACTOR_ROLLOUT_CONFIG.copy()
    config["model"]["path"] = args.model_path
    config["model"]["tokenizer_path"] = args.model_path
    config["rollout"]["name"] = args.rollout_name
    config["rollout"]["template_type"] = args.template_type

    if world_size > 1:
        cuda_device = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(cuda_device)
        config["actor"]["fsdp_config"]["fsdp_size"] = world_size
        config["rollout"]["tensor_model_parallel_size"] = world_size

    print(f"Rollout: {config['rollout']['name']}")
    print(f"Template: {config['rollout']['template_type']}")
    print(
        f"Prompt length: {config['rollout']['prompt_length']}, Response length: {config['rollout']['response_length']}"
    )

    # Initialize worker/model
    print("\nInitializing ActorRolloutRefWorker...")
    worker = ActorRolloutRefWorker(config=DictConfig(config), role="rollout")
    worker.init_model()
    tokenizer = worker.tokenizer

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    print(
        f"\n🚀 Generating for {len(entries)} entries in batches of {args.batch_size}..."
    )

    for batch_idx, batch_entries in enumerate(batched(entries, args.batch_size)):
        # Build prompts with optional MBPP-style prefix
        mbpp_prefix = (
            "You should write self-contained code starting with:\n"
            "def task_func(arg1, arg2, ...):\n"
        )
        prompts: List[str] = []
        explicit_tasks: List[str] = []
        for e in batch_entries:
            # Prefer 'question' if present, otherwise fallback to 'prompt'
            q = e.get("question") or e.get("prompt") or ""
            content = q if args.disable_mbpp_prefix else f"{mbpp_prefix}{q}"
            prompts.append(content)
            explicit_tasks.append(e.get("extra_info", {}).get("explicit_task", ""))

        # Build prompts DataProto
        prompts_dataproto = create_prompts_dataproto(
            tokenizer=tokenizer,
            questions=prompts,
            max_prompt_length=ACTOR_ROLLOUT_CONFIG["rollout"]["prompt_length"],
            template_type=config["rollout"]["template_type"],
        )
        # Add explicit tasks to meta_info for the model
        prompts_dataproto.meta_info["explicit_tasks"] = explicit_tasks

        # Generate
        torch.cuda.empty_cache()
        output = worker.generate_sequences(prompts_dataproto)

        # Process outputs
        for i, entry in enumerate(batch_entries):
            response_tokens = output.batch["responses"][i]
            response_tokens = response_tokens[response_tokens != tokenizer.pad_token_id]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            interleaved_components = parse_interleaved_components(response_text)

            question_text = entry.get("question") or entry.get("prompt") or ""

            # Extract tests and canonical solution if available
            test_list = entry.get("extra_info", {}).get("generated_tests", [])
            if not isinstance(test_list, list):
                test_list = []

            if not test_list:
                reward_model = entry.get("reward_model", {})
                if reward_model:
                    unit_tests_raw = reward_model.get("unit_tests", [])
                    if isinstance(unit_tests_raw, list) and unit_tests_raw:
                        unit_tests_raw = unit_tests_raw[0]
                    if isinstance(unit_tests_raw, str):
                        test_list = [
                            line.strip()
                            for line in unit_tests_raw.split("\n")
                            if line.strip() and "assert" in line
                        ]
                    elif isinstance(unit_tests_raw, list):
                        test_list = unit_tests_raw

            # Normalize tests to only assert statements
            test_list = [t for t in test_list if isinstance(t, str) and "assert" in t]

            # Extract model-generated code (last interleaved answer / code blocks)
            generated_code = extract_code_from_response(
                response_text, config["rollout"]["template_type"]
            )

            # Evaluate the model-generated code against tests (MBPP-style)
            evaluation = evaluate_code_against_tests(
                code=generated_code,
                test_list=test_list,
                entry_point="task_func",
                test_imports=[],
            )

            result = {
                # Fields used by MBPP HTML
                "problem_id": entry.get("extra_info", {}).get("index", 0),
                "prompt": question_text,
                "generated_code": generated_code,
                "full_response": response_text,
                "evaluation": evaluation,
                "test_list": test_list,
                "entry_point": "task_func",
                "interleaved_components": interleaved_components,
                "template_type": config["rollout"]["template_type"],
                # Extra fields for context
                "question": question_text,
                "response": response_text,
                "num_tokens": int(response_tokens.shape[0])
                if hasattr(response_tokens, "shape")
                else 0,
                "original_intent": entry.get("extra_info", {}).get("explicit_task", ""),
            }
            all_results.append(result)

        print(f"Batch {batch_idx + 1}: processed {len(batch_entries)} entries")

    # Save outputs
    stem = Path(args.input_file).stem
    if world_size > 1:
        jsonl_path = output_dir / f"{stem}_{config['rollout']['name']}_gpu{rank}.jsonl"
        json_path = output_dir / f"{stem}_{config['rollout']['name']}_gpu{rank}.json"
    else:
        jsonl_path = output_dir / f"{stem}_{config['rollout']['name']}.jsonl"
        json_path = output_dir / f"{stem}_{config['rollout']['name']}.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n✅ Saved:")
    print(f"  JSONL: {jsonl_path}")
    print(f"  JSON:  {json_path}")

    # Optionally create MBPP-style HTML visualization
    if args.make_html:
        if args.html_file:
            html_path = Path(args.html_file)
        else:
            if world_size > 1:
                html_path = (
                    output_dir / f"{stem}_{config['rollout']['name']}_gpu{rank}.html"
                )
            else:
                html_path = output_dir / f"{stem}_{config['rollout']['name']}.html"
        print("\n🎨 Creating MBPP HTML visualization...")
        create_mbpp_html_visualization(all_results, str(html_path))
        print(f"🎉 HTML saved to: {html_path}")

    # Cleanup distributed
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()

    print("\n✨ Generation completed successfully!")
    return True


if __name__ == "__main__":
    main()
