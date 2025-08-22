#!/usr/bin/env python3
"""
Simplified example of using ActorRolloutRefWorker for vLLM rollout.

This script shows the basic pattern for:
1. Setting up the configuration following FSDP workers pattern
2. Creating input data in the correct format
3. Calling generate_sequences through the worker
4. Processing the output

Run with: python simple_vllm_rollout_demo.py
"""

import os
import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from omegaconf import DictConfig
from tensordict import TensorDict

# Set environment variables for distributed setup (single GPU)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
# os.environ.setdefault("MASTER_PORT", "12355")

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto
from verl.utils.templates import format_system_message


def create_prompts_dataproto(
    tokenizer,
    questions,
    max_prompt_length=1024,
    template_type="default",
    explicit_tasks=None,
    enable_thinking=True,
):
    """
    Create a DataProto object with prompts formatted for ActorRolloutRefWorker.

    Args:
        tokenizer: HuggingFace tokenizer
        questions: List of question strings
        max_prompt_length: Maximum length for prompt padding
        template_type: Type of system template to use
        explicit_tasks: Optional list of explicit task descriptions (stored in non_tensor_batch)
        enable_thinking: Whether to enable thinking in the chat template (default: True)

    Returns:
        DataProto object ready for generate_sequences()
    """
    batch_size = len(questions)

    # Apply chat template to format questions properly for instruction model
    formatted_prompts = []
    for question in questions:
        # Format as a conversation with configurable system template
        messages = [
            format_system_message(template_type),
            {"role": "user", "content": question},
        ]

        # Apply the chat template
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            formatted_prompts.append(formatted_prompt)

            # Debug output for first few prompts
            if len(formatted_prompts) <= 2:
                print(f"    Formatted prompt {len(formatted_prompts)}:")
                print(f"    {formatted_prompt[:200]}...")
                print()

        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
            # Fallback to simple format
            formatted_prompt = (
                f"System: {messages[0]['content']}\n\nUser: {question}\n\nAssistant:"
            )
            formatted_prompts.append(formatted_prompt)

    # Tokenize the formatted prompts with left padding (vLLM requirement)
    tokenizer.padding_side = "left"

    encoded = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Create position_ids - this accounts for left padding
    position_ids = torch.zeros_like(input_ids)
    for i in range(batch_size):
        non_pad_mask = input_ids[i] != tokenizer.pad_token_id
        if non_pad_mask.any():
            first_token_pos = non_pad_mask.nonzero(as_tuple=False)[0][0]
            seq_len = max_prompt_length - first_token_pos
            position_ids[i, first_token_pos:] = torch.arange(seq_len)

    # Create batch TensorDict
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    # Meta info required by ActorRolloutRefWorker
    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "validate": False,
    }

    meta_info["original_prompt"] = questions

    if explicit_tasks:
        meta_info["explicit_tasks"] = explicit_tasks

    return DataProto(batch=batch, meta_info=meta_info)


def load_validation_data(file_paths: List[str]) -> tuple[List[str], List[str]]:
    """
    Load validation prompts from files.

    Returns:
        Tuple of (questions, explicit_tasks) where explicit_tasks may be None if not available
    """
    questions = []
    explicit_tasks = []

    for file_path in file_paths:
        path = Path(file_path)
        print(f"    Loading file: {file_path}")

        try:
            if path.suffix.lower() == ".txt":
                # Load from text file (one prompt per line)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                file_prompts = []
                for line in lines:
                    line = line.strip()
                    if line:
                        file_prompts.append(line)

                questions.extend(file_prompts)
                print(f"    ✓ Loaded {len(file_prompts)} prompts from text file")

            elif path.suffix.lower() == ".jsonl":
                # Load from JSONL file
                with open(path, "r", encoding="utf-8") as f:
                    file_prompts = []
                    for line in f:
                        data = json.loads(line.strip())
                        if "prompt" in data:
                            file_prompts.append(data["prompt"])
                        elif "question" in data:
                            file_prompts.append(data["question"])
                        elif "content" in data:
                            file_prompts.append(data["content"])

                questions.extend(file_prompts)
                print(f"    ✓ Loaded {len(file_prompts)} prompts from JSONL file")

            elif path.suffix.lower() == ".parquet":
                # Load from parquet file
                import pandas as pd

                df = pd.read_parquet(path)
                print(f"    Columns: {list(df.columns)}")

                # Check if we have explicit task information
                if "question" in df.columns and "extra_info" in df.columns:
                    # Extract questions and explicit tasks from the extra_info column
                    file_questions = []
                    file_explicit_tasks = []

                    for _, row in df.iterrows():
                        question = row["question"]
                        extra_info = row["extra_info"]

                        if pd.notna(question) and question.strip():
                            file_questions.append(str(question).strip())

                            # Try to extract explicit_task from extra_info
                            if (
                                isinstance(extra_info, dict)
                                and "explicit_task" in extra_info
                            ):
                                explicit_task = extra_info["explicit_task"]
                                if pd.notna(explicit_task) and explicit_task.strip():
                                    file_explicit_tasks.append(
                                        str(explicit_task).strip()
                                    )
                                else:
                                    file_explicit_tasks.append("")
                            else:
                                file_explicit_tasks.append("")

                    questions.extend(file_questions)
                    explicit_tasks.extend(file_explicit_tasks)
                    print(
                        f"    ✓ Loaded {len(file_questions)} prompts with explicit tasks from parquet file"
                    )

                elif "question" in df.columns:
                    # Just questions, no explicit tasks
                    file_prompts = df["question"].tolist()
                    file_prompts = [
                        str(p).strip()
                        for p in file_prompts
                        if pd.notna(p) and str(p).strip()
                    ]
                    questions.extend(file_prompts)
                    print(f"    ✓ Loaded {len(file_prompts)} prompts from parquet file")

                elif "prompt" in df.columns:
                    # Standard prompt column
                    file_prompts = df["prompt"].tolist()
                    file_prompts = [
                        str(p).strip()
                        for p in file_prompts
                        if pd.notna(p) and str(p).strip()
                    ]
                    questions.extend(file_prompts)
                    print(f"    ✓ Loaded {len(file_prompts)} prompts from parquet file")

                else:
                    # Use first column if standard columns not found
                    first_col = df.columns[0]
                    print(f"    Warning: Using first column '{first_col}' as prompts")
                    file_prompts = df[first_col].tolist()
                    file_prompts = [
                        str(p).strip()
                        for p in file_prompts
                        if pd.notna(p) and str(p).strip()
                    ]
                    questions.extend(file_prompts)
                print(f"    ✓ Loaded {len(file_prompts)} prompts from parquet file")

            else:
                print(f"    Warning: Unsupported file format {path.suffix}")

        except Exception as e:
            print(f"    ✗ Failed to load file {file_path}: {e}")
            continue

    # Return explicit_tasks only if we found any
    if explicit_tasks and len(explicit_tasks) == len(questions):
        return questions, explicit_tasks
    else:
        return questions, None


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="ActorRolloutRefWorker vLLM Demo")
    parser.add_argument(
        "--model_path", type=str, default="Qwen/Qwen3-8B", help="Path to the model"
    )
    parser.add_argument(
        "--template_type", type=str, default="default", help="System template type"
    )
    parser.add_argument(
        "--val_data", type=str, nargs="+", help="Validation data file(s)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--num_samples", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=20,
        help="Number of candidates to generate per prompt",
    )
    parser.add_argument(
        "--enable_iterative",
        action="store_true",
        default=True,
        help="Enable iterative reprompting for diverse answers",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum iterations for diverse answer generation",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for filtering (0.0-1.0)",
    )
    parser.add_argument(
        "--disable_similarity_filtering",
        action="store_true",
        help="Disable similarity filtering of answers",
    )
    parser.add_argument(
        "--rollout_name", type=str, default="vllm_answer_repeat", help="Rollout name"
    )

    args = parser.parse_args()

    print("=== ActorRolloutRefWorker vLLM Demo ===\n")

    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This demo requires a GPU.")
        return False

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

    # Model setup
    print(f"Loading model: {args.model_path}")
    print(f"Using template type: {args.template_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if tokenizer supports thinking
    has_thinking_support = (
        hasattr(tokenizer, "apply_chat_template")
        and "enable_thinking" in tokenizer.apply_chat_template.__code__.co_varnames
    )
    print(f"Tokenizer thinking support: {has_thinking_support}")
    if not has_thinking_support:
        print("Warning: Tokenizer may not support enable_thinking parameter")

    # Load validation data
    if args.val_data:
        print(f"Loading validation data from {len(args.val_data)} file(s):")
        questions, explicit_tasks = load_validation_data(args.val_data)

        if not questions:
            print("❌ No prompts loaded from validation files!")
            return False

        print(f"✓ Successfully loaded {len(questions)} prompts from validation files")
    else:
        # Default sample questions
        questions = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short poem about the ocean.",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "What is the capital of France?",
        ]
        print(f"Using {len(questions)} default sample questions")

    # Limit number of samples
    if len(questions) > args.num_samples:
        questions = questions[: args.num_samples]
        print(f"Limited to {len(questions)} samples")
    else:
        print(f"Processing all {len(questions)} available questions")

    # Use global configuration and update with command line arguments
    config = json.load(open("rollout_config.json"))
    config["model"]["path"] = args.model_path
    config["model"]["tokenizer_path"] = args.model_path
    config["rollout"]["template_type"] = args.template_type
    config["rollout"]["temperature"] = args.temperature
    config["rollout"]["response_length"] = args.max_tokens
    config["rollout"]["n_candidates"] = args.n_candidates
    config["rollout"]["enable_iterative_reprompting"] = args.enable_iterative
    config["rollout"]["max_iterative_iterations"] = args.max_iterations
    config["rollout"]["similarity_threshold"] = args.similarity_threshold
    config["rollout"][
        "use_similarity_filtering"
    ] = not args.disable_similarity_filtering
    config["rollout"]["name"] = args.rollout_name

    print(
        f"Config: prompt_length={config['rollout']['prompt_length']}, response_length={config['rollout']['response_length']}"
    )
    print(
        f"Generation: temperature={config['rollout']['temperature']}, top_p={config['rollout']['top_p']}"
    )
    print(
        f"Autorater: n_candidates={config['rollout']['n_candidates']}, enable_iterative={config['rollout']['enable_iterative_reprompting']}"
    )
    print(
        f"Diversity: similarity_filtering={config['rollout']['use_similarity_filtering']}, threshold={config['rollout']['similarity_threshold']}"
    )
    print(f"Iterative: max_iterations={config['rollout']['max_iterative_iterations']}")

    # Initialize ActorRolloutRefWorker
    print("\nInitializing ActorRolloutRefWorker...")
    try:
        worker = ActorRolloutRefWorker(config=DictConfig(config), role="rollout")
        print("✓ ActorRolloutRefWorker initialized successfully!")

        worker.init_model()
        print("✓ Model initialized successfully!")

    except Exception as e:
        print(f"❌ Error initializing ActorRolloutRefWorker: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Prepare prompts
    print(f"\nPreparing {len(questions)} prompts...")
    if explicit_tasks:
        print(f"Storing explicit tasks in non_tensor_batch for reference")
        # Show a preview of what's being stored
        for i in range(min(2, len(questions))):
            print(f"  Storing for prompt {i + 1}:")
            print(f"    Question: {questions[i]}")
            print(f"    Explicit task: {explicit_tasks[i]}")
            print()

        prompts = create_prompts_dataproto(
            tokenizer,
            questions,
            config["rollout"]["prompt_length"],
            template_type=config["rollout"]["template_type"],
            explicit_tasks=explicit_tasks,
        )
    else:
        print(f"No explicit tasks available")

        prompts = create_prompts_dataproto(
            tokenizer,
            questions,
            config["rollout"]["prompt_length"],
            template_type=config["rollout"]["template_type"],
        )

    print(f"Prompt batch shape: {prompts.batch['input_ids'].shape}")

    # Generate responses
    print("\n🚀 Generating responses...")
    try:
        torch.cuda.empty_cache()

        output = worker.generate_sequences(prompts)
        print("✅ Generation completed!")

        # Process results
        results = []
        for i, question in enumerate(questions):
            # Get response tokens (excluding padding)
            response_tokens = output.batch["responses"][i]
            response_tokens = response_tokens[response_tokens != tokenizer.pad_token_id]

            # Decode response
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

            result = {
                "question": question,
                "response": response_text,
                "num_tokens": len(response_tokens),
            }
            results.append(result)

            print(f"\n--- Example {i + 1} ---")
            print(f"Q: {question}")
            print(f"A: {response_text}")
            print(f"   ({len(response_tokens)} tokens)")

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "generation_results.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n✅ Results saved to {output_file}")

        # Generate simplified HTML visualization
        try:
            from make_html import create_simple_html_visualization

            html_file = output_dir / "simple_rollout_visualization.html"
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            create_simple_html_visualization(results, html_file, tokenizer)
            print(f"🎨 Simplified HTML visualization saved to {html_file}")
        except ImportError:
            print("ℹ️  make_html.py not available, skipping HTML visualization")
        except Exception as e:
            print(f"⚠️  Error generating HTML visualization: {e}")

        print(f"✨ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Example usage with new options:
    # python simple_vllm_rollout_demo.py --n_candidates 30 --max_iterations 5 --similarity_threshold 0.8
    # python simple_vllm_rollout_demo.py --disable_similarity_filtering --enable_iterative
    # python simple_vllm_rollout_demo.py --val_data data/ambiguous_prompts.parquet
    main()
