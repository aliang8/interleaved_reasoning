#!/usr/bin/env python3
"""
Script to rate answers in a JSONL file using the AutoRater service.
Usage:
    python rate_jsonl_with_autorater.py \
    --input input.jsonl \
    --output output.jsonl \
    --autorater_url http://10.128.0.30:81 \
    --tokenizer Qwen/Qwen2.5-7B-Instruct \
    --batch_size 16 \
    --template outline
"""

import argparse
import json
from typing import List, Dict, Any
from tqdm import tqdm

from verl.workers.autorater.autorater_utils import extract_solution
from verl.utils.autorater_client import call_autorater_service
from verl.utils.tokenizer import hf_tokenizer


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Rate answers in a JSONL file using AutoRater service."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--autorater_url",
        required=True,
        help="Base URL for AutoRater service (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Model name or path for tokenizer (e.g. mistralai/Mistral-7B-Instruct-v0.2)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for autorater requests"
    )
    parser.add_argument(
        "--template", required=True, help="Template to use for autorater requests"
    )
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = hf_tokenizer(args.tokenizer)

    # Load data
    data = load_jsonl(args.input)

    print(f"Loaded {len(data)} entries")

    # Prepare batches for autorater
    autorater_inputs = []  # (entry_idx, answer_idx, question, answer)
    for entry_idx, entry in enumerate(data):
        question = entry.get("question", "")
        answer_field = entry.get("answer", "")
        extracted_answers = extract_solution(answer_field, extract_all=True)
        if not extracted_answers:
            extracted_answers = []
        for answer_idx, extracted in enumerate(extracted_answers):
            autorater_inputs.append((entry_idx, answer_idx, question, extracted))

    # Batch and rate
    batch_size = args.batch_size
    autorater_scores = [None] * len(autorater_inputs)
    autorater_decisions = [None] * len(autorater_inputs)
    autorater_explanations = [None] * len(autorater_inputs)
    autorater_raw_responses = [None] * len(autorater_inputs)

    for start in tqdm(
        range(0, len(autorater_inputs), batch_size), desc="Rating with AutoRater"
    ):
        batch = autorater_inputs[start : start + batch_size]
        batch_questions = [q for (_, _, q, _) in batch]
        batch_answers = [a for (_, _, _, a) in batch]
        # For helpfulness, we can use the question as prompt and answer as response
        # (ground truth is not needed for helpfulness, so we use empty string)
        tokenized_prompts = tokenizer(
            batch_questions,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids.tolist()
        tokenized_responses = tokenizer(
            batch_answers,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).input_ids.tolist()

        payload = {
            "prompts": tokenized_prompts,
            "responses": tokenized_responses,
            "attention_mask": [[1] * len(r) for r in tokenized_responses],
            "position_ids": [list(range(len(r))) for r in tokenized_responses],
            "reward_model_info": [
                {"template": args.template, "ground_truth": ""}
                for _ in range(len(batch))
            ],
        }
        scores, decisions, explanations, raw_responses = call_autorater_service(
            args.autorater_url,
            payload,
            batch_size=len(batch),
            endpoint="/evaluate_autorater",
        )
        autorater_scores[start : start + batch_size] = scores
        autorater_decisions[start : start + batch_size] = decisions
        autorater_explanations[start : start + batch_size] = explanations
        autorater_raw_responses[start : start + batch_size] = raw_responses

        import ipdb

        ipdb.set_trace()

    # Attach ratings to entries
    # We'll add a new field: 'autorater_helpfulness' as a list of dicts (one per extracted answer)
    for entry in data:
        entry["autorater_helpfulness"] = []

    for idx, (entry_idx, answer_idx, question, extracted) in enumerate(
        autorater_inputs
    ):
        rating = {
            "answer": extracted,
            "score": autorater_scores[idx],
            "decision": autorater_decisions[idx],
            "explanation": autorater_explanations[idx],
            "raw_response": autorater_raw_responses[idx],
        }
        data[entry_idx]["autorater_helpfulness"].append(rating)

    # Save output
    save_jsonl(data, args.output)
    print(f"Wrote rated data to {args.output}")


if __name__ == "__main__":
    main()
