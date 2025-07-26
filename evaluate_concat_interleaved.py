#!/usr/bin/env python3
"""
Evaluate paired interleaved coding traces from BigCodeBench parquet files.
Loads the parquet file, extracts answers, and runs the code evaluator.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer
from omegaconf import DictConfig, OmegaConf
from verl.workers.autorater.autorater_utils import extract_solution
from verl.workers.code_evaluator import CodeEvaluator

def load_pair_interleaved_data(parquet_file: str) -> List[Dict[str, Any]]:
    """Load paired interleaved data from parquet file."""
    print(f"Loading data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    data = df.to_dict('records')
    print(f"Loaded {len(data)} paired samples")
    return data

def extract_answers_from_interleaved(answer: str) -> List[str]:
    """Extract individual code answers from interleaved format."""
    # The interleaved format is: <think>1</think>\n<answer>code1</answer>\n<think>2</think>\n<answer>code2</answer>
    answers = []
    
    # Extract code from <answer> tags
    import re
    answer_matches = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)
    
    for match in answer_matches:
        code = match.strip()
        if code:
            answers.append(code)
    
    return answers

def prepare_reward_model_info(data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare reward model info for both problems in the pair."""
    reward_model = data_item.get('reward_model', {})
    
    # Extract unit tests and libs for both problems
    test_1 = reward_model["test_1"]
    test_2 = reward_model["test_2"]
    libs_1 = reward_model["libs_1"]
    libs_2 = reward_model["libs_2"]
    
    # Create reward model info for each problem
    rm_info_1 = {
        "style": "code",
        "unit_tests": test_1,
        "libs": libs_1,
        "ground_truth": ""
    }
    
    rm_info_2 = {
        "style": "code", 
        "unit_tests": test_2,
        "libs": libs_2,
        "ground_truth": ""
    }
    
    return [rm_info_1, rm_info_2]

def evaluate_pair_interleaved_data(
    data: List[Dict[str, Any]], 
    config: DictConfig,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate paired interleaved data using the code evaluator."""
    
    # Initialize code evaluator
    code_evaluator = CodeEvaluator(config=config, tokenizer=tokenizer)
    
    # Limit samples if specified
    if max_samples:
        data = data[:max_samples]
    
    all_results = []
    
    for i, data_item in enumerate(data):
        print(f"\n--- Evaluating pair {i+1}/{len(data)} ---")
        
        # Extract the interleaved answer
        answer = data_item.get('answer', '')
        if not answer:
            print(f"  ❌ No answer found for pair {i+1}")
            continue
        
        # Extract individual code answers
        code_answers = extract_answers_from_interleaved(answer)
        if len(code_answers) != 2:
            print(f"  ❌ Expected 2 code answers, found {len(code_answers)}")
            continue
        
        print(f"  Found {len(code_answers)} code answers")
        
        # Prepare reward model info for both problems
        rm_infos = prepare_reward_model_info(data_item)

        # Extract prompts for both problems
        prompt_1 = data_item.get('prompt_1', '')
        prompt_2 = data_item.get('prompt_2', '')
        prompts = [prompt_1, prompt_2]
        
        # Evaluate each problem separately
        # Evaluate problem 1
        print(f"  Evaluating problem 1...")
        print(rm_infos[0]["libs"])
        result_1 = code_evaluator.evaluate_code(
            answers=[code_answers[0]],
            prompts=[prompts[0]],
            rm_infos=[rm_infos[0]],
            batch_indices=[i]
        )

        # Evaluate problem 2  
        print(f"  Evaluating problem 2...")
        print(rm_infos[1]["libs"])
        result_2 = code_evaluator.evaluate_code(
            answers=[code_answers[1]],
            prompts=[prompts[1]], 
            rm_infos=[rm_infos[1]],
            batch_indices=[i]
        )
        
        # Combine results
        pair_result = {
            "pair_index": i,
            "task_id_1": data_item.get('task_id_1', ''),
            "task_id_2": data_item.get('task_id_2', ''),
            "problem_1": {
                "pass_rate": result_1["unit_test_pass_rate"][0] if result_1["unit_test_pass_rate"] else 0.0,
                "code": code_answers[0][:200] + "..." if len(code_answers[0]) > 200 else code_answers[0]
            },
            "problem_2": {
                "pass_rate": result_2["unit_test_pass_rate"][0] if result_2["unit_test_pass_rate"] else 0.0,
                "code": code_answers[1][:200] + "..." if len(code_answers[1]) > 200 else code_answers[1]
            },
            "combined_pass_rate": (result_1["unit_test_pass_rate"][0] + result_2["unit_test_pass_rate"][0]) / 2 if result_1["unit_test_pass_rate"] and result_2["unit_test_pass_rate"] else 0.0
        }
        
        all_results.append(pair_result)
        print(f"  ✅ Problem 1 pass rate: {pair_result['problem_1']['pass_rate']:.3f}")
        print(f"  ✅ Problem 2 pass rate: {pair_result['problem_2']['pass_rate']:.3f}")
        print(f"  ✅ Combined pass rate: {pair_result['combined_pass_rate']:.3f}")
            
    # Calculate summary statistics
    if all_results:
        problem_1_pass_rates = [r["problem_1"]["pass_rate"] for r in all_results]
        problem_2_pass_rates = [r["problem_2"]["pass_rate"] for r in all_results]
        combined_pass_rates = [r["combined_pass_rate"] for r in all_results]
        
        summary = {
            "total_pairs_evaluated": len(all_results),
            "problem_1": {
                "mean_pass_rate": sum(problem_1_pass_rates) / len(problem_1_pass_rates),
                "max_pass_rate": max(problem_1_pass_rates),
                "min_pass_rate": min(problem_1_pass_rates)
            },
            "problem_2": {
                "mean_pass_rate": sum(problem_2_pass_rates) / len(problem_2_pass_rates),
                "max_pass_rate": max(problem_2_pass_rates),
                "min_pass_rate": min(problem_2_pass_rates)
            },
            "combined": {
                "mean_pass_rate": sum(combined_pass_rates) / len(combined_pass_rates),
                "max_pass_rate": max(combined_pass_rates),
                "min_pass_rate": min(combined_pass_rates)
            }
        }
    else:
        summary = {"error": "No pairs were successfully evaluated"}
    
    return {
        "summary": summary,
        "detailed_results": all_results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate paired interleaved coding traces")
    parser.add_argument("--parquet_file", type=str, required=True,
                       help="Path to the paired interleaved parquet file")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Model name for tokenizer")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map for model loading")
    args = parser.parse_args()
    
    print(f"\n🚀 PAIRED INTERLEAVED EVALUATION")
    print(f"Parquet file: {args.parquet_file}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Model: {args.model_name}")
    print("="*60)
    
    # Load data
    data = load_pair_interleaved_data(args.parquet_file)
    
    # Load tokenizer
    print(f"\n🚀 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Create config for code evaluator
    config = OmegaConf.create({
        "max_concurrent": 4,
        "execute_sequential": False
    })
    
    # Evaluate data
    results = evaluate_pair_interleaved_data(
        data=data,
        config=config,
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_file}")
    
    # Print summary
    summary = results["summary"]
    if "error" not in summary:
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"Total pairs evaluated: {summary['total_pairs_evaluated']}")
        print(f"\nProblem 1:")
        print(f"  Mean pass rate: {summary['problem_1']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['problem_1']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['problem_1']['min_pass_rate']:.3f}")
        print(f"\nProblem 2:")
        print(f"  Mean pass rate: {summary['problem_2']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['problem_2']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['problem_2']['min_pass_rate']:.3f}")
        print(f"\nCombined:")
        print(f"  Mean pass rate: {summary['combined']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['combined']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['combined']['min_pass_rate']:.3f}")
    else:
        print(f"❌ {summary['error']}")

if __name__ == "__main__":
    main() 