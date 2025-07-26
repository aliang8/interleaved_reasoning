#!/usr/bin/env python3
"""
Evaluate code_list interleaved coding traces from parquet files.
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

def load_code_list_interleaved_data(parquet_file: str) -> List[Dict[str, Any]]:
    """Load code_list interleaved data from parquet file."""
    print(f"Loading data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    data = df.to_dict('records')
    print(f"Loaded {len(data)} code_list samples")
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
    """Prepare reward model info for each code solution in the interleaved format."""
    reward_model = data_item.get('reward_model', {})
    
    # Extract unit tests and libs (t uses single test field)
    unit_tests = reward_model["test"]
    libs = reward_model["libs"] 
    libs = ["numpy", "pandas"]

    print(f"libs: {libs}")
    
    # Create reward model info for each solution (same unit tests for all solutions)
    rm_infos = []
    num_solutions = len(extract_answers_from_interleaved(data_item.get('answer', '')))
    
    for _ in range(num_solutions):
        rm_info = {
            "style": "code",
            "unit_tests": unit_tests, 
            "libs": libs,
            "ground_truth": ""
        }
        rm_infos.append(rm_info)
    
    return rm_infos

def evaluate_code_list_interleaved_data(
    data: List[Dict[str, Any]], 
    config: DictConfig,
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate code_list interleaved data using the code evaluator."""
    
    # Initialize code evaluator
    code_evaluator = CodeEvaluator(config=config, tokenizer=tokenizer)
    
    # Limit samples if specified
    if max_samples:
        data = data[:max_samples]
    
    all_results = []
    
    for i, data_item in enumerate(data):
        print(f"\n--- Evaluating sample {i+1}/{len(data)} ---")
        
        # Extract the interleaved answer
        answer = data_item.get('answer', '')
        if not answer:
            print(f"  ❌ No answer found for sample {i+1}")
            continue
        # Extract individual code answers
        code_answers = extract_answers_from_interleaved(answer)
        if len(code_answers) == 0:
            print(f"  ❌ No code answers found for sample {i+1}")
            continue
        
        print(f"  Found {len(code_answers)} code answers")
        
        # Prepare reward model info for each solution
        rm_infos = prepare_reward_model_info(data_item)
        
        # Extract prompt
        prompt = data_item.get('prompt', '')
        
        # Evaluate each solution
        solution_results = []
        for j, code_answer in enumerate(code_answers):
            print(f"  Evaluating solution {j+1}...")
            result = code_evaluator.evaluate_code(
                answers=[code_answer],
                prompts=[prompt],
                rm_infos=[rm_infos[j]],
                batch_indices=[i]
            )
            
            unit_test_pass_rate = result["unit_test_pass_rate"][0]

            # if unit_test_pass_rate == 0.0:
            #     import ipdb; ipdb.set_trace()

            solution_result = {
                "solution_index": j,
                "pass_rate": unit_test_pass_rate,
                "code": code_answer[:200] + "..." if len(code_answer) > 200 else code_answer
            }
            solution_results.append(solution_result)
            print(f"    ✅ Solution {j+1} pass rate: {solution_result['pass_rate']:.3f}")

        
        # Calculate combined results for this sample
        if solution_results:
            pass_rates = [r["pass_rate"] for r in solution_results]
            combined_pass_rate = sum(pass_rates) / len(pass_rates)
            
            sample_result = {
                "sample_index": i,
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "function_signature": data_item.get('function_signature', ''),
                "num_solutions": len(code_answers),
                "solutions": solution_results,
                "combined_pass_rate": combined_pass_rate,
                "best_pass_rate": max(pass_rates),
                "worst_pass_rate": min(pass_rates)
            }
            
            all_results.append(sample_result)
            print(f"  ✅ Combined pass rate: {combined_pass_rate:.3f}")
            print(f"  ✅ Best pass rate: {sample_result['best_pass_rate']:.3f}")
            print(f"  ✅ Worst pass rate: {sample_result['worst_pass_rate']:.3f}")
            
    # Calculate summary statistics
    if all_results:
        combined_pass_rates = [r["combined_pass_rate"] for r in all_results]
        best_pass_rates = [r["best_pass_rate"] for r in all_results]
        worst_pass_rates = [r["worst_pass_rate"] for r in all_results]
        
        # Calculate per-solution statistics
        all_solution_pass_rates = []
        for result in all_results:
            all_solution_pass_rates.extend([s["pass_rate"] for s in result["solutions"]])
        
        summary = {
            "total_samples_evaluated": len(all_results),
            "total_solutions_evaluated": len(all_solution_pass_rates),
            "average_solutions_per_sample": sum(len(r["solutions"]) for r in all_results) / len(all_results),
            "combined": {
                "mean_pass_rate": sum(combined_pass_rates) / len(combined_pass_rates),
                "max_pass_rate": max(combined_pass_rates),
                "min_pass_rate": min(combined_pass_rates)
            },
            "best_solutions": {
                "mean_pass_rate": sum(best_pass_rates) / len(best_pass_rates),
                "max_pass_rate": max(best_pass_rates),
                "min_pass_rate": min(best_pass_rates)
            },
            "worst_solutions": {
                "mean_pass_rate": sum(worst_pass_rates) / len(worst_pass_rates),
                "max_pass_rate": max(worst_pass_rates),
                "min_pass_rate": min(worst_pass_rates)
            },
            "all_solutions": {
                "mean_pass_rate": sum(all_solution_pass_rates) / len(all_solution_pass_rates),
                "max_pass_rate": max(all_solution_pass_rates),
                "min_pass_rate": min(all_solution_pass_rates)
            }
        }
    else:
        summary = {"error": "No samples were successfully evaluated"}
    
    return {
        "summary": summary,
        "detailed_results": all_results
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate code_list interleaved coding traces")
    parser.add_argument("--parquet_file", type=str, required=True,
                       help="Path to the code_list interleaved parquet file")
    parser.add_argument("--output_file", type=str, default="code_list_evaluation_results.json",
                       help="Output file for evaluation results")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                       help="Model name for tokenizer")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map for model loading")
    args = parser.parse_args()
    
    print(f"\n🚀 CODE_LIST INTERLEAVED EVALUATION")
    print(f"Parquet file: {args.parquet_file}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Model: {args.model_name}")
    print("="*60)
    
    # Load data
    data = load_code_list_interleaved_data(args.parquet_file)
    
    # Load tokenizer
    print(f"\n🚀 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Create config for code evaluator
    config = OmegaConf.create({
        "max_concurrent": 4,
        "execute_sequential": False
    })
    
    # Evaluate data
    results = evaluate_code_list_interleaved_data(
        data=data,
        config=config,
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Filter and save clean parquet
    # Load original data as DataFrame
    df = pd.read_parquet(args.parquet_file)
    # Find indices to keep (where all unit_test_pass_rate > 0)
    keep_indices = []
    for i, res in enumerate(results['detailed_results']):
        if all(sol['pass_rate'] > 0.0 for sol in res['solutions']):
            keep_indices.append(res['sample_index'])
    print(f"\nOriginal samples: {len(df)}; After filtering: {len(keep_indices)}")
    df_clean = df.iloc[keep_indices]
    # Reset index to get clean integer indices
    df_clean = df_clean.reset_index(drop=True)
    
    # Create extra_info column if it doesn't exist
    if "extra_info" not in df_clean.columns:
        df_clean["extra_info"] = [{} for _ in range(len(df_clean))]
    
    for i, row in df_clean.iterrows():
        df_clean.at[i, "prompt"] = [{"role": "user", "content": row["prompt"]}]
        df_clean.at[i, 'reward_model']['libs'] = ""
        # Update extra_info with the current index
        new_extra_info = {"index": i}
        df_clean.at[i, "extra_info"] = new_extra_info

    clean_parquet_file = args.parquet_file.replace('.parquet', '_clean.parquet')
    df_clean.to_parquet(clean_parquet_file)
    print(f"Filtered clean parquet saved to: {clean_parquet_file}")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_file}")
    
    # Print summary
    summary = results["summary"]
    if "error" not in summary:
        print(f"\n📊 SUMMARY STATISTICS")
        print(f"Total samples evaluated: {summary['total_samples_evaluated']}")
        print(f"Total solutions evaluated: {summary['total_solutions_evaluated']}")
        print(f"Average solutions per sample: {summary['average_solutions_per_sample']:.2f}")
        print(f"\nCombined (per sample):")
        print(f"  Mean pass rate: {summary['combined']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['combined']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['combined']['min_pass_rate']:.3f}")
        print(f"\nBest solutions (per sample):")
        print(f"  Mean pass rate: {summary['best_solutions']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['best_solutions']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['best_solutions']['min_pass_rate']:.3f}")
        print(f"\nWorst solutions (per sample):")
        print(f"  Mean pass rate: {summary['worst_solutions']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['worst_solutions']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['worst_solutions']['min_pass_rate']:.3f}")
        print(f"\nAll solutions:")
        print(f"  Mean pass rate: {summary['all_solutions']['mean_pass_rate']:.3f}")
        print(f"  Max pass rate: {summary['all_solutions']['max_pass_rate']:.3f}")
        print(f"  Min pass rate: {summary['all_solutions']['min_pass_rate']:.3f}")
    else:
        print(f"❌ {summary['error']}")

if __name__ == "__main__":
    main() 