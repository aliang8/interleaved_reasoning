#!/usr/bin/env python3
"""
Example script demonstrating how to use random selection instead of oracle/autorater
in the vLLM Best-of-N rollout.

This script shows how to configure the evaluation to use random plan selection
instead of calling the autorater service for plan evaluation.
"""

from evaluation_base import EvaluationConfig
from math_eval import MathEvaluator
import pyrallis


def main():
    """Example of using random selection in best-of-n rollout."""
    
    # Create configuration with random selection enabled
    config = EvaluationConfig(
        # Basic configuration
        model_path="Qwen/Qwen3-8B",
        output_dir="logs/evaluation_random_selection",
        batch_size=10,
        max_problems=50,
        
        # Use best-of-n rollout
        rollout_name="vllm_best_of_n",
        n_candidates=5,  # Generate 5 candidates per prompt
        
        # Enable random selection instead of oracle
        use_random_selection=True,
        random_seed=123,  # Fixed seed for reproducible results
        
        # Math evaluation specific
        dataset="math500",
        template_type="plan_first",
        response_length=2048,
    )
    
    print("=== Random Selection Best-of-N Example ===")
    print(f"Rollout: {config.rollout_name}")
    print(f"Candidates per prompt: {config.n_candidates}")
    print(f"Random selection: {config.use_random_selection}")
    print(f"Random seed: {config.random_seed}")
    print(f"Dataset: {config.dataset}")
    print(f"Template: {config.template_type}")
    print("=" * 50)
    
    # Create and run evaluator
    evaluator = MathEvaluator(config)
    success = evaluator.run_evaluation()
    
    if success:
        print("\n✅ Random selection evaluation completed successfully!")
        print("📊 Check the output directory for results and HTML visualization")
    else:
        print("\n❌ Random selection evaluation failed!")
    
    return success


if __name__ == "__main__":
    main() 