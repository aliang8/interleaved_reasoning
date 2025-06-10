#!/usr/bin/env python3
"""
Test script for accuracy_reward function on different datasets.
Tests with correct and incorrect sample answers to verify reward computation.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import DATASET_CONFIGS, load_single_dataset
from rewards import accuracy_reward, format_reward, kk_exact_reward, kk_partial_reward

@dataclass
class TestConfig:
    """Simple config for testing purposes"""
    hf_cache_dir: str = "/scr/aliang80/hf_cache"
    kk_subset_size: Optional[int] = 3
    math500_subset_size: Optional[int] = 3
    gpqa_subset_size: Optional[int] = 3

def create_sample_answers(question, correct_answer, dataset_type):
    """
    Create sample answers: 1 correct and 2 wrong for each question.
    """
    if dataset_type == "kk":
        # Knights and Knaves - logic puzzles
        # Create realistic KK format answers
        correct = f"<think>Let me analyze this step by step.</think><answer>{correct_answer}</answer>"
        
        # Wrong answer 1: Mixed up some assignments
        wrong1 = f"<think>Let me analyze this step by step.</think><answer>David is a knight, Isabella is a knave, Evelyn is a knight.</answer>"
        
        # Wrong answer 2: All knights (common wrong pattern)
        wrong2 = f"<think>Let me analyze this step by step.</think><answer>Everyone is a knight.</answer>"
    
    elif dataset_type == "math500":
        # Math problems
        correct = f"<think>Let me solve this step by step.</think><answer>{correct_answer}</answer>"
        wrong1 = f"<think>Let me solve this step by step.</think><answer>42</answer>"
        wrong2 = f"<think>Let me solve this step by step.</think><answer>x = 5</answer>"
    
    elif dataset_type == "gpqa":
        # Science questions
        correct = f"<think>Let me analyze this scientific question.</think><answer>{correct_answer}</answer>"
        wrong1 = f"<think>Let me analyze this scientific question.</think><answer>Option A</answer>"
        wrong2 = f"<think>Let me analyze this scientific question.</think><answer>The answer is B</answer>"
    
    else:
        # Default format
        correct = f"<think>Let me think about this.</think><answer>{correct_answer}</answer>"
        wrong1 = f"<think>Let me think about this.</think><answer>Wrong answer 1</answer>"
        wrong2 = f"<think>Let me think about this.</think><answer>Wrong answer 2</answer>"
    
    return correct, wrong1, wrong2

def test_dataset_rewards(dataset_key, config):
    """
    Test accuracy rewards for a specific dataset.
    """
    print(f"\n{'='*60}")
    print(f"TESTING {dataset_key.upper()} DATASET")
    print(f"{'='*60}")
    
    # Load dataset
    train_data, test_data, info = load_single_dataset(dataset_key, config)
    
    # If dataset is kk, replace the answer field with the solution field
    if dataset_key == "kk":
        test_data = test_data.map(lambda x: {**x, "solution": x["solution_text"]})
        train_data = train_data.map(lambda x: {**x, "solution": x["solution_text"]})

    # Use test data if available, otherwise train data
    data_to_test = test_data if test_data is not None else train_data
    if data_to_test is None:
        print(f"No data available for {dataset_key}")
        return
    
    print(f"Dataset info: {info}")
    
    # Get dataset config for field names
    dataset_config = DATASET_CONFIGS[dataset_key]
    question_field = dataset_config["question_field"]
    answer_field = dataset_config["answer_field"]
    
    print(f"Question field: '{question_field}', Answer field: '{answer_field}'")
    
    # Test on first few examples
    num_examples = min(3, len(data_to_test))
    
    for i in range(num_examples):
        example = data_to_test[i]
        question = example[question_field]
        correct_answer = example[answer_field]
        
        print(f"\n{'-'*40}")
        print(f"EXAMPLE {i+1}")
        print(f"{'-'*40}")
        
        # Show question and correct answer
        question_preview = question[:200] + "..." if len(question) > 200 else question
        answer_preview = correct_answer[:100] + "..." if len(correct_answer) > 100 else correct_answer
        
        print(f"Question: {question_preview}")
        print(f"Correct Answer: {answer_preview}")
        
        # Create sample answers
        correct_sample, wrong_sample1, wrong_sample2 = create_sample_answers(
            question, correct_answer, dataset_key
        )
        
        # Test each answer with accuracy_reward
        test_cases = [
            ("Correct Answer", correct_sample),
            ("Wrong Answer 1", wrong_sample1), 
            ("Wrong Answer 2", wrong_sample2)
        ]
        
        print(f"\nTesting rewards:")
        for case_name, sample_answer in test_cases:
            # Prepare data for accuracy_reward function
            completions = [[{"content": sample_answer}]]
            kwargs = {"solution": [correct_answer]}
            
            try:
                 # Test format reward too for comparison
                format_rewards = format_reward(completions, **kwargs)

                # If this is Knights and Knaves dataset, also test KK-specific rewards
                if dataset_key == "kk":
                    kk_exact_rewards = kk_exact_reward(completions, **kwargs)
                    kk_partial_rewards = kk_partial_reward(completions, **kwargs)
                    print(f"    KK Exact Match Reward: {kk_exact_rewards[0]:.3f}")
                    print(f"    KK Partial Match Reward: {kk_partial_rewards[0]:.3f}")
                else:
                    # Test accuracy reward for math problems
                    accuracy_rewards = accuracy_reward(completions, **kwargs)
                    print(f"  {case_name}:")
                    print(f"    General Accuracy Reward: {accuracy_rewards[0]:.3f}")
                
                print(f"    Format Reward: {format_rewards[0]:.3f}")
                print(f"    Sample: {sample_answer[:80]}...")
                
            except Exception as e:
                print(f"  {case_name}: ERROR - {str(e)}")

def main():
    """
    Main test function - tests accuracy rewards on all datasets except MuSiQue.
    """
    print("ACCURACY REWARD TESTING")
    print("="*80)
    print("Testing accuracy_reward function on different datasets")
    print("Each test shows 1 correct answer and 2 wrong answers per example")
    
    config = TestConfig()
    
    # Test datasets (excluding MuSiQue as requested)
    datasets_to_test = ["kk", "math500", "gpqa"]
    
    for dataset_key in datasets_to_test:
        try:
            test_dataset_rewards(dataset_key, config)
        except Exception as e:
            print(f"\nERROR testing {dataset_key}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 