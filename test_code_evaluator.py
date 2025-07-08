#!/usr/bin/env python3
"""
Test script for CodeEvaluator functionality
Tests local code evaluation with unit tests and assert statements
"""

import sys
import os
import time
import argparse
from typing import List, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'verl'))

# Create tokenizer once at module level
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)
    print("✅ Tokenizer loaded successfully")
except ImportError:
    print("❌ transformers not available, using mock tokenizer")
    class MockTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return f"decoded_text_{len(ids)}"
    TOKENIZER = MockTokenizer()

def test_code_evaluator_basic(tokenizer):
    """Test basic CodeEvaluator functionality with simple assert statements."""
    
    print("\n🔬 Testing CodeEvaluator with basic assert statements...")
    
    try:
        from omegaconf import OmegaConf
        from verl.workers.code_evaluator import CodeEvaluator
        
        # Initialize CodeEvaluator with passed tokenizer
        config = OmegaConf.create({})

        original_prompts = [
            "I need to create a function that adds two numbers. This is a basic arithmetic operation that takes two parameters and returns their sum."
        ]

        code_evaluator = CodeEvaluator(
            config=config,
            tokenizer=tokenizer,
            template_type=None,
        )
        
        # Test data with simple functions
        predicted_answers = [
            """```python
def add(a, b):
    return a + b
```""",
            """```python
def multiply(a, b):
    # Bug: returns sum instead of product
    return a + b
```""",
            """```python
def divide(a, b):
    return a / b
```""",
        ]
        
        reward_model_info = [
            {
                "ground_truth": "code",
                "unit_tests": [
                    "assert add(1, 2) == 3",
                    "assert add(-1, 5) == 4"
                ]
            },
            {
                "ground_truth": "code", 
                "unit_tests": [
                    "assert multiply(2, 3) == 6",
                    "assert multiply(-1, 4) == -4"
                ]
            },
            {
                "ground_truth": "code",
                "unit_tests": [
                    "assert divide(6, 2) == 3",
                    "assert divide(5, 2) == 2.5"
                ]
            }
        ]
        
        # Evaluate code
        start_time = time.time()
        scores, decisions, explanations, raw_responses = code_evaluator.evaluate_code(
            predicted_answers, original_prompts, reward_model_info, len(predicted_answers)
        )
        processing_time = time.time() - start_time
        
        print("✅ Basic assert statements test successful!")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Code scores: {scores}")
        print(f"   Decisions: {decisions}")
        print(f"   Explanations: {explanations}")
        
        # Clean up
        code_evaluator.close()
        
        # Return True if at least one test passed
        return any(score > 0 for score in scores)
        
    except Exception as e:
        print(f"❌ Basic assert statements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_evaluator_unittest(tokenizer):
    """Test CodeEvaluator with proper unittest framework."""
    
    print("\n🔬 Testing CodeEvaluator with unittest framework...")
    
    try:
        from omegaconf import OmegaConf
        from verl.workers.code_evaluator import CodeEvaluator
        
        # Initialize CodeEvaluator with passed tokenizer
        config = OmegaConf.create({})
        
        original_prompts = [
            "I need to create a function that adds two numbers. This is a basic arithmetic operation that takes two parameters and returns their sum."
        ]
        
        code_evaluator = CodeEvaluator(
            config=config,
            tokenizer=tokenizer,
            template_type=None,
        )
        
        # Test data with pandas DataFrame function
        predicted_answers = [
            """```python
import pandas as pd

def task_func(df: pd.DataFrame, n: int):
    return df.iloc[:, :n], n
```""",
        ]
        
        reward_model_info = [
            {
                "ground_truth": "code",
                "unit_tests": [
                    """import unittest
import pandas as pd
import numpy as np

class TestCases(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'Column1': np.random.rand(10), 'Column2': np.random.rand(10)})

    def test_transformed_data_shape(self):
        transformed_data, n = task_func(self.data, 2)
        self.assertEqual(transformed_data.shape, (10, 2))

    def test_return_value_type(self):
        transformed_data, n = task_func(self.data, 1)
        self.assertIsInstance(transformed_data, pd.DataFrame)
        self.assertIsInstance(n, int)

if __name__ == '__main__':
    unittest.main()
"""
                ]
            }
        ]
        
        # Evaluate code
        start_time = time.time()
        scores, decisions, explanations, raw_responses = code_evaluator.evaluate_code(
            predicted_answers, original_prompts, reward_model_info, len(predicted_answers)
        )
        processing_time = time.time() - start_time
        
        print("✅ Unittest framework test successful!")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Code scores: {scores}")
        print(f"   Decisions: {decisions}")
        print(f"   Explanations: {explanations}")
        
        # Clean up
        code_evaluator.close()
        
        # Return True if at least one test passed
        return any(score > 0 for score in scores)
        
    except Exception as e:
        print(f"❌ Unittest framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_evaluator_with_libraries(tokenizer):
    """Test CodeEvaluator with external libraries (numpy, requests)."""
    
    print("\n🔬 Testing CodeEvaluator with external libraries...")
    
    try:
        from omegaconf import OmegaConf
        from verl.workers.code_evaluator import CodeEvaluator
        
        # Initialize CodeEvaluator with passed tokenizer
        config = OmegaConf.create({})
        
        original_prompts = [
            "I need to create a function that calculates the mean of an array. This is a basic arithmetic operation that takes an array and returns its mean."
        ]
        
        code_evaluator = CodeEvaluator(
            config=config,
            tokenizer=tokenizer,
            template_type=None,
        )
        
        # Test cases with external libraries
        predicted_answers = [
            """```python
import numpy as np

def calculate_mean(arr):
    return np.mean(arr)
```""",
            """```python
import requests

def get_status_code(url):
    response = requests.get(url)
    return response.status_code
```""",
        ]
        
        reward_model_info = [
            {
                "ground_truth": "code",
                "unit_tests": [
                    "import numpy as np\nassert calculate_mean(np.array([1, 2, 3, 4, 5])) == 3.0",
                    "import numpy as np\nassert calculate_mean(np.array([10, 20, 30])) == 20.0"
                ],
                "libs": ["numpy"]
            },
            {
                "ground_truth": "code", 
                "unit_tests": [
                    """import requests
from unittest.mock import patch, MagicMock
with patch('requests.get') as mock_get:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    assert get_status_code('http://example.com') == 200"""
                ],
                "libs": ["requests"]
            }
        ]
        
        # Evaluate code
        start_time = time.time()
        scores, decisions, explanations, raw_responses = code_evaluator.evaluate_code(
            predicted_answers, original_prompts, reward_model_info, len(predicted_answers)
        )
        processing_time = time.time() - start_time
        
        print("✅ External libraries test successful!")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Code scores: {scores}")
        print(f"   Decisions: {decisions}")
        print(f"   Explanations: {explanations}")
        print(f"   Libraries used: numpy, requests")
        
        # Clean up
        code_evaluator.close()
        
        # Return True if at least one test passed
        return any(score > 0 for score in scores)
        
    except Exception as e:
        print(f"❌ External libraries test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_evaluator_interleaved(tokenizer):
    """Test CodeEvaluator with interleaved reasoning."""
    
    print("\n🔬 Testing CodeEvaluator with interleaved reasoning...")
    
    try:
        from omegaconf import OmegaConf
        from verl.workers.code_evaluator import CodeEvaluator
        
        # Initialize CodeEvaluator with passed tokenizer and interleaved reasoning enabled
        config = OmegaConf.create({
            "enable_interleaved_reasoning": True,
            "interleaved_reward_weights": {
                "description": 1.0,
                "code": 2.0,
                "unit_tests": 1.5
            }
        })

        original_prompts = [
            "I need to create a function that adds two numbers. This is a basic arithmetic operation that takes two parameters and returns their sum."
        ]
        
        code_evaluator = CodeEvaluator(
            config=config,
            tokenizer=tokenizer,
            template_type="interleave",
        )
        
        # Test data with interleaved reasoning format
        predicted_answers = [
            """<answer>
1. **Create a function that adds two numbers**:
   - Define a function `add` that takes two parameters `a` and `b` and returns their sum.
   - This is a basic arithmetic operation that takes two parameters and returns their sum.
</answer>

<answer>```python
def add(a, b):
    return a + b
```</answer>

<answer>```python
def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
```</answer>""",
        ]
        
        reward_model_info = [
            {
                "ground_truth": "code",
                "unit_tests": [
                    """import unittest
class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 5), 4)

    def test_add_with_zero(self):
        self.assertEqual(add(0, 0), 0)
"""
                ]
            }
        ]
        
        # Evaluate code with interleaved reasoning
        start_time = time.time()
        scores, decisions, explanations, raw_responses = code_evaluator.evaluate_code(
            predicted_answers, original_prompts, reward_model_info, len(predicted_answers)
        )
        processing_time = time.time() - start_time
        
        print("✅ Interleaved reasoning test successful!")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Total scores: {scores}")
        print(f"   Decisions: {decisions}")
        print(f"   Explanations: {explanations}")
        print(f"   Raw responses: {raw_responses}")
        
        # Clean up
        code_evaluator.close()
        
        # Return True if got a positive score
        return any(score > 0 for score in scores)
        
    except Exception as e:
        print(f"❌ Interleaved reasoning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test CodeEvaluator functionality")
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "unittest", "libs", "interleaved", "all"],
        default="all",
        help="Which test to run: \n"
             "  basic      -> basic assert statements \n"
             "  unittest   -> unittest framework \n"
             "  libs       -> external libraries (numpy, requests) \n"
             "  interleaved -> interleaved reasoning \n"
             "  all        -> run all tests",
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Testing CodeEvaluator functionality")
    print("=" * 50)
    
    results = {}
    
    if args.test == "basic" or args.test == "all":
        results["basic"] = test_code_evaluator_basic(TOKENIZER)
    
    if args.test == "unittest" or args.test == "all":
        results["unittest"] = test_code_evaluator_unittest(TOKENIZER)
    
    if args.test == "libs" or args.test == "all":
        results["libs"] = test_code_evaluator_with_libraries(TOKENIZER)
    
    if args.test == "interleaved" or args.test == "all":
        results["interleaved"] = test_code_evaluator_interleaved(TOKENIZER)
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")
    
    if results:
        print("\nResults Summary:")
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        overall_success = all(results.values())
        print(f"\nOverall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")


if __name__ == "__main__":
    main() 