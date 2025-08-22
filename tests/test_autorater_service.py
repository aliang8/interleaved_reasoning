#!/usr/bin/env python3
"""
Test script for AutoRater FastAPI Service
Run this from a different VM to test the AutoRater service
"""

import requests
import json
import time
import argparse
from typing import List, Dict, Any


def test_health(base_url: str) -> bool:
    """Test the health endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {health_data['status']}")
            print(f"   AutoRater initialized: {health_data['autorater_initialized']}")
            print(f"   GPU available: {health_data['gpu_available']}")
            if health_data.get("memory_usage"):
                mem = health_data["memory_usage"]
                print(
                    f"   GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['total_gb']:.2f}GB total"
                )
            return health_data["autorater_initialized"]
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_initialization(base_url: str, config_data: Dict[str, Any]) -> bool:
    """Test manual initialization (if not auto-initialized)"""
    try:
        init_request = {
            "config": config_data,
            "num_gpus": 1,
            "gpu_ids": [0],
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "master_addr": "127.0.0.1",
            "master_port": 29500,
        }

        response = requests.post(
            f"{base_url}/initialize", json=init_request, timeout=120
        )
        if response.status_code == 200:
            print("✅ Initialization successful!")
            result = response.json()
            print(f"   Status: {result['status']}")
            print(f"   Message: {result['message']}")
            return True
        else:
            print(f"❌ Initialization failed: HTTP {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False


def create_sample_evaluation_data() -> Dict[str, Any]:
    """Create sample data for evaluation testing using real tokenizer"""

    from transformers import AutoTokenizer  # type: ignore

    # Load the same tokenizer used by the AutoRater service
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False
    )

    # Sample prompts asking to implement simple Python functions + DataFrame transform
    sample_prompts = [
        "Write a Python function add(a, b) that returns their sum.",
        "Write a Python function multiply(a, b) that returns their product.",
        "Write a Python function divide(a, b) that returns a / b.",
        "Write a Python function task_func(df: pandas.DataFrame, n: int) that returns the first n columns and n.",
    ]

    # Predicted answers
    sample_responses = [
        """```python\ndef add(a, b):\n    return a + b\n```""",
        """```python\ndef multiply(a, b):\n    # bug: returns sum instead of product\n    return a + b\n```""",
        "I couldn't write the function.",
        """```python\nimport pandas as pd\n
def task_func(df: pd.DataFrame, n: int):\n    return df.iloc[:, :n], n\n```""",
    ]

    # Tokenize prompts and responses
    tokenized_prompts = []
    tokenized_responses = []
    attention_masks = []
    position_ids = []

    for prompt, response in zip(sample_prompts, sample_responses):
        # Tokenize prompt
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)

        # Tokenize response
        response_tokens = tokenizer.encode(response, add_special_tokens=False)

        # Create attention mask (all 1s for valid tokens)
        attention_mask = [1] * len(response_tokens)

        # Create position IDs (sequential from 0)
        position_id = list(range(len(response_tokens)))

        tokenized_prompts.append(prompt_tokens)
        tokenized_responses.append(response_tokens)
        attention_masks.append(attention_mask)
        position_ids.append(position_id)

    # Unit tests for the functions
    unit_tests_cases = [
        ["assert add(1, 2) == 3", "assert add(-1, 5) == 4"],
        ["assert multiply(2, 3) == 6", "assert multiply(-1, 4) == -4"],
        ["assert divide(6, 2) == 3", "assert divide(5, 2) == 2.5"],
        [
            """import unittest\nimport pandas as pd\nimport numpy as np\nclass TestCases(unittest.TestCase):\n    def setUp(self):\n        self.data = pd.DataFrame({'Column1': np.random.rand(10), 'Column2': np.random.rand(10)})\n\n    def test_transformed_data_shape(self):\n        transformed_data, n = task_func(self.data, 2)\n        self.assertEqual(transformed_data.shape, (10, 2))\n"""
        ],
    ]

    reward_infos = []
    for tests in unit_tests_cases:
        reward_infos.append({"ground_truth": "code", "unit_tests": tests})

    sample_data = {
        "prompts": tokenized_prompts,
        "responses": tokenized_responses,
        "attention_mask": attention_masks,
        "position_ids": position_ids,
        "reward_model_info": reward_infos,
    }

    print(f"   Created real tokenized data:")
    print(f"   - Prompt 1 tokens: {len(tokenized_prompts[0])} tokens")
    print(f"   - Response 1 tokens: {len(tokenized_responses[0])} tokens")
    print(f"   - Prompt 2 tokens: {len(tokenized_prompts[1])} tokens")
    print(f"   - Response 2 tokens: {len(tokenized_responses[1])} tokens")
    print(f"   - Prompt 3 tokens: {len(tokenized_prompts[2])} tokens")
    print(f"   - Response 3 tokens: {len(tokenized_responses[2])} tokens")

    return sample_data


def create_sample_outline_data() -> Dict[str, Any]:
    """Create sample data for outline evaluation testing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False
    )

    problem_description = "Given a Directed Acyclic Graph (DAG), write a function to count all possible paths from a given source node to a given destination node."

    # A good, detailed outline and a poor, vague one
    sample_outlines = [
        """
1. **Algorithm Choice**: Use Depth First Search (DFS) starting from the source.
2. **Memoization**: Use a dictionary to store the number of paths from each node to the destination to avoid re-computation.
3. **Base Cases**:
   - If current node is the destination, return 1.
   - If current node is in the memoization table, return its stored value.
4. **Recursive Step**: For the current node, iterate its neighbors. Recursively call DFS for each neighbor and sum the results.
5. **Store Result**: Store the calculated total for the current node in the memoization table before returning.
        """,
        "Just loop through the graph and count the paths.",
        "Breh",
        "You did a good job.",
        """
        1. **Algorithm Choice**: Use Depth First Search (DFS) starting from the source.
2. **Memoization**: Use a dictionary to store the number of paths from each node to the destination to avoid re-computation.
3. **Base Cases**:
   - If current node is the destination, return 1.
   - If current node is in the memoization table, return its stored value.
   """,
    ]

    tokenized_prompts = [
        tokenizer.encode(problem_description, add_special_tokens=True)
    ] * len(sample_outlines)
    tokenized_responses = [
        tokenizer.encode(o, add_special_tokens=False) for o in sample_outlines
    ]
    attention_masks = [[1] * len(r) for r in tokenized_responses]
    position_ids = [list(range(len(r))) for r in tokenized_responses]

    # This is the key part: specifying the 'outline' template
    reward_infos = [{"template": "outline", "ground_truth": ""}] * len(sample_outlines)

    sample_data = {
        "prompts": tokenized_prompts,
        "responses": tokenized_responses,
        "attention_mask": attention_masks,
        "position_ids": position_ids,
        "reward_model_info": reward_infos,
    }

    print("\n   Created sample data for outline evaluation:")
    print(f"   - Prompt tokens: {len(tokenized_prompts[0])} tokens")
    print(f"   - Good outline tokens: {len(tokenized_responses[0])} tokens")
    print(f"   - Bad outline tokens: {len(tokenized_responses[1])} tokens")

    return sample_data


def test_evaluation(base_url: str) -> bool:
    """Test the evaluation endpoint"""

    print("\n🔬 Testing AutoRater evaluation...")

    # Create sample data
    eval_data = create_sample_evaluation_data()

    # Send evaluation request
    start_time = time.time()
    response = requests.post(f"{base_url}/evaluate", json=eval_data, timeout=60)
    request_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print("✅ Evaluation successful!")
        print(f"   Request time: {request_time:.2f}s")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Success: {result['success']}")
        print(f"   Scores: {result['autorater_scores']}")
        print(f"   Decisions: {result['autorater_decisions']}")

        if result.get("code_scores"):
            print(f"   Code scores: {result['code_scores']}")
            print(
                f"   Tests passed: {result.get('code_tests_passed')} / {result.get('code_total_tests')}"
            )
            print(f"   Code stdout: {result.get('code_stdout')}")
            print(f"   Code stderr: {result.get('code_stderr')}")
            print(f"   Code error: {result.get('code_error')}")

        if result.get("autorater_explanations"):
            print(
                f"   Explanations available: {len(result['autorater_explanations'])} items"
            )

        return result["success"]
    else:
        print(f"❌ Evaluation failed: HTTP {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_outline_evaluation(base_url: str) -> bool:
    """Test the /evaluate_autorater endpoint with a code outline request."""
    print("\n🔬 Testing Code Outline evaluation...")

    eval_data = create_sample_outline_data()

    start_time = time.time()
    response = requests.post(
        f"{base_url}/evaluate_autorater", json=eval_data, timeout=60
    )
    request_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print("✅ Outline evaluation successful!")
        print(f"   Request time: {request_time:.2f}s")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Success: {result['success']}")
        print(f"   Scores (good vs. bad outline): {result['autorater_scores']}")
        print(f"   Decisions (good vs. bad outline): {result['autorater_decisions']}")

        # We expect the good outline to get a better score/decision
        if (
            result["autorater_decisions"][0] == 1
            and result["autorater_decisions"][1] == 0
        ):
            print("   ✅ PASSED: Good outline was approved, bad outline was rejected.")
        else:
            print(
                "   ❌ FAILED: The model did not correctly differentiate between the good and bad outlines."
            )

        if result.get("autorater_explanations"):
            print(
                f"   Explanations available: {len(result['autorater_explanations'])} items"
            )

        for explanation in result["autorater_explanations"]:
            print(f"   Explanation: {explanation}")
        return result["success"]
    else:
        print(f"❌ Outline evaluation failed: HTTP {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def test_shutdown(base_url: str) -> bool:
    """Test graceful shutdown"""
    try:
        response = requests.post(f"{base_url}/shutdown", timeout=10)
        if response.status_code == 200:
            print("✅ Shutdown initiated successfully!")
            return True
        else:
            print(f"❌ Shutdown failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Shutdown failed: {e}")
        return False


def test_autorater_only(base_url: str) -> bool:
    """Test the /evaluate_autorater endpoint (LLM scoring only)."""

    print("\n🔬 Testing /evaluate_autorater endpoint…")

    eval_data = create_sample_evaluation_data()

    start_time = time.time()
    response = requests.post(
        f"{base_url}/evaluate_autorater", json=eval_data, timeout=60
    )
    request_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print("✅ Autorater-only evaluation successful!")
        print(f"   Request time: {request_time:.2f}s")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Success: {result['success']}")
        print(f"   Autorater scores: {result['autorater_scores']}")
        print(f"   Decisions: {result['autorater_decisions']}")
        if result.get("autorater_explanations"):
            print(
                f"   Explanations available: {len(result['autorater_explanations'])} items"
            )
        return result["success"]
    else:
        print(f"❌ Autorater-only evaluation failed: HTTP {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def create_sample_plan_evaluation_data() -> Dict[str, Any]:
    """Create sample data for plan evaluation testing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False
    )

    # The prompt asking for a plan to implement a Python function
    prompt = "Write a Python function that takes a list of strings and removes duplicate strings."

    # Multiple candidate plans with varying quality
    #     sample_plans = [
    #         """
    # 1. **Input Validation**: Check if the input is a list and contains only strings
    # 2. **Use Set for Deduplication**: Convert the list to a set to automatically remove duplicates
    # 3. **Preserve Order**: Convert back to a list to maintain the original order
    # 4. **Return Result**: Return the deduplicated list
    # 5. **Edge Cases**: Handle empty list and single element cases
    #         """,
    #         """
    # 1. **Loop Through List**: Iterate through each string in the list
    # 2. **Check for Duplicates**: For each string, check if it already exists in a new list
    # 3. **Build Result**: Add the string to the result list only if it's not a duplicate
    # 4. **Return**: Return the new list without duplicates
    #         """,
    #         """
    # Just remove duplicates from the list.
    #         """,
    #         """
    # 1. **Import Collections**: Use collections.Counter to count occurrences
    # 2. **Count Frequencies**: Count how many times each string appears
    # 3. **Filter Unique**: Keep only strings that appear once
    # 4. **Return**: Return the filtered list
    #         """,
    #         """
    # 1. **Use Dictionary**: Create a dictionary to track seen strings
    # 2. **Iterate and Check**: Go through the list and add unseen strings to result
    # 3. **Maintain Order**: Preserve the order of first appearance
    # 4. **Handle Edge Cases**: Check for None, empty strings, and non-string types
    #         """
    #     ]
    sample_plans = [
        [
            """1. Define the function with a parameter for the list of strings                                                                                                             
2. Create a new list to store the cleaned strings                                                                                                                           
3. Iterate over each string in the input list                                                                                                                               
4. Strip whitespace from each string                                                                                                                                        
5. Check if the stripped string is not empty                                                                                                                                
6. Add non-empty stripped strings to the new list and remove duplicates                                                                                                                         
7. Return the new list of cleaned strings.""",
            """1. Define the function with a parameter for the list of strings  
2. Remove empty strings from the list  
3. Strip whitespace from each string  
4. Convert all strings to lowercase  
5. Return the cleaned list""",
        ]
    ]

    # Tokenize the prompt and

    # Specify the 'plan_evaluation' template
    sample_data = {
        "prompts": [prompt] * len(sample_plans),
        "responses": sample_plans,
        "gt_answers": [""] * len(sample_plans),
        "template_types": ["plan_evaluation"] * len(sample_plans),
        "context": [""] * len(sample_plans),
    }

    tokenized_prompts = [tokenizer.encode(prompt, add_special_tokens=True)] * len(
        sample_plans
    )
    tokenized_responses = [
        tokenizer.encode(plan, add_special_tokens=False) for plan in sample_plans
    ]
    attention_masks = [[1] * len(r) for r in tokenized_responses]
    position_ids = [list(range(len(r))) for r in tokenized_responses]

    print("\n   Created sample data for plan evaluation:")
    print(f"   - Prompt: {prompt}")
    print(f"   - Number of plans: {len(sample_plans[0])}")
    for i, plan in enumerate(sample_plans):
        print(f"   - Plan {i + 1}: {plan}, {len(tokenized_responses[i])} tokens")

    return sample_data


def test_plan_evaluation(base_url: str) -> bool:
    """Test the /evaluate_autorater endpoint with plan evaluation."""
    print("\n🔬 Testing Plan Evaluation...")

    eval_data = create_sample_plan_evaluation_data()

    start_time = time.time()
    response = requests.post(f"{base_url}/evaluate", json=eval_data, timeout=60)
    request_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print("✅ Plan evaluation successful!")
        print(f"   Request time: {request_time:.2f}s")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Success: {result['success']}")
        print(f"   Autorater decisions: {result['autorater_decisions']}")

        # We expect the set-based plan (Plan 1) to be selected as best
        # since it's most efficient and clear
        if result["autorater_decisions"]:
            best_plan = int(result["autorater_decisions"][0])
            print(f"   ✅ Best plan selected: Plan {best_plan}")

        if result.get("autorater_explanations"):
            print(
                f"   Explanations available: {len(result['autorater_explanations'])} items"
            )
            for i, explanation in enumerate(result["autorater_explanations"]):
                print(f"   Explanation {i + 1}: {explanation}")

        return result["success"]
    else:
        print(f"❌ Plan evaluation failed: HTTP {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test AutoRater FastAPI Service")
    parser.add_argument(
        "--host", type=str, default="10.128.0.30", help="AutoRater service host IP"
    )
    parser.add_argument("--port", type=int, default=81, help="AutoRater service port")
    parser.add_argument(
        "--skip-init",
        action="store_true",
        help="Skip initialization test (if auto-initialized)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "full",
            "autorater",
            "tests",
            "split",
            "libs",
            "all",
            "outline",
            "plan",
        ],
        default="full",
        help="Which evaluation endpoint(s) to test: \n"
        "  full      -> /evaluate (LLM AutoRater, legacy endpoint) \n"
        "  autorater -> /evaluate_autorater (LLM only, recommended) \n"
        "  outline   -> /evaluate_autorater (code outline evaluation) \n"
        "  plan      -> /evaluate_autorater (plan evaluation) \n"
        "  all       -> run full + autorater (code tests moved to separate file)",
    )
    parser.add_argument(
        "--shutdown", action="store_true", help="Send shutdown command at the end"
    )

    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"🚀 Testing AutoRater service at {base_url}")
    print("=" * 50)

    # Test health
    print("\n1. Testing health endpoint...")
    is_initialized = test_health(base_url)

    # Test initialization if needed
    if not args.skip_init and not is_initialized:
        print("\n2. Testing initialization...")
        sample_config = {
            "model": {"path": "/path/to/model", "tensor_parallel_size": 1},
            "rollout": {"tensor_model_parallel_size": 1},
            "autorater": {"tensor_model_parallel_size": 1},
        }
        test_initialization(base_url, sample_config)
    else:
        print(
            "\n2. Skipping initialization (already initialized or --skip-init specified)"
        )

    # Evaluation endpoint tests based on mode
    print("\n3. Testing evaluation endpoints…")
    if args.mode == "full":
        test_evaluation(base_url)
    elif args.mode == "autorater":
        test_autorater_only(base_url)
    elif args.mode == "tests":
        test_unit_tests_only(base_url)
    elif args.mode == "outline":
        test_outline_evaluation(base_url)
    elif args.mode == "plan":
        test_plan_evaluation(base_url)
    elif args.mode == "split":
        test_split_unit_tests_endpoint(base_url)
    elif args.mode == "libs":
        test_with_external_libs(base_url)
    elif args.mode == "all":
        ok_full = test_evaluation(base_url)
        ok_auto = test_autorater_only(base_url)
        ok_outline = test_outline_evaluation(base_url)
        ok_plan = test_plan_evaluation(base_url)
        ok_tests = test_unit_tests_only(base_url)  # Now just shows deprecation message
        ok_split = test_split_unit_tests_endpoint(
            base_url
        )  # Now just shows deprecation message
        ok_libs = test_with_external_libs(
            base_url
        )  # Now just shows deprecation message
        print(
            "\nSummary: full=%s, autorater=%s, outline=%s, plan=%s"
            % (ok_full, ok_auto, ok_outline, ok_plan)
        )
        print(
            "Note: For code evaluation tests, run 'python test_code_evaluator.py --test all'"
        )
    else:
        print(f"Unknown mode {args.mode}")

    # Test shutdown if requested
    if args.shutdown:
        print("\n4. Testing shutdown...")
        test_shutdown(base_url)

    print("\n" + "=" * 50)
    print("🏁 Testing completed!")


if __name__ == "__main__":
    main()
