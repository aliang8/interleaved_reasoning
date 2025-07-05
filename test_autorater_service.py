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
            if health_data.get('memory_usage'):
                mem = health_data['memory_usage']
                print(f"   GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['total_gb']:.2f}GB total")
            return health_data['autorater_initialized']
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
            "master_port": 29500
        }
        
        response = requests.post(f"{base_url}/initialize", json=init_request, timeout=120)
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

    from transformers import AutoTokenizer
    
    # Load the same tokenizer used by the AutoRater service
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=False)
    
    # Sample prompts asking to implement simple Python functions
    sample_prompts = [
        "Write a Python function add(a, b) that returns their sum.",
        "Write a Python function multiply(a, b) that returns their product."
    ]

    # Predicted answers: first is correct implementation, second intentionally wrong
    sample_responses = [
        """```python\ndef add(a, b):\n    return a + b\n```""",
        """```python\ndef multiply(a, b):\n    # bug: returns sum instead of product\n    return a + b\n```"""
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
        [
            "assert add(1, 2) == 3",
            "assert add(-1, 5) == 4"
        ],
        [
            "assert multiply(2, 3) == 6",
            "assert multiply(-1, 4) == -4"
        ]
    ]

    reward_infos = []
    for tests in unit_tests_cases:
        reward_infos.append({
            "ground_truth": "code",
            "unit_tests": tests
        })

    sample_data = {
        "prompts": tokenized_prompts,
        "responses": tokenized_responses,
        "attention_mask": attention_masks,
        "position_ids": position_ids,
        "reward_model_info": reward_infos
    }
    
    print(f"   Created real tokenized data:")
    print(f"   - Prompt 1 tokens: {len(tokenized_prompts[0])} tokens")
    print(f"   - Response 1 tokens: {len(tokenized_responses[0])} tokens")
    print(f"   - Prompt 2 tokens: {len(tokenized_prompts[1])} tokens") 
    print(f"   - Response 2 tokens: {len(tokenized_responses[1])} tokens")
    
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
        
        if result.get('code_scores'):
            print(f"   Code scores: {result['code_scores']}")
            print(f"   Tests passed: {result.get('code_tests_passed')} / {result.get('code_total_tests')}")
        
        if result.get('autorater_explanations'):
            print(f"   Explanations available: {len(result['autorater_explanations'])} items")
        
        return result['success']
    else:
        print(f"❌ Evaluation failed: HTTP {response.status_code}")
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

def main():
    parser = argparse.ArgumentParser(description="Test AutoRater FastAPI Service")
    parser.add_argument("--host", type=str, required=True, help="AutoRater service host IP")
    parser.add_argument("--port", type=int, default=80, help="AutoRater service port")
    parser.add_argument("--skip-init", action="store_true", help="Skip initialization test (if auto-initialized)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation test")
    parser.add_argument("--shutdown", action="store_true", help="Send shutdown command at the end")
    
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
            "model": {
                "path": "/path/to/model",
                "tensor_parallel_size": 1
            },
            "rollout": {
                "tensor_model_parallel_size": 1
            },
            "autorater": {
                "tensor_model_parallel_size": 1
            }
        }
        test_initialization(base_url, sample_config)
    else:
        print("\n2. Skipping initialization (already initialized or --skip-init specified)")
    
    # Test evaluation
    if not args.skip_eval:
        print("\n3. Testing evaluation...")
        test_evaluation(base_url)
    else:
        print("\n3. Skipping evaluation test")
    
    # Test shutdown if requested
    if args.shutdown:
        print("\n4. Testing shutdown...")
        test_shutdown(base_url)
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

if __name__ == "__main__":
    main() 