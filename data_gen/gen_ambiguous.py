#!/usr/bin/env python3
"""
Generate subtly modified, more ambiguous MBPP task descriptions by making small changes.
Creates slightly unclear versions of original tasks that introduce subtle uncertainty.
Saves the modified prompts to parquet files for training.

Usage: python data_gen/gen_ambiguous.py --output_dir data --num_samples 50
"""

import argparse
from typing import List, Dict, Any, Tuple
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from helpers import StandardizedRewardModel, save_to_parquet, save_jsonl

# Prompt for generating ambiguous MBPP descriptions
AMBIGUITY_PROMPT = """You are given a coding task prompt.
Your job is to generate a couple versions of this prompt:

Ambiguous version: A vague or under-specified version that could be interpreted in multiple ways.

True intent versions: A precise and unambiguous version that captures the exact coding task.

Format your response as:

Ambiguous prompt: <your ambiguous version>  
True intent prompt 1: <your clarified golden version>  
True intent prompt 2: <your clarified golden version>  
True intent prompt 3: <your clarified golden version>  

Make sure the true intent versions are different from each other. 
The true intent versions do not need to do the same thing as the original prompt, but it should be
a possible interpretation of the ambiguous prompt.

Make sure to not output any other text than the generated prompts. Output the word STOP after the last prompt.

Example:

Input prompt: Write a function that takes a list of strings and sorts them based on the length of the string.

Ambiguous prompt: Write a function that processes lists. 
True intent prompt 1: Write a function that takes a list of strings and sorts them based on the length of the string.
True intent prompt 2: Write a function that takes a list of strings and sorts them in descending order.
True intent prompt 3: Write a function that takes a list of strings and sorts them alphabetically.

Original prompt: {original_task}

Output: """

# Prompt for generating canonical solutions
SOLUTION_PROMPT = """You are given a coding task prompt.
Your job is to generate a canonical solution function named `task_func`.

Format your response as:

```python
def task_func(parameters):
    # Your implementation here
    pass
```

Make sure:
- The function is named exactly `task_func`
- The solution is correct and handles edge cases
- The code is clean and well-structured

Output the word STOP after the function definition. Do not output any other text.

Task: {task_description}

Please proceed with the solution: """

# Prompt for generating unit tests
TESTS_PROMPT = """You are given a coding task prompt and a function definition.
Your job is to generate a list of assert statements that verify the function works correctly.

Function:
{function_code}

Generate unit tests as assert statements:

```python
# Test case 1: basic functionality
assert task_func(input1) == expected_output1

# Test case 2: edge cases
assert task_func(input2) == expected_output2

# Test case 3: error handling
assert task_func(input3) == expected_output3
```

Make sure:
- Tests use assert statements with clear input/output pairs
- Tests cover normal cases, edge cases, and error conditions
- Tests are comprehensive and would catch common bugs
- All tests use the exact function name `task_func`

Output the word STOP after generating the unit tests. Do not output any other text, but the test cases themselves.

Task: {task_description}

Please proceed with the unit tests: """


@contextmanager
def timer(name: str, verbose: bool = True):
    """Context manager for timing operations."""
    start_time = time.time()
    if verbose:
        print(f"    Starting {name}...")
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print(f"    {name} completed in {duration:.2f}s")


class AmbiguityGenerator:
    """Generate ambiguous MBPP task descriptions."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-32B", device_map: str = "auto", 
                 use_similarity_filtering: bool = True, similarity_threshold: float = 0.85):
        """Initialize the ambiguity generator."""
        self.model_name = model_name
        self.device_map = device_map
        self.use_similarity_filtering = use_similarity_filtering
        self.similarity_threshold = similarity_threshold
        
        # Initialize sentence transformer for similarity filtering
        self.sentence_transformer = None
        if self.use_similarity_filtering:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                print(f"✓ Sentence transformer initialized for similarity filtering (threshold: {self.similarity_threshold})")
            except ImportError:
                print("⚠️  Sentence transformers not available, similarity filtering disabled")
                self.use_similarity_filtering = False
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )
        
        print("✓ Model loaded successfully")
    
    def generate_ambiguities(self, original_task: str, max_new_tokens: int = 512, temperature: float = 0.7, num_return_sequences: int = 3) -> List[Dict[str, List[str]]]:
        """Generate subtly modified, more ambiguous versions of a task."""
        
        # Format the prompt
        prompt = AMBIGUITY_PROMPT.format(original_task=original_task)
        
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            # Add STOP as a stop token
            stop_token_id = self.tokenizer.encode("STOP", add_special_tokens=False)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_id[0] if stop_token_id else self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response and parse the structured output
        parsed_responses = []
        for ids in outputs:
            output_ids = ids[len(inputs.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Parse the structured output
            parsed = self.parse_ambiguity_output(response)
            if parsed['ambiguous_prompt'] and parsed['true_intent_prompts']:
                parsed_responses.append(parsed)
        
        return parsed_responses
    
    def generate_solution_and_tests(self, task_description: str, max_new_tokens: int = 1024, temperature: float = 0.3) -> Dict[str, str]:
        """Generate canonical solution and unit tests for a given task."""
        
        # Step 1: Generate the canonical solution
        print(f"        🔧 Generating solution...")
        solution_prompt = SOLUTION_PROMPT.format(task_description=task_description)
        solution_inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": solution_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        solution_inputs = self.tokenizer(solution_inputs, return_tensors="pt")

        # Generate solution
        with torch.no_grad():
            stop_token_id = self.tokenizer.encode("STOP", add_special_tokens=False)
            
            solution_outputs = self.model.generate(
                **solution_inputs,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_id[0] if stop_token_id else self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
        
        # Decode solution response
        solution_output_ids = solution_outputs[0][len(solution_inputs.input_ids[0]):].tolist()
        solution_response = self.tokenizer.decode(solution_output_ids, skip_special_tokens=True).strip()
        canonical_solution = self.parse_solution_output(solution_response)
        
        if not canonical_solution:
            print(f"        ❌ Failed to generate solution")
            return {'canonical_solution': '', 'unit_tests': ''}
        
        print(f"        ✅ Solution generated successfully")
        
        # Step 2: Generate unit tests based on the solution
        print(f"        🧪 Generating unit tests...")
        tests_prompt = TESTS_PROMPT.format(
            task_description=task_description,
            function_code=canonical_solution
        )
        tests_inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": tests_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tests_inputs = self.tokenizer(tests_inputs, return_tensors="pt")

        # Generate tests
        with torch.no_grad():
            tests_outputs = self.model.generate(
                **tests_inputs,
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.9,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_id[0] if stop_token_id else self.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        # Decode tests response
        tests_output_ids = tests_outputs[0][len(tests_inputs.input_ids[0]):].tolist()
        tests_response = self.tokenizer.decode(tests_output_ids, skip_special_tokens=True).strip()
        unit_tests = self.parse_tests_output(tests_response)
        
        if not unit_tests:
            print(f"        ❌ Failed to generate tests")
            return {'canonical_solution': canonical_solution, 'unit_tests': ''}
        
        print(f"        ✅ Tests generated successfully")
        
        return {
            'canonical_solution': canonical_solution,
            'unit_tests': unit_tests
        }
    
    def parse_solution_test_output(self, response: str) -> Dict[str, str]:
        """
        Parse the generated response to extract canonical solution and unit tests.
        
        Args:
            response: The generated response from the model
            
        Returns:
            Dictionary with 'canonical_solution' and 'unit_tests' keys
        """
        result = {
            'canonical_solution': '',
            'unit_tests': ''
        }
        import re
        # Split by sections
        if 'CANONICAL_SOLUTION:' in response and 'UNIT_TESTS:' in response:
            parts = response.split('CANONICAL_SOLUTION:')
            if len(parts) > 1:
                solution_part = parts[1].split('UNIT_TESTS:')[0]
                tests_part = parts[1].split('UNIT_TESTS:')[1] if 'UNIT_TESTS:' in parts[1] else ''
                
                # Extract solution (everything between ```python and ```)
                solution_match = re.search(r'```python\s*(.*?)\s*```', solution_part, re.DOTALL)
                if solution_match:
                    result['canonical_solution'] = solution_match.group(1).strip()
                
                # Extract tests (everything between ```python and ```)
                tests_match = re.search(r'```python\s*(.*?)\s*```', tests_part, re.DOTALL)
                if tests_match:
                    result['unit_tests'] = tests_match.group(1).strip()
        
        return result
    
    def parse_solution_output(self, response: str) -> str:
        """
        Parse the generated response to extract the canonical solution.
        
        Args:
            response: The generated response from the model
            
        Returns:
            The extracted solution code or empty string if parsing fails
        """
        import re
        
        # Extract everything before STOP
        if "STOP" in response:
            response = response.split("STOP")[0].strip()
        
        # Extract solution (everything between ```python and ```)
        solution_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1).strip()
            # Validate that it contains a function definition
            if 'def task_func' in solution:
                return solution
        
        # If no code block found, try to extract function definition directly
        lines = response.split('\n')
        solution_lines = []
        in_function = False
        
        for line in lines:
            if 'def task_func' in line:
                in_function = True
            if in_function:
                solution_lines.append(line)
                # Check if we've reached the end of the function
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    if 'def ' in line and 'def task_func' not in line:
                        break
        
        if solution_lines:
            solution = '\n'.join(solution_lines).strip()
            if 'def task_func' in solution:
                return solution
        
        return ''
    
    def parse_tests_output(self, response: str) -> str:
        """
        Parse the generated response to extract the unit tests.
        
        Args:
            response: The generated response from the model
            
        Returns:
            The extracted test code as a list of assert statements, or empty string if parsing fails
        """
        import re
        
        # Extract everything before STOP
        if "STOP" in response:
            response = response.split("STOP")[0].strip()
        
        # Extract tests (everything between ```python and ```)
        tests_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if tests_match:
            tests = tests_match.group(1).strip()
        else:
            # If no code block found, try to extract assert statements directly
            lines = response.split('\n')
            test_lines = []
            
            for line in lines:
                if 'assert' in line.strip():
                    test_lines.append(line)
            
            if test_lines:
                tests = '\n'.join(test_lines).strip()
            else:
                return ''
        
        # Validate that it contains assert statements
        if 'assert' not in tests:
            return ''
        
        # Clean up the tests: remove comments and extract only assert statements
        lines = tests.split('\n')
        assert_statements = []
        
        for line in lines:
            line = line.strip()
            if line and 'assert' in line:
                # Remove inline comments
                if '#' in line:
                    line = line.split('#')[0].strip()
                if line:  # Make sure we still have content after removing comments
                    assert_statements.append(line)
        
        if assert_statements:
            return assert_statements
        else:
            return []
    
    def _parse_single_version(self, response: str) -> str:
        """Parse the single alternative-intent version from the generated response."""
        # Extract everything before STOP
        if "STOP" in response:
            response = response.split("STOP")[0].strip()
        
        # Remove "Modified:" prefix if present
        if response.startswith("Modified:"):
            response = response[9:].strip()
        
        return response
    
    def parse_ambiguity_output(self, response: str) -> Dict[str, List[str]]:
        """
        Parse the generated response to extract ambiguous prompt and true intent prompts.
        
        Args:
            response: The generated response from the model
            
        Returns:
            Dictionary with 'ambiguous_prompt' and 'true_intent_prompts' keys
        """
        # Extract everything before STOP
        if "STOP" in response:
            response = response.split("STOP")[0].strip()
        
        # Initialize result
        result = {
            'ambiguous_prompt': '',
            'true_intent_prompts': []
        }
        
        # Split by lines and parse each section
        lines = response.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.startswith('Ambiguous prompt:'):
                current_section = 'ambiguous'
                # Extract the content after the colon
                content = line[len('Ambiguous prompt:'):].strip()
                if content:
                    result['ambiguous_prompt'] = content
            elif line.startswith('True intent prompt'):
                current_section = 'true_intent'
                # Extract the content after the colon
                content = line[len('True intent prompt'):].strip()
                if ':' in content:
                    content = content.split(':', 1)[1].strip()
                if content:
                    result['true_intent_prompts'].append(content)
            elif current_section == 'ambiguous' and result['ambiguous_prompt']:
                # Continue reading ambiguous prompt if it spans multiple lines
                result['ambiguous_prompt'] += ' ' + line
            elif current_section == 'true_intent' and result['true_intent_prompts']:
                # Continue reading true intent prompt if it spans multiple lines
                result['true_intent_prompts'][-1] += ' ' + line
        
        # Clean up the prompts
        result['ambiguous_prompt'] = result['ambiguous_prompt'].strip()
        result['true_intent_prompts'] = [p.strip() for p in result['true_intent_prompts'] if p.strip()]
        
        return result
    
    def filter_similar_ambiguity_sets(self, ambiguity_results: List[Dict[str, List[str]]], threshold: float = None) -> Tuple[List[Dict[str, List[str]]], List[int]]:
        """
        Filter out ambiguity sets that are too similar based on sentence embeddings.
        
        Args:
            ambiguity_results: List of ambiguity result dictionaries
            threshold: Similarity threshold (uses self.similarity_threshold if None)
            
        Returns:
            Tuple of (filtered_results, kept_indices)
        """
        if not self.use_similarity_filtering or not self.sentence_transformer or len(ambiguity_results) <= 1:
            return ambiguity_results, list(range(len(ambiguity_results)))
        
        if threshold is None:
            threshold = self.similarity_threshold
        
        print(f"  Filtering {len(ambiguity_results)} ambiguity sets with similarity threshold: {threshold}")
        
        try:
            # Generate embeddings for ambiguous prompts
            ambiguous_prompts = [result['ambiguous_prompt'] for result in ambiguity_results]
            embeddings = self.sentence_transformer.encode(ambiguous_prompts, convert_to_tensor=True)
            
            # Calculate cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
            
            # Filter ambiguity sets based on similarity
            kept_indices = [0]  # Always keep the first set
            filtered_results = [ambiguity_results[0]]
            
            for i in range(1, len(ambiguity_results)):
                # Check similarity with all previously kept sets
                max_similarity = max(similarity_matrix[i][j] for j in kept_indices)
                
                if max_similarity < threshold:
                    # This set is sufficiently different, keep it
                    kept_indices.append(i)
                    filtered_results.append(ambiguity_results[i])
                    print(f"    Kept ambiguity set {i+1} (max similarity: {max_similarity:.3f})")
                else:
                    print(f"    Filtered out ambiguity set {i+1} (max similarity: {max_similarity:.3f} >= {threshold})")
            
            print(f"  Kept {len(filtered_results)}/{len(ambiguity_results)} ambiguity sets after similarity filtering")
            return filtered_results, kept_indices
            
        except Exception as e:
            print(f"  Error in similarity filtering: {e}, returning all results")
            return ambiguity_results, list(range(len(ambiguity_results)))


def generate_ambiguous_dataset(
    generator: AmbiguityGenerator,
    num_samples: int = 50,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    num_return_sequences: int = 3,
    interactive: bool = False,
    auto_confirm: bool = False,
    max_solution_retries: int = 3
) -> List[Dict[str, Any]]:
    """Generate subtly modified, more ambiguous MBPP task descriptions."""
    print(f"Generating subtly ambiguous MBPP dataset with {num_samples} samples...")

    # Load MBPP problems
    problems = load_dataset("mbpp")
    test_problems = problems["test"]
    test_problems = test_problems.select(range(num_samples))

    entries = []

    for i, problem in enumerate(tqdm(test_problems, desc="Generating subtle ambiguities")):
        print(f"\n  Problem {i+1}/{len(test_problems)}: {problem.get('task_id', 'Unknown')}")

        original_task = problem["text"]
        
        with timer(f"Subtle ambiguity generation for problem {i+1}", verbose=True):
            # Generate structured ambiguity output
            ambiguity_results = generator.generate_ambiguities(
                original_task, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences
            )

        if ambiguity_results and len(ambiguity_results) > 0:
                # Apply similarity filtering to ensure diversity
            filtered_ambiguity_results, kept_indices = generator.filter_similar_ambiguity_sets(ambiguity_results, threshold=0.98)
            
            print(f"\n    📝 Generated {len(ambiguity_results)} ambiguity sets, kept {len(filtered_ambiguity_results)} after similarity filtering:")
            print(f"      Original: {original_task[:80]}...")
            
            # Process each filtered ambiguity result
            for result_idx, ambiguity_result in enumerate(filtered_ambiguity_results):
                ambiguous_prompt = ambiguity_result['ambiguous_prompt']
                true_intent_prompts = ambiguity_result['true_intent_prompts']
                
                print(f"\n    🔍 Processing Ambiguity Set {result_idx + 1}:")
                print(f"      Ambiguous: {ambiguous_prompt[:80]}...")
                print(f"      True Intents: {len(true_intent_prompts)} versions")
                
                # Interactive confirmation for each ambiguity set
                confirmed_ambiguity = False
                    if auto_confirm:
                    confirmed_ambiguity = True
                    print(f"    ✅ Auto-confirmed Ambiguity Set {result_idx + 1}")
                    elif interactive:
                        while True:
                        response = input(f"\n    ✅ Add this ambiguity set to dataset? (y/n/s=skip all remaining for this task): ").strip().lower()
                            if response in ['y', 'yes']:
                            confirmed_ambiguity = True
                            print(f"    ✅ Added Ambiguity Set {result_idx + 1}")
                                break
                            elif response in ['n', 'no']:
                            print(f"    ❌ Skipped Ambiguity Set {result_idx + 1}")
                                break
                            elif response in ['s', 'skip']:
                            print(f"    ⏭️  Skipping all remaining ambiguity sets for this task")
                                break
                            else:
                                print(f"    ❓ Please enter 'y', 'n', or 's'")
                        
                        if response in ['s', 'skip']:
                            break
                    else:
                    # Default: confirm all ambiguity sets without prompting
                    confirmed_ambiguity = True
                    print(f"    ✅ Added Ambiguity Set {result_idx + 1} (default)")
                
                if confirmed_ambiguity:
                    # Create paired entries: ambiguous prompt paired with each true intent
                    for intent_idx, true_intent_prompt in enumerate(true_intent_prompts):
                        print(f"      Generating solution and tests for intent {intent_idx + 1}...")
                        
                        # Generate canonical solution and unit tests for this true intent with retry logic
                        canonical_solution = ''
                        unit_tests = ''
                        
                        for retry in range(max_solution_retries):
                            with timer(f"Solution generation for intent {intent_idx + 1} (attempt {retry + 1})", verbose=False):
                                solution_test_result = generator.generate_solution_and_tests(
                                    true_intent_prompt,
                                    max_new_tokens=1024,
                                    temperature=0.3
                                )
                            
                            canonical_solution = solution_test_result.get('canonical_solution', '')
                            unit_tests = solution_test_result.get('unit_tests', '')
                            
                            # Validate that we got a proper solution and tests
                            has_function = 'def task_func' in canonical_solution
                            has_asserts = unit_tests and all('assert' in test for test in unit_tests)
                            
                            if canonical_solution and unit_tests and has_function and has_asserts:
                                # Count clean assert statements (unit_tests is now a list)
                                test_count = len(unit_tests)
                                print(f"        ✅ Generated solution and {test_count} test cases (attempt {retry + 1})")
                                break
                            else:
                                missing = []
                                if not canonical_solution:
                                    missing.append("solution")
                                if not unit_tests:
                                    missing.append("tests")
                                if not has_function:
                                    missing.append("function definition")
                                if not has_asserts:
                                    missing.append("assert statements")
                                print(f"        ⚠️  Attempt {retry + 1} failed: missing {', '.join(missing)}")
                                if retry < max_solution_retries - 1:
                                    print(f"        🔄 Retrying...")
                                    # Increase temperature and add small delay for retries
                                    import time
                                    time.sleep(0.5)  # Small delay between retries
                                    
                                    # Progressive temperature increase: 0.3 -> 0.5 -> 0.7
                                    retry_temperature = 0.3 + (retry + 1) * 0.2
                                    solution_test_result = generator.generate_solution_and_tests(
                                        true_intent_prompt,
                                        max_new_tokens=1024,
                                        temperature=retry_temperature
                                                                          )
                          
                          # If all retries failed, use original solution and tests
                        if not canonical_solution or not unit_tests:
                            print(f"        ❌ All {max_solution_retries} attempts failed, using original solution and tests")
                            canonical_solution = problem["code"]
                            # Convert original test_list to list format if it's not already
                            if isinstance(problem["test_list"], str):
                                unit_tests = [line.strip() for line in problem["test_list"].split('\n') if line.strip() and 'assert' in line.strip()]
                            else:
                                unit_tests = problem["test_list"]
                        
                        # Create standardized reward model with generated solution and tests
                        # Convert unit_tests list to string format for StandardizedRewardModel
                        unit_tests_str = '\n'.join(unit_tests) if isinstance(unit_tests, list) else str(unit_tests)
                        
                reward_model = StandardizedRewardModel(
                            ground_truth=[canonical_solution],
                    style="code",
                            unit_tests=[unit_tests_str],
                    libs=[],
                )

                        # Entry with ambiguous prompt as question and true intent as explicit task
                        ambiguous_entry = {
                            "data_source": f"mbpp_ambiguous_{i}_set_{result_idx}_intent_{intent_idx}",
                            "prompt": ambiguous_prompt,
                            "answer": canonical_solution,  # Use generated canonical solution
                    "reward_model": reward_model.to_dict(),
                    "system_instruction_type": "default",
                    "extra_info": {
                        "split": "test",
                            "index": i,
                                "ambiguity_set_index": result_idx,
                                "intent_index": intent_idx,
                            "original_task": original_task,
                            "original_task_id": problem.get("task_id", "Unknown"),
                                "question": ambiguous_prompt,  # Store the ambiguous prompt as the question
                                "explicit_task": true_intent_prompt,  # Store the true intent as explicit task
                                "answer": [canonical_solution],
                                "ambiguity_type": "subtle_ambiguity",
                                "true_intent_prompt": true_intent_prompt,
                                "generated_solution": canonical_solution,
                                "generated_tests": unit_tests,
                            },
                        }

                        entries.append(ambiguous_entry)
                    
                    print(f"    ✅ Successfully added {len(true_intent_prompts)} paired entries for Ambiguity Set {result_idx + 1}")
            
            print(f"    ✅ Total entries added for this problem: {len([e for e in entries if e['extra_info']['index'] == i])}")
        else:
            print(f"    ⚠️  Warning: No ambiguity results generated for problem {i+1}")

    print(f"\n✅ Successfully generated {len(entries)} subtly ambiguous task descriptions")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Generate subtly modified, more ambiguous MBPP task descriptions"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name for generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_samples", type=int, default=25, help="Number of problems to process"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--device_map", type=str, default="auto", help="Device mapping for model"
    )
    parser.add_argument(
        "--num_return_sequences", type=int, default=20, help="Number of generation sequences for diversity (will be filtered)"
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.98, help="Similarity threshold for filtering (0.0-1.0)"
    )
    parser.add_argument(
        "--disable_similarity_filtering", action="store_true", help="Disable similarity filtering"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Enable interactive confirmation for each generated version"
    )
    parser.add_argument(
        "--auto_confirm", action="store_true", help="Automatically confirm all versions (overrides interactive mode)"
    )
    parser.add_argument(
        "--max_solution_retries", type=int, default=3, 
        help="Maximum number of retries for solution generation (default: 3)"
    )

    args = parser.parse_args()

    print(f"🚀 MBPP SUBTLE AMBIGUITY GENERATOR")
    print(f"Model: {args.model_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Return Sequences: {args.num_return_sequences}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Initialize generator
    generator = AmbiguityGenerator(
        model_name=args.model_name,
        device_map=args.device_map,
        use_similarity_filtering=not args.disable_similarity_filtering,
        similarity_threshold=args.similarity_threshold
    )

    ambiguous_data = generate_ambiguous_dataset(
        generator,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        num_return_sequences=args.num_return_sequences,
        interactive=args.interactive,
        auto_confirm=args.auto_confirm,
        max_solution_retries=args.max_solution_retries
    )

    if ambiguous_data:
        # Save as parquet files
        filename_prefix = f"sft/mbpp_subtly_ambiguous"
        save_to_parquet(ambiguous_data, "", args.output_dir, filename_prefix)

        # Also save as JSONL
        jsonl_file = os.path.join(args.output_dir, f"{filename_prefix}.jsonl")
        save_jsonl(ambiguous_data, jsonl_file)

        print(f"\n{'='*60}")
        print("MBPP SUBTLE AMBIGUITY GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Dataset: MBPP Subtly Ambiguous Tasks")
        print(f"Total modifications generated: {len(ambiguous_data)}")

        # Show examples and summary
        print(f"\n📋 Example ambiguity pairs:")
            
            # Show first few examples
            for i, entry in enumerate(ambiguous_data[:3]):
                print(f"\n  🎯 Task {i + 1}:")
                original_task = entry['extra_info']['original_task']
            ambiguous_prompt = entry['extra_info']['question']
            true_intent = entry['extra_info']['explicit_task']
            canonical_solution = entry['extra_info'].get('generated_solution', 'N/A')
            unit_tests = entry['extra_info'].get('generated_tests', 'N/A')
            print(f"    Original: {original_task[:80]}...")
            print(f"    Ambiguous: {ambiguous_prompt[:80]}...")
            print(f"    True Intent: {true_intent[:80]}...")
            print(f"    Solution: {canonical_solution[:80] if isinstance(canonical_solution, str) else 'Generated'}...")
            # Count clean assert statements
            if isinstance(unit_tests, list) and unit_tests:
                test_count = len(unit_tests)
                print(f"    Tests: {test_count} assert statements")
            else:
                print(f"    Tests: 0 assert statements")
            
            # Show summary statistics
            print(f"\n📊 Generation Summary:")
            print(f"  Total entries generated: {len(ambiguous_data)}")
            print(f"  Original tasks processed: {len(set(entry['extra_info']['index'] for entry in ambiguous_data))}")
        print(f"  Total ambiguity sets: {len(set((entry['extra_info']['index'], entry['extra_info']['ambiguity_set_index']) for entry in ambiguous_data))}")
        print(f"  Average intents per ambiguity set: {len(ambiguous_data) / len(set((entry['extra_info']['index'], entry['extra_info']['ambiguity_set_index']) for entry in ambiguous_data)):.1f}")
        
        # Count generated solutions and tests
        generated_solutions = sum(1 for entry in ambiguous_data if entry['extra_info'].get('generated_solution') and entry['extra_info']['generated_solution'] != entry['extra_info']['original_task'])
        generated_tests = sum(1 for entry in ambiguous_data if entry['extra_info'].get('generated_tests') and isinstance(entry['extra_info']['generated_tests'], list) and len(entry['extra_info']['generated_tests']) > 0)
        print(f"  Generated solutions: {generated_solutions}/{len(ambiguous_data)}")
        print(f"  Generated test suites: {generated_tests}/{len(ambiguous_data)}")
        print(f"  Max solution retries: {args.max_solution_retries}")
        print(f"  Generation method: Two-step (solution → tests)")
        
            print(f"  Similarity filtering: {'enabled' if generator.use_similarity_filtering else 'disabled'}")
            if generator.use_similarity_filtering:
                print(f"  Similarity threshold: {generator.similarity_threshold}")
            print(f"  Interactive mode: {'enabled' if args.interactive else 'disabled'}")
            print(f"  Auto-confirm: {'enabled' if args.auto_confirm else 'disabled'}")
            
            # Show data structure info
            example = ambiguous_data[0]
            print(f"\n📋 Data Structure:")
            print(f"  Data Source: {example['data_source']}")
            print(f"  Reward Model Style: {example['reward_model']['style']}")
            print(f"  System Instruction Type: {example['system_instruction_type']}")
        print(f"  Ambiguity Type: {example['extra_info']['ambiguity_type']}")
        print(f"  Has Generated Solution: {'Yes' if example['extra_info'].get('generated_solution') else 'No'}")
        print(f"  Has Generated Tests: {'Yes' if example['extra_info'].get('generated_tests') else 'No'}")
    else:
        print("❌ No data generated")


if __name__ == "__main__":
    main() 