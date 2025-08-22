#!/usr/bin/env python3
"""
Standardized MBPP (Mostly Basic Python Problems) Evaluation Script

This script uses the BaseEvaluator framework to provide a clean,
maintainable implementation of code generation evaluation.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
from datasets import load_dataset

import pyrallis
from evaluation_base import BaseEvaluator, EvaluationConfig
from helpers import create_prompts_dataproto
from visualization.create_mbpp_html import create_mbpp_html_visualization


class MBPPEvaluator(BaseEvaluator):
    """MBPP code generation evaluation implementation using the base framework."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.total_tests = 0
        self.total_passed_tests = 0
        self.pass_at_1_count = 0
    
    def _load_dataset(self) -> List[Dict]:
        """Load MBPP dataset and return the sanitized subset test split."""
        print("Loading MBPP dataset...")

        # Load the dataset
        dataset = load_dataset("mbpp", name="sanitized")

        # Get the sanitized subset test split
        test_data = dataset["test"]

        print(f"✓ Loaded MBPP test split with {len(test_data)} examples")

        # Convert to list of dictionaries for easier processing
        examples = []
        for i, example in enumerate(test_data):
            # Extract the original function name from test cases
            original_entry_point = self._extract_function_name_from_tests(example["test_list"])

            # Convert test cases to use task_func instead of original function names
            converted_test_list = []
            for test in example["test_list"]:
                # Replace the original function name with task_func in test cases
                converted_test = test.replace(f"{original_entry_point}(", "task_func(")
                converted_test_list.append(converted_test)

            examples.append({
                "id": i,
                "prompt": example["prompt"],
                "test_list": converted_test_list,
                "entry_point": "task_func",  # Always use task_func
                "original_entry_point": original_entry_point,  # Keep original for reference
                "canonical_solution": example.get("canonical_solution", ""),
                "description": example.get("description", ""),
                "test_imports": example.get("test_imports", []),  # Extract test imports if available
            })

            # Debug output for first few examples
            if i < 3:
                print(f"  Example {i + 1}: {original_entry_point} → task_func")
                print(f"    Test: {example['test_list'][0] if example['test_list'] else 'No tests'}")
                print(f"    Converted: {converted_test_list[0] if converted_test_list else 'No tests'}")
                if example.get("test_imports"):
                    print(f"    Test imports: {example['test_imports']}")
                else:
                    print(f"    Test imports: None")

        return examples
    
    def _extract_function_name_from_tests(self, test_list):
        """Extract the function name from test cases using regex."""
        if not test_list:
            return "task_func"

        # Look for function calls in test cases
        # Pattern: function_name(args) or function_name(args, args)
        function_pattern = r"(\w+)\s*\([^)]*\)"

        for test in test_list:
            match = re.search(function_pattern, test)
            if match:
                function_name = match.group(1)
                # Skip common Python keywords and built-ins
                if function_name not in [
                    "assert", "print", "len", "str", "int", "float",
                    "list", "dict", "set", "tuple"
                ]:
                    return function_name

        return "task_func"
    
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """Create MBPP prompts with the required prefix."""
        # Add the required prefix to each prompt
        prefix = "You should write self-contained code starting with:\n```\ndef task_func(arg1, arg2, ...):\n```\n\n"

        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]

            # Create prompts with prefix
            prompts = []
            for example in batch_examples:
                # Create a more specific prompt that emphasizes the function name
                original_func_name = example.get("original_entry_point", "unknown_function")
                enhanced_prompt = f"{prefix}The function should be named 'task_func' and take 'args' as a parameter.\n\n{example['prompt']}"
                prompts.append(enhanced_prompt)

            # Create DataProto for this batch
            prompts_dataproto = create_prompts_dataproto(
                tokenizer=self.tokenizer,  # Use the tokenizer from worker
                questions=prompts,
                max_prompt_length=self.config["rollout"]["prompt_length"],
                template_type=self.config["rollout"]["template_type"],
                enable_thinking=self.config["rollout"]["enable_thinking"],
            )

            yield batch_examples, prompts_dataproto
    
    def _extract_content_from_response(self, response_text: str) -> str:
        """Extract Python code from response text."""
        if self.config["rollout"]["template_type"] == "plan_first":
            # For plan_first template, extract the last answer block
            answer_pattern = r"<answer>(.*?)</answer>"
            answer_matches = re.findall(
                answer_pattern, response_text, re.DOTALL | re.IGNORECASE
            )

            if answer_matches:
                # Use the last answer block
                last_answer = answer_matches[-1].strip()

                # Look for Python code blocks within the last answer
                python_pattern = r"```python\s*(.*?)\s*```"
                matches = re.findall(python_pattern, last_answer, re.DOTALL | re.IGNORECASE)

                if matches:
                    return matches[0].strip()

                # Fallback: look for any code block
                code_pattern = r"```\s*(.*?)\s*```"
                matches = re.findall(code_pattern, last_answer, re.DOTALL)

                if matches:
                    return matches[0].strip()

                # If no code blocks found, return the last answer as-is
                return last_answer.strip()

        # For other template types, use the original logic
        # First, extract everything after </think> if it exists
        if "</think>" in response_text:
            parts = response_text.split("</think>")
            if len(parts) > 1:
                response_text = parts[1].strip()

        # Look for Python code blocks
        python_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(python_pattern, response_text, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        # Fallback: look for any code block
        code_pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(code_pattern, response_text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # If no code blocks found, return empty string
        return ""
    
    def _evaluate_content(self, example: Dict, extracted_content: str) -> Dict[str, Any]:
        """Evaluate generated code against test cases."""
        results = {
            "code": extracted_content,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tests": len(example["test_list"]),
            "test_results": [],
            "execution_error": None,
            "test_imports": example.get("test_imports", []),
        }

        if not extracted_content.strip():
            results["execution_error"] = "No code generated"
            return results

        if not example["test_list"] or len(example["test_list"]) == 0:
            results["execution_error"] = "No test cases provided"
            return results

        # Create a safe execution environment
        local_vars = {}

        # Execute test imports first if available
        if example.get("test_imports"):
            print(f"  Executing {len(example['test_imports'])} test imports...")
            for import_stmt in example["test_imports"]:
                try:
                    exec(import_stmt, {}, local_vars)
                    print(f"    ✓ Imported: {import_stmt}")
                except Exception as e:
                    print(f"    ✗ Failed to execute import '{import_stmt}': {e}")
        else:
            print(f"  No test imports to execute")

        # Execute the code
        try:
            exec(extracted_content, {}, local_vars)
        except Exception as e:
            results["execution_error"] = f"Error executing code: {e}"
            return results

        # Check if the function exists
        if example["entry_point"] not in local_vars:
            results["execution_error"] = (
                f"Function {example['entry_point']} not found in generated code"
            )
            return results

        # Run each test
        print(f"  Running {len(example['test_list'])} tests...")
        for i, test in enumerate(example["test_list"]):
            try:
                # Execute the test assertion
                exec(test, {}, local_vars)
                results["test_results"].append(
                    {"test": test, "passed": True, "error": None}
                )
                results["tests_passed"] += 1
            except Exception as e:
                results["test_results"].append(
                    {"test": test, "passed": False, "error": str(e)}
                )
                results["tests_failed"] += 1

        return results
    
    def _add_task_specific_fields(self, example: Dict, evaluation: Dict) -> Dict[str, Any]:
        """Add MBPP-specific fields to the result."""
        # Track test metrics
        self.total_tests += evaluation["total_tests"]
        self.total_passed_tests += evaluation["tests_passed"]

        # Track pass@1 (if all tests pass, it's a pass@1)
        if (evaluation["total_tests"] > 0 and 
            evaluation["tests_passed"] == evaluation["total_tests"]):
            self.pass_at_1_count += 1
        
        return {
            'test_list': example["test_list"],
            'entry_point': example["entry_point"],
            'original_entry_point': example.get("original_entry_point", "unknown"),
            'test_imports': example.get("test_imports", []),
        }
    
    def _print_progress(self, batch_idx: int, i: int, evaluation: Dict):
        """Print progress for the current MBPP problem."""
        pass_rate = (
            (evaluation["tests_passed"] / evaluation["total_tests"] * 100)
            if evaluation["total_tests"] > 0
            else 0
        )
        print(
            f"  Problem {batch_idx * self.cfg.batch_size + i + 1}: {evaluation['tests_passed']}/{evaluation['total_tests']} tests passed ({pass_rate:.1f}%)"
        )
    
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for filename generation."""
        return "mbpp"
    
    def _generate_html_visualization(self, output_dir: Path):
        """Generate HTML visualization for MBPP problems."""
        thinking_tag = "thinking" if ACTOR_ROLLOUT_CONFIG['rollout']['enable_thinking'] else "no_thinking"
        filename = f"{self.cfg.template_type}_{ACTOR_ROLLOUT_CONFIG['rollout']['name']}_{thinking_tag}"
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{ACTOR_ROLLOUT_CONFIG['rollout']['n_candidates']}"
        
        html_file = output_dir / f"{filename}.html"
        create_mbpp_html_visualization(self.all_results, html_file)
        print(f"🎨 HTML visualization saved to {html_file}")
    
    def _print_task_specific_metrics(self):
        """Print MBPP-specific metrics."""
        unit_test_pass_rate = (
            (self.total_passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        )
        pass_at_1_rate = (self.pass_at_1_count / len(self.examples) * 100) if self.examples else 0
        
        print(f"Unit Test Pass Rate: {unit_test_pass_rate:.1f}% ({self.total_passed_tests}/{self.total_tests})")
        print(f"Pass@1 Rate: {pass_at_1_rate:.1f}% ({self.pass_at_1_count}/{len(self.examples)})")


def main():
    """Main MBPP evaluation function."""
    print("=== MBPP Code Generation Evaluation ===\n")
    
    # Parse configuration using pyrallis
    cfg = pyrallis.parse(config_class=EvaluationConfig)
    
    # Create and run evaluator
    evaluator = MBPPEvaluator(cfg)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    main() 