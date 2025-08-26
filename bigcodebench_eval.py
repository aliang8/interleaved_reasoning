#!/usr/bin/env python3
"""
Standardized BigCodeBench Evaluation Script

This script uses the BaseEvaluator framework to provide a clean,
maintainable implementation of BigCodeBench code generation evaluation.
"""

import re
from typing import List, Dict, Any
from pathlib import Path
from datasets import load_dataset

import pyrallis
from evaluation_base import BaseEvaluator, EvaluationConfig
from helpers import create_prompts_dataproto, extract_solution_from_response
from verl.workers.code_evaluator.code_evaluator import CodeEvaluator
print("  Using verl code evaluator for BigCodeBench evaluation")

class BigCodeBenchEvaluator(BaseEvaluator):
    """BigCodeBench code generation evaluation implementation using the base framework."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.total_tests = 0
        self.total_passed_tests = 0
        self.pass_at_1_count = 0
    
    def _load_dataset(self) -> List[Dict]:
        """Load BigCodeBench dataset and return prompts 100-150."""
        print("Loading BigCodeBench dataset...")

        # Load the dataset
        dataset = load_dataset("bigcode/bigcodebench")["v0.1.4"]
        filtered_data = dataset.select(range(100, 400))  # 100-300

        print(f"✓ Loaded BigCodeBench test split with {len(filtered_data)} examples (prompts 100-300)")

        # Convert to list of dictionaries for easier processing
        examples = []
        for i, example in enumerate(filtered_data):
            examples.append({
                "id": i + 99,  # Keep original prompt number (100-150)
                "prompt": example["instruct_prompt"],
                "test_list": example["test"],
                "entry_point": "task_func",  # Use default entry point
                "libs": example.get("libs", ""),  # BigCodeBench specific: libraries needed
            })

            # Debug output for first few examples
            if i < 3:
                print(f"  Example {i + 100}:")
                print(f"    Prompt length: {len(example['instruct_prompt'])}")
                print(f"    Test: {example['test']}")
                print(f"    Libraries: {example.get('libs', '')}")
                print(f"    Test content: {example['test'][:100] if example['test'] else 'No test'}...")

        return examples
    

    
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """Create BigCodeBench prompts."""
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]

            # Use prompts as-is
            prompts = [example['prompt'] for example in batch_examples]

            # Create DataProto for this batch
            prompts_dataproto = create_prompts_dataproto(
                tokenizer=self.tokenizer,  # Use the tokenizer from worker
                questions=prompts,
                max_prompt_length=self.rollout_config["rollout"]["prompt_length"],
                template_type=self.rollout_config["rollout"]["template_type"],
                enable_thinking=self.rollout_config["rollout"]["enable_thinking"],
            )

            yield batch_examples, prompts_dataproto
    
    def _extract_content_from_response(self, response_text: str) -> str:
        """Extract Python code from response text."""
        solution = extract_solution_from_response(response_text, self.rollout_config["rollout"]["template_type"], self.rollout_config["rollout"]["enable_thinking"])

        # Look for Python code blocks
        python_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(python_pattern, solution, re.DOTALL | re.IGNORECASE)

        if matches:
            return matches[0].strip()

        # Fallback: look for any code block
        code_pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(code_pattern, solution, re.DOTALL)

        if matches:
            return matches[0].strip()

        return solution
    
    def _evaluate_batch(self, examples: List[Dict], extracted_contents: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of code solutions against test cases using the code evaluator."""
        batch_evaluations = []
        
        # Use the verl code evaluator
        # Create a mock config for the code evaluator
        from omegaconf import OmegaConf
        mock_config = OmegaConf.create({
            "max_concurrent": 4,
            "execute_sequential": True,
            "autorater_service_url": "http://localhost:8000"  # Default, can be overridden
        })
        
        # Create code evaluator instance
        code_evaluator = CodeEvaluator(mock_config, self.tokenizer)
        
        # Prepare data for code evaluator
        prompts = [example["prompt"] for example in examples]
        rm_infos = []
        for example in examples:
            rm_info = {
                "tests": [example["test_list"]],  # test_list is a string, not a list
                "libs": [example.get("libs", "")]  # libs is a string, not a list
            }
            rm_infos.append(rm_info)
        
        batch_indices = list(range(len(examples)))
        
        # Evaluate using code evaluator
        code_rewards = code_evaluator.evaluate_code(
            answers=extracted_contents,
            prompts=prompts,
            rm_infos=rm_infos,
            batch_indices=batch_indices
        )
        
        # Get sandbox results for detailed logging
        _, sandbox_results = code_evaluator._evaluate_code_helper(extracted_contents, rm_infos, batch_indices)
        
        # Convert rewards to evaluation results
        for i, example in enumerate(examples):
            # Get sandbox result for this example
            sandbox_result = sandbox_results[i] if i < len(sandbox_results) else None
            
            # Compute total tests by parsing sandbox output (similar to code evaluator)
            total_tests = 0
            if sandbox_result and len(sandbox_result) > 0:
                for test_result in sandbox_result:
                    if test_result:
                        stderr = test_result.get("stderr", "")
                        if "Ran" in stderr:
                            # Parse "Ran X tests in Y seconds" to get total test count
                            import re
                            ran_match = re.search(r"Ran (\d+) tests? in", stderr)
                            if ran_match:
                                total_tests = max(total_tests, int(ran_match.group(1)))
            
            # If we couldn't parse total tests from sandbox, default to 1
            if total_tests == 0:
                total_tests = 1
            
            # Calculate tests passed based on unit test pass rate
            tests_passed = int(code_rewards["unit_test_pass_rate"][i] * total_tests) if code_rewards["unit_test_pass_rate"][i] > 0 else 0
            tests_failed = total_tests - tests_passed
            
            results = {
                "code": extracted_contents[i],
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "total_tests": total_tests,
                "test_results": [],  # Code evaluator doesn't provide individual test results
                "execution_error": None,
                "libs": example.get("libs", ""),  # Handle as string
                "unit_test_pass_rate": code_rewards["unit_test_pass_rate"][i],
                "pass_at_1": code_rewards["pass@1"][i],
                "sandbox_results": sandbox_results[i] if i < len(sandbox_results) else None
            }
            batch_evaluations.append(results)
            
            # Debug: Print progress for this example
            print(f"    Processed example {i + 1}: {tests_passed}/{total_tests} tests passed")
        
        print(f"  ✓ Code evaluator processed {len(examples)} examples")
        return batch_evaluations
                        
    def _create_base_result(self, example: Dict, response_text: str, extracted_content: str,
                           evaluation: Dict, interleaved_components: List[Dict],
                           task_completed: bool, num_tokens: int, ttft_ratio: float,
                           total_tokens_generated: int = None, tokens_to_first_answer: int = None) -> Dict:
        """Create the base result structure for BigCodeBench problems.
        If evaluation is not yet available, return the base result unchanged.
        """
        base_result = super()._create_base_result(
            example, response_text, extracted_content, evaluation,
            interleaved_components, task_completed, num_tokens, ttft_ratio,
            total_tokens_generated, tokens_to_first_answer
        )
        
        # Only enrich the evaluation structure when batch evaluation has populated it
        if isinstance(evaluation, dict) and 'tests_passed' in evaluation:
            base_result['evaluation'] = {
                'tests_passed': evaluation['tests_passed'],
                'tests_failed': evaluation['tests_failed'],
                'total_tests': evaluation['total_tests'],
                'test_results': evaluation['test_results'],
                'execution_error': evaluation.get('execution_error'),
                'libs': evaluation.get('libs', []),
                'unit_test_pass_rate': evaluation.get('unit_test_pass_rate', 0.0),
                'pass_at_1': evaluation.get('pass_at_1', 0),
                'sandbox_results': evaluation.get('sandbox_results')
            }
        
        return base_result
    
    def _add_task_specific_fields(self, example: Dict, evaluation: Dict) -> Dict[str, Any]:
        """Add BigCodeBench-specific fields to the result."""
        # Track test metrics
        self.total_tests += evaluation["total_tests"]
        self.total_passed_tests += evaluation["tests_passed"]

        # Track pass@1 (if all tests pass, it's a pass@1)
        if (evaluation["total_tests"] > 0 and 
            evaluation["tests_passed"] == evaluation["total_tests"]):
            self.pass_at_1_count += 1
        
        return {
            'test_list': example["test_list"],  # This is a string
            'entry_point': example["entry_point"],
            'libs': example.get("libs", ""),  # This is a string
            'test_cases': example["test_list"],  # Add test cases for visualization (as string)
            'sandbox_results': evaluation.get("sandbox_results")  # Add sandbox results
        }
    
    def _print_progress(self, batch_idx: int, i: int, evaluation: Dict):
        """Print progress for the current BigCodeBench problem."""
        if evaluation.get("unit_test_pass_rate") is not None:
            # Using code evaluator results
            pass_rate = evaluation["unit_test_pass_rate"] * 100
            print(f"  Problem {batch_idx * self.cfg.batch_size + i + 1}: {pass_rate:.1f}% unit test pass rate ({evaluation['tests_passed']}/{evaluation['total_tests']})")
        else:
            # Fallback to basic test counting
            pass_rate = (
                (evaluation["tests_passed"] / evaluation["total_tests"] * 100)
                if evaluation["total_tests"] > 0
                else 0
            )
            print(f"  Problem {batch_idx * self.cfg.batch_size + i + 1}: {evaluation['tests_passed']}/{evaluation['total_tests']} tests passed ({pass_rate:.1f}%)")
    
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for filename generation."""
        return "bigcodebench"
    
    def _generate_html_visualization(self, output_dir: Path):
        """Generate HTML visualization for BigCodeBench problems."""
        # Get the template directory from base class
        template_dir = super()._generate_html_visualization(output_dir)
        
        # Generate filename: rollout_name_response_length.html
        filename = self.rollout_config['rollout']['name']
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{self.rollout_config['rollout']['n_candidates']}"
        
        filename += f"_{self.cfg.response_length}"
        
        html_file = template_dir / f"{filename}.html"
        
        # Import and use the custom BigCodeBench HTML visualization function
        try:
            from create_bigcodebench_html import create_bigcodebench_html_visualization
            create_bigcodebench_html_visualization(self.all_results, html_file)
            print(f"🎨 HTML visualization saved to {html_file}")
        except ImportError:
            print(f"🎨 HTML visualization would be saved to {html_file}")
            print("   (Custom BigCodeBench HTML visualization module not found)")
    
    def _print_task_specific_metrics(self):
        """Print BigCodeBench-specific metrics."""
        if hasattr(self, 'examples') and self.examples:
            unit_test_pass_rate = (
                (self.total_passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            )
            pass_at_1_rate = (self.pass_at_1_count / len(self.examples) * 100) if self.examples else 0
            
            print(f"Unit Test Pass Rate: {unit_test_pass_rate:.1f}% ({self.total_passed_tests}/{self.total_tests})")
            print(f"Pass@1 Rate: {pass_at_1_rate:.1f}% ({self.pass_at_1_count}/{len(self.examples)})")
        else:
            print("No examples loaded, cannot compute metrics")


def main():
    """Main BigCodeBench evaluation function."""
    print("=== BigCodeBench Code Generation Evaluation ===\n")
    
    # Parse configuration using pyrallis
    cfg = pyrallis.parse(config_class=EvaluationConfig)
    
    # Create and run evaluator
    evaluator = BigCodeBenchEvaluator(cfg)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    main() 