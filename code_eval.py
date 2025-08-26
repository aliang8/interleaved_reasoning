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
from helpers import create_prompts_dataproto, extract_solution_from_response
from visualization.create_mbpp_html import create_mbpp_html_visualization


class CodeDatasetEvaluator(BaseEvaluator):
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

            # Extract function signature and imports from canonical solution
            canonical_solution = example.get("canonical_solution", example.get("code", ""))
            function_signature, import_statements = self._extract_function_signature_and_imports(canonical_solution, original_entry_point)

            examples.append({
                "id": i,
                "prompt": example["prompt"],
                "test_list": example["test_list"],
                "entry_point": function_signature.split('(')[0].split()[-1],  # Use extracted function name
                "original_entry_point": original_entry_point,  # Keep original for reference
                "canonical_solution": canonical_solution,
                "description": example.get("description", ""),
                "function_signature": function_signature,
                "import_statements": import_statements,
                "test_imports": example.get("test_imports", []),  # Extract test imports if available
            })

            # Debug output for first few examples
            if i < 3:
                print(f"  Example {i + 1}: {original_entry_point} → {function_signature.split('(')[0].split()[-1]}")
                print(f"    Function signature: {function_signature}")
                print(f"    Import statements: {import_statements}")
                print(f"    Test: {example['test_list'][0] if example['test_list'] else 'No tests'}")

        return examples
    
    def _extract_function_name_from_tests(self, test_list):
        """Extract the function name from test cases using regex."""
        if not test_list:
            return "task_func"

        # Look for function calls in test cases
        # Pattern: function_name(args) or function_name(args, args)
        # We want to find the innermost function call, not library functions like math.square()
        function_pattern = r"(\w+)\s*\([^)]*\)"

        for test in test_list:
            # Find all function calls in this test
            matches = re.findall(function_pattern, test)
            if matches:
                # Look for the first function call that's not a library function
                for match in matches:
                    function_name = match
                    # Skip common Python keywords, built-ins, and library functions
                if function_name not in [
                    "assert", "print", "len", "str", "int", "float",
                        "list", "dict", "set", "tuple", "math", "random", "re",
                        "square", "sqrt", "pow", "abs", "max", "min", "sum"
                ]:
                    return function_name
                
                # If we found library functions but no custom functions, look for nested calls
                # Pattern to find function calls inside other function calls: func(inner_func(args))
                nested_pattern = r"\(\s*(\w+)\s*\([^)]*\)"
                nested_matches = re.findall(nested_pattern, test)
                if nested_matches:
                    for nested_func in nested_matches:
                        if nested_func not in [
                            "assert", "print", "len", "str", "int", "float",
                            "list", "dict", "set", "tuple", "math", "random", "re",
                            "square", "sqrt", "pow", "abs", "max", "min", "sum"
                        ]:
                            return nested_func

        return "task_func"
    
    def _extract_function_signature_and_imports(self, canonical_solution: str, fallback_function_name: str) -> tuple:
        """Extract function signature and import statements from canonical solution."""
        import_statements = []
        function_signature = f"def {fallback_function_name}():"  # Default fallback
        
        if not canonical_solution.strip():
            return function_signature, import_statements
        
        lines = canonical_solution.strip().split('\n')
        
        # Extract import statements (lines starting with 'import' or 'from')
        for line in lines:
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                import_statements.append(line)
        
        # Extract function definition (lines starting with 'def ')
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                # Extract the function signature (everything up to the colon)
                if ':' in line:
                    function_signature = line[:line.index(':') + 1]
                    break
        
        return function_signature, import_statements
    
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """Create MBPP prompts with the required prefix."""
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]

            # Create prompts with prefix
            prompts = []
            for example in batch_examples:
                # Get the extracted function signature and imports
                function_signature = example.get("function_signature", "def task_func():")
                import_statements = example.get("import_statements", [])
                
                # Build the prefix with actual imports and function signature from ground truth
                prefix_parts = ["You should write self-contained code starting with:"]
                if import_statements:
                    prefix_parts.extend(import_statements)
                    prefix_parts.append("")  # Empty line after imports
                
                prefix_parts.append("```")
                prefix_parts.append(function_signature)
                prefix_parts.append("```")
                prefix_parts.append("")  # Empty line before the prompt
                
                prefix = "\n".join(prefix_parts)
                
                # Create the enhanced prompt
                enhanced_prompt = f"{example['prompt']}\n\n{prefix}"
                prompts.append(enhanced_prompt)

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

        # If no code blocks found, return empty string
        return ""
    
    def _evaluate_batch(self, examples: List[Dict], extracted_contents: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of code solutions against test cases."""
        batch_evaluations = []
        
        for example, extracted_content in zip(examples, extracted_contents):
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
                batch_evaluations.append(results)
                continue

            if not example["test_list"] or len(example["test_list"]) == 0:
                results["execution_error"] = "No test cases provided"
                batch_evaluations.append(results)
                continue

            # Create a safe execution environment
            local_vars = {}

            # Execute test imports first if available
            if example.get("test_imports"):
                # print(f"  Executing {len(example['test_imports'])} test imports...")
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
                batch_evaluations.append(results)
                continue
                
            # Check if the function exists
            if example["entry_point"] not in local_vars:
                results["execution_error"] = (
                    f"Function {example['entry_point']} not found in generated code"
                )
                batch_evaluations.append(results)
                continue

            # Run each test
            # print(f"  Running {len(example['test_list'])} tests...")
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

            batch_evaluations.append(results)
        
        return batch_evaluations
    
    def _create_base_result(self, example: Dict, response_text: str, extracted_content: str,
                           evaluation: Dict, interleaved_components: List[Dict],
                           task_completed: bool, num_tokens: int, ttft_ratio: float,
                           total_tokens_generated: int = None, tokens_to_first_answer: int = None) -> Dict:
        """Create the base result structure for MBPP problems.
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
                'test_imports': evaluation.get('test_imports', [])
            }
        
        return base_result
    
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
            'function_signature': example.get("function_signature", "def task_func():"),
            'import_statements': example.get("import_statements", []),
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
        # Get the template directory from base class
        template_dir = super()._generate_html_visualization(output_dir)
        
        # Generate filename: rollout_name_response_length.html
        filename = self.rollout_config['rollout']['name']
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{self.rollout_config['rollout']['n_candidates']}"
        
        filename += f"_{self.cfg.response_length}"
        
        html_file = template_dir / f"{filename}.html"
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
    evaluator = CodeDatasetEvaluator(cfg)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    main() 