#!/usr/bin/env python3
"""
Standardized Math500/AIME 2024 Evaluation Script

This script uses the BaseEvaluator framework to provide a clean,
maintainable implementation of math problem evaluation.
"""

from typing import List, Dict, Any
from pathlib import Path

import pyrallis
from evaluation_base import BaseEvaluator, EvaluationConfig
from visualization.create_math500_html import create_math500_html_visualization
from verl.utils.autorater_client import call_autorater_service
from data import load_math500_dataset
from helpers import extract_solution_from_response, create_prompts_dataproto


class MathEvaluator(BaseEvaluator):
    """Math problem evaluation implementation using the base framework."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.total_correct_solutions = 0
    
    def _load_dataset(self) -> List[Dict]:
        """Load Math500 or AIME 2024 dataset."""
        return load_math500_dataset(self.cfg.dataset)
        
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """Create Math500 prompts."""
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            # Create prompts
            prompts = []
            for example in batch_examples:
                prompts.append(example['problem'])
            
            # Create DataProto for this batch
            prompts_dataproto = create_prompts_dataproto(
                tokenizer=self.tokenizer,
                questions=prompts,
                max_prompt_length=self.rollout_config["rollout"]["prompt_length"],
                template_type=self.rollout_config["rollout"]["template_type"],
                enable_thinking=self.rollout_config["rollout"]["enable_thinking"]
            )
            
            yield batch_examples, prompts_dataproto
    
    def _extract_content_from_response(self, response_text: str) -> str:
        """Extract solution from response text."""
        return extract_solution_from_response(response_text, self.rollout_config["rollout"]["template_type"], self.rollout_config["rollout"]["enable_thinking"])
    
    def _evaluate_batch(self, examples: List[Dict], extracted_contents: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of solutions against ground truth using autorater."""
        batch_evaluations = []
        
        # Prepare autorater payload for batch evaluation
        prompts = []
        responses = []
        gt_answers = []
        
        for example, extracted_content in zip(examples, extracted_contents):
            prompts.append(example['problem'])
            responses.append(extracted_content)
            gt_answers.append(example['answer'])
        
        # Call autorater service once for the entire batch
        autorater_payload = {
            "prompts": prompts,
            "responses": responses,
            "gt_answers": gt_answers,
            "template_types": ["autorater"] * len(prompts),  # Use standard autorater template for math problems
        }

        print(f"  Evaluating {len(prompts)} solutions with autorater...")
        
        # Call autorater service
        autorater_decisions, autorater_explanations, autorater_raw_responses = call_autorater_service(
            self.cfg.autorater_service_url, autorater_payload, batch_size=len(prompts)
        )
        
        # Process results for each example
        for i, (example, extracted_content) in enumerate(zip(examples, extracted_contents)):
            results = {
                'problem': example['problem'],
                'generated_solution': extracted_content,
                'ground_truth_answer': example['answer'],
                'is_correct': False,
                'confidence': 0.0,
                'explanation': '',
                'evaluation_error': None
            }
            
            if not extracted_content.strip():
                results['evaluation_error'] = 'No solution generated'
                batch_evaluations.append(results)
                continue
            
            # Parse the response to get the correctness decision
            if autorater_decisions and i < len(autorater_decisions):
                decision = autorater_decisions[i]
                
                results['is_correct'] = float(decision) == 1.0
                
                # Get explanation if available
                if autorater_explanations and i < len(autorater_explanations):
                    results['explanation'] = autorater_explanations[i]
            else:
                results['evaluation_error'] = 'No decision from autorater'
            
            batch_evaluations.append(results)
       
        return batch_evaluations
    
    def _create_base_result(self, example: Dict, response_text: str, extracted_content: str,
                           evaluation: Dict, interleaved_components: List[Dict],
                           task_completed: bool, num_tokens: int, ttft_ratio: float,
                           total_tokens_generated: int = None, tokens_to_first_answer: int = None) -> Dict:
        """Create the base result structure for math problems.
        If evaluation is not yet available, return the base result unchanged.
        """
        base_result = super()._create_base_result(
            example, response_text, extracted_content, evaluation,
            interleaved_components, task_completed, num_tokens, ttft_ratio,
            total_tokens_generated, tokens_to_first_answer
        )
        
        # Only enrich the evaluation structure when batch evaluation has populated it
        if isinstance(evaluation, dict) and 'is_correct' in evaluation:
            base_result['evaluation'] = {
                'tests_passed': 1 if evaluation['is_correct'] else 0,
                'tests_failed': 0 if evaluation['is_correct'] else 1,
                'total_tests': 1,
                'test_results': [{
                    'test': f"Solution correctness: {evaluation['is_correct']}",
                    'passed': evaluation['is_correct'],
                    'error': None if evaluation['is_correct'] else evaluation.get('explanation')
                }],
                'execution_error': evaluation.get('evaluation_error'),
                'test_imports': []
            }
        
        return base_result
    
    def _add_task_specific_fields(self, example: Dict, evaluation: Dict) -> Dict[str, Any]:
        """Add math-specific fields to the result."""
        # Track correct solutions
        if evaluation['is_correct']:
            self.total_correct_solutions += 1
        
        return {
            'test_list': [f"Solution should match: {example['answer']}"],
            'entry_point': 'solution',
            'ground_truth_answer': example['answer'],
            # Math500 specific
            'problem': example['problem'],
            'answer': example['answer'],
            'level': example.get('level', None),  # May be None for AIME 2024
            'solution_correct': evaluation['is_correct'],
            'autorater_explanation': evaluation['explanation']
        }
    
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for filename generation."""
        return self.cfg.dataset
    
    def _generate_html_visualization(self, output_dir: Path):
        """Generate HTML visualization for math problems."""
        # Get the template directory from base class
        template_dir = super()._generate_html_visualization(output_dir)
        
        # Generate filename: rollout_name_response_length.html
        filename = self.rollout_config['rollout']['name']
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{self.rollout_config['rollout']['n_candidates']}"
        
        filename += f"_{self.cfg.response_length}"
        
        html_file = template_dir / f"{filename}.html"
        create_math500_html_visualization(self.all_results, html_file)
        print(f"🎨 HTML visualization saved to {html_file}")
    
    def _print_task_specific_metrics(self):
        """Print math-specific metrics."""
        solution_accuracy = (self.total_correct_solutions / self.total_problems * 100) if self.total_problems > 0 else 0
        print(f"Solution Accuracy: {solution_accuracy:.1f}% ({self.total_correct_solutions}/{self.total_problems})")


def main():
    """Main Math500 evaluation function."""
    print("=== Math500 Problem Solving Evaluation ===\n")
    
    # Parse configuration using pyrallis
    cfg = pyrallis.parse(config_class=EvaluationConfig)
    
    # Create and run evaluator
    evaluator = MathEvaluator(cfg)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    main() 