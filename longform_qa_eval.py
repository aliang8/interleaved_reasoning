#!/usr/bin/env python3
"""
Standardized Long-Form QA Evaluation Script

This script uses the BaseEvaluator framework to provide a clean,
maintainable implementation of long-form question answering evaluation.
"""

import json
import re
from typing import List, Dict, Any
from pathlib import Path

import pyrallis
from evaluation_base import BaseEvaluator, EvaluationConfig
from helpers import create_prompts_dataproto, extract_solution_from_response


class LongFormQAEvaluator(BaseEvaluator):
    """Long-form QA evaluation implementation using the base framework."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.total_correct_answers = 0
    
    def _load_dataset(self) -> List[Dict]:
        """Load QuALITY dataset from JSONL file."""
        print("Loading QuALITY dataset...")
        
        # Load the dataset from the JSONL file
        data_path = "data/QuALITY/QuALITY.v1.0.1.htmlstripped.train"
        
        try:
            with open(data_path, "r") as f:
                data = f.readlines()
            
            # Parse JSON lines
            data = [json.loads(line) for line in data]
            print(f"✓ Loaded {len(data)} articles from QuALITY dataset")
            
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {data_path}")
            print("Please ensure the QuALITY dataset is available at the specified path")
            return []
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
        
        # Convert to the format expected by the evaluator
        examples = []
        example_id = 0
        
        for article in data:
            article_text = article["article"]
            questions = article["questions"]
            
            for question_data in questions:
                # Extract question information
                question_text = question_data["question"]
                options = question_data["options"]
                gold_label = question_data["gold_label"]  # This should be 1-indexed
                
                # Create example entry
                example = {
                    "id": example_id,
                    "context": article_text,
                    "question": question_text,
                    "options": options,
                    "gold_label": gold_label,
                    "question_id": f"{article['article_id']}_{question_data.get('question_id', 'q')}",
                    "article_id": article["article_id"],
                    "title": article["title"],
                    "author": article["author"],
                    "topic": article["topic"]
                }
                
                examples.append(example)
                example_id += 1
                
                # Debug output for first few examples
                if example_id <= 3:
                    print(f"  Example {example_id}: {question_text[:50]}...")
                    print(f"    Options: {len(options)} choices")
                    print(f"    Gold label: {gold_label}")
        
        print(f"✓ Converted {len(examples)} questions from {len(data)} articles")
        return examples
    
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """Create long-form QA prompts with context and multiple choice options."""
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            # Create prompts
            prompts = []
            for example in batch_examples:
                # Format the prompt as specified
                prompt = f"""Context: {example['context']}

Question: {example['question']}

Choices: \nA. {example['options'][0]}\nB. {example['options'][1]}\nC. {example['options'][2]}\nD. {example['options'][3]}

Please select the correct answer from the choices above. Output only the letter of the correct answer."""
                
                prompts.append(prompt)
            
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
        """Extract the selected answer from response text."""
        return extract_solution_from_response(response_text, self.rollout_config["rollout"]["template_type"], self.rollout_config["rollout"]["enable_thinking"])
    
    def _evaluate_batch(self, examples: List[Dict], extracted_contents: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of answers against ground truth."""
        batch_evaluations = []
        
        for example, extracted_content in zip(examples, extracted_contents):
            results = {
                'question': example['question'],
                'context': example['context'],
                'options': example['options'],
                'gold_label': example['gold_label'],
                'selected_answer': extracted_content,
                'is_correct': False,
                'confidence': 0.0,
                'explanation': '',
                'evaluation_error': None
            }
            
            if not extracted_content.strip():
                results['evaluation_error'] = 'No answer generated'
                batch_evaluations.append(results)
                continue
            
            # Extract the selected option from the response
            selected_option = self._extract_selected_option(extracted_content)
            
            if selected_option is None:
                results['evaluation_error'] = 'Could not determine selected option from response'
                batch_evaluations.append(results)
                continue
            
            # Check if the selected option matches the gold label
            # Convert gold_label (1-indexed) to 0-indexed for array access
            gold_option_index = example['gold_label'] - 1
            gold_answer = example['options'][gold_option_index]
            
            results['selected_option'] = selected_option
            results['gold_answer'] = gold_answer
            results['is_correct'] = (selected_option == example['gold_label'])
            
            # Add explanation based on correctness
            if results['is_correct']:
                results['explanation'] = f"Correct! Selected option {selected_option} matches the gold label {example['gold_label']}."
            else:
                results['explanation'] = f"Incorrect. Selected option {selected_option}, but gold label is {example['gold_label']} ({gold_answer})."
            
            batch_evaluations.append(results)
        
        return batch_evaluations
    
    def _extract_selected_option(self, response_text: str) -> int:
        """Extract the selected option (A, B, C, or D) from the response."""
        # Look for explicit option selection patterns
        response_lower = response_text.lower().strip()
        
        # Pattern 1: "The answer is A" or "Answer: B"
        answer_patterns = [
            r"the answer is\s*([abcd])",
            r"answer:\s*([abcd])",
            r"option\s*([abcd])",
            r"choice\s*([abcd])",
            r"select\s*([abcd])",
            r"([abcd])\s*is correct",
            r"correct answer is\s*([abcd])"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response_lower)
            if match:
                option = match.group(1).upper()
                return self._option_to_number(option)
        
        # Pattern 2: Look for the option letter in context
        # Check if the response contains one of the option letters prominently
        option_counts = {}
        for option in ['A', 'B', 'C', 'D']:
            # Count occurrences of the option letter
            count = response_lower.count(option.lower())
            option_counts[option] = count
        
        # If one option is mentioned significantly more than others, use it
        max_count = max(option_counts.values())
        if max_count > 0:
            # Find the option with the highest count
            for option, count in option_counts.items():
                if count == max_count:
                    return self._option_to_number(option)
        
        # Pattern 3: Look for the actual answer text in the response
        # This is more complex and would require matching the option text
        # For now, return None if we can't determine
        
        return None
    
    def _option_to_number(self, option: str) -> int:
        """Convert option letter to number (A=1, B=2, C=3, D=4)."""
        option_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        return option_map.get(option.upper(), None)
    
    def _create_base_result(self, example: Dict, response_text: str, extracted_content: str,
                           evaluation: Dict, interleaved_components: List[Dict],
                           task_completed: bool, num_tokens: int, ttft_ratio: float,
                           total_tokens_generated: int = None, tokens_to_first_answer: int = None) -> Dict:
        """Create the base result structure for long-form QA problems."""
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
                    'test': f"Answer correctness: {evaluation['is_correct']}",
                    'passed': evaluation['is_correct'],
                    'error': None if evaluation['is_correct'] else evaluation.get('explanation')
                }],
                'execution_error': evaluation.get('evaluation_error'),
                'test_imports': []
            }
        
        return base_result
    
    def _add_task_specific_fields(self, example: Dict, evaluation: Dict) -> Dict[str, Any]:
        """Add long-form QA-specific fields to the result."""
        # Track correct answers
        if evaluation['is_correct']:
            self.total_correct_answers += 1
        
        return {
            'test_list': [f"Answer should match: {evaluation.get('gold_answer', 'Unknown')}"],
            'entry_point': 'answer_selection',
            'context': example['context'],
            'question': example['question'],
            'options': example['options'],
            'gold_label': example['gold_label'],
            'selected_option': evaluation.get('selected_option'),
            'answer_correct': evaluation['is_correct'],
            'explanation': evaluation.get('explanation')
        }
    
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for filename generation."""
        return "longform_qa"
    
    def _generate_html_visualization(self, output_dir: Path):
        """Generate HTML visualization for long-form QA problems."""
        thinking_tag = "thinking" if self.rollout_config["rollout"]["enable_thinking"] else "no_thinking"
        filename = f"{self.cfg.dataset}_{self.cfg.template_type}_{self.cfg.rollout_name}_{thinking_tag}_{self.cfg.response_length}"
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{self.rollout_config['rollout']['n_candidates']}"
        
        html_file = output_dir / f"{filename}.html"
        
        # Import and use the HTML visualization function
        try:
            from create_longform_qa_html import create_longform_qa_html_visualization
            create_longform_qa_html_visualization(self.all_results, html_file)
            print(f"🎨 HTML visualization saved to {html_file}")
        except ImportError:
            print(f"🎨 HTML visualization would be saved to {html_file}")
            print("   (HTML visualization module not found)")
    
    def _print_task_specific_metrics(self):
        """Print long-form QA-specific metrics."""
        answer_accuracy = (self.total_correct_answers / self.total_problems * 100) if self.total_problems > 0 else 0
        print(f"Answer Accuracy: {answer_accuracy:.1f}% ({self.total_correct_answers}/{self.total_problems})")


def main():
    """Main long-form QA evaluation function."""
    print("=== Long-Form QA Evaluation ===\n")
    
    # Parse configuration using pyrallis
    cfg = pyrallis.parse(config_class=EvaluationConfig)
    
    # Create and run evaluator
    evaluator = LongFormQAEvaluator(cfg)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    main() 