#!/usr/bin/env python3
"""
Auto-rater test script using a small LLM to evaluate predicted responses against ground truth.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
from typing import List, Dict, Tuple

# Auto-rater template
AUTO_RATER_TEMPLATE = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground truth
answer. Your task is to determine if the ground truth answer is present in the LLM's response.
Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers – look for equivalent information or correct answers. Do
not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the "Ground
Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {predicted_answer}
- Ground Truth Answer: {ground_truth_answer}
===Output Format===
Provide your final evaluation in the following format:
"Decision:" ("TRUE" or "FALSE")

Please proceed with the evaluation.
Decision: """

class AutoRater:
    def __init__(self, model_name: str = "gpt2-medium"):
        """
        Initialize the auto-rater with a better LLM.
        
        Args:
            model_name: Name of the model to use (default: GPT-2 medium for better performance)
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create text generation pipeline with better parameters
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=10,
            do_sample=False,
            temperature=0.0,  # Lower temperature for more focused responses
            top_p=1.0,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print(f"✅ Model loaded successfully!")
    
    def format_prompt(self, question: str, predicted_answer: str, ground_truth_answer: str) -> str:
        """Format the auto-rater prompt with the given inputs."""
        return AUTO_RATER_TEMPLATE.format(
            question=question,
            predicted_answer=predicted_answer,
            ground_truth_answer=ground_truth_answer
        )
    
    def parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the model's response to extract explanation and decision.
        
        Returns:
            Tuple of (explanation, decision)
        """
        # More flexible parsing patterns
        explanation_patterns = [
            r'Explanation:\s*(.+?)(?=Decision:|$)',
            r'(?:^|\n)(.+?)(?=Decision:|$)',
        ]
        
        decision_patterns = [
            r'Decision:\s*["\']?(TRUE|FALSE)["\']?',
            r'\b(TRUE|FALSE)\b',
            r'(true|false)',
            r'answer is\s+(TRUE|FALSE)',
            r'decision is\s+(TRUE|FALSE)',
        ]
        
        explanation = "No explanation found"
        decision = "UNKNOWN"
        
        # Try to find explanation
        for pattern in explanation_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                explanation = match.group(1).strip()
                break
        
        # Try to find decision
        for pattern in decision_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                decision = match.group(1).upper()
                break
        
        return explanation, decision
    
    def evaluate(self, question: str, predicted_answer: str, ground_truth_answer: str) -> Dict[str, str]:
        """
        Evaluate a predicted answer against ground truth using the LLM.
        
        Returns:
            Dictionary with evaluation results
        """
        prompt = self.format_prompt(question, predicted_answer, ground_truth_answer)
        
        try:
            # Generate response with more tokens
            response = self.generator(
                prompt,
                max_new_tokens=10,
                num_return_sequences=1,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.1
            )[0]['generated_text']
            
            # Extract only the new generated part (after the prompt)
            generated_part = response[len(prompt):].strip()
            
            # Parse the response
            explanation, decision = self.parse_response(generated_part)
            
            return {
                'question': question,
                'predicted_answer': predicted_answer,
                'ground_truth_answer': ground_truth_answer,
                'explanation': explanation,
                'decision': decision,
                'raw_response': generated_part,
                'full_response': response
            }
            
        except Exception as e:
            return {
                'question': question,
                'predicted_answer': predicted_answer,
                'ground_truth_answer': ground_truth_answer,
                'explanation': f"Error during evaluation: {str(e)}",
                'decision': "ERROR",
                'raw_response': "",
                'full_response': ""
            }

def create_test_cases() -> List[Dict[str, str]]:
    """Create a set of test cases with obvious TRUE and FALSE examples."""
    return [
        # Clear TRUE cases
        {
            'question': "What is the capital of France?",
            'predicted_answer': "The capital of France is Paris.",
            'ground_truth_answer': "Paris",
            'expected': "TRUE"
        },
        {
            'question': "What is 2 + 2?",
            'predicted_answer': "Two plus two equals four. The answer is 4.",
            'ground_truth_answer': "4",
            'expected': "TRUE"
        },
        {
            'question': "Who wrote Romeo and Juliet?",
            'predicted_answer': "Romeo and Juliet was written by William Shakespeare.",
            'ground_truth_answer': "William Shakespeare",
            'expected': "TRUE"
        },
        
        # Clear FALSE cases
        {
            'question': "What is the capital of France?",
            'predicted_answer': "The capital of France is London.",
            'ground_truth_answer': "Paris",
            'expected': "FALSE"
        },
        {
            'question': "What is 2 + 2?",
            'predicted_answer': "Two plus two equals five.",
            'ground_truth_answer': "4",
            'expected': "FALSE"
        },
        {
            'question': "Who wrote Romeo and Juliet?",
            'predicted_answer': "Romeo and Juliet was written by Charles Dickens.",
            'ground_truth_answer': "William Shakespeare",
            'expected': "FALSE"
        },
        
        # Partial/Ambiguous cases
        {
            'question': "What are the primary colors?",
            'predicted_answer': "The primary colors are red and blue.",
            'ground_truth_answer': "Red, blue, and yellow",
            'expected': "FALSE"  # Missing yellow
        },
        {
            'question': "What is the largest planet in our solar system?",
            'predicted_answer': "Jupiter is the biggest planet in the solar system by both mass and volume.",
            'ground_truth_answer': "Jupiter",
            'expected': "TRUE"  # Same meaning, different wording
        }
    ]

def main():
    """Main function to run the auto-rater test."""
    print("🤖 Auto-Rater Test Script")
    print("=" * 50)
    
    # Initialize the auto-rater with better models
    model_options = [
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    
    rater = None
    for model_name in model_options:
        try:
            print(f"🔄 Trying model: {model_name}")
            rater = AutoRater(model_name)
            break
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue
    
    if rater is None:
        print("❌ Failed to load any model!")
        return
    
    # Get test cases
    test_cases = create_test_cases()
    
    print(f"\n📋 Running auto-rater on {len(test_cases)} test cases...")
    print("=" * 50)
    
    # Evaluate each test case
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}/{len(test_cases)}")
        print(f"Question: {test_case['question']}")
        print(f"Predicted: {test_case['predicted_answer']}")
        print(f"Ground Truth: {test_case['ground_truth_answer']}")
        print(f"Expected Decision: {test_case['expected']}")
        
        # Run evaluation
        result = rater.evaluate(
            test_case['question'],
            test_case['predicted_answer'],
            test_case['ground_truth_answer']
        )
        
        print(f"\n📝 Raw LLM Response:")
        print("─" * 40)
        print(result['raw_response'])
        print("─" * 40)
        
        print(f"\n🤖 LLM Decision: {result['decision']}")
        print(f"📋 Parsed Explanation: {result['explanation']}")
        
        # Check if decision matches expected
        is_correct = result['decision'] == test_case['expected']
        correctness = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"📊 Evaluation: {correctness}")
        
        results.append({
            **result,
            'expected': test_case['expected'],
            'is_correct': is_correct
        })
        
        print("-" * 70)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"Model used: {rater.model_name}")
    print(f"Total test cases: {total_count}")
    print(f"Correct evaluations: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    
    print(f"\n🎯 Decision Breakdown:")
    true_decisions = sum(1 for r in results if r['decision'] == 'TRUE')
    false_decisions = sum(1 for r in results if r['decision'] == 'FALSE')
    unknown_decisions = sum(1 for r in results if r['decision'] == 'UNKNOWN')
    error_decisions = sum(1 for r in results if r['decision'] == 'ERROR')
    
    print(f"  TRUE decisions: {true_decisions}")
    print(f"  FALSE decisions: {false_decisions}")
    print(f"  UNKNOWN decisions: {unknown_decisions}")
    print(f"  ERROR decisions: {error_decisions}")
    
    # Show incorrect cases
    incorrect_cases = [r for r in results if not r['is_correct']]
    if incorrect_cases:
        print(f"\n❌ Incorrect Evaluations:")
        for case in incorrect_cases:
            print(f"  Q: {case['question']}")
            print(f"  Expected: {case['expected']}, Got: {case['decision']}")
            print(f"  Raw response preview: {case['raw_response'][:100]}...")
    
    # Show unknown cases for debugging
    unknown_cases = [r for r in results if r['decision'] == 'UNKNOWN']
    if unknown_cases:
        print(f"\n❓ UNKNOWN Decisions (for debugging):")
        for case in unknown_cases:
            print(f"  Q: {case['question']}")
            print(f"  Raw response: {case['raw_response']}")
            print()
    
    print(f"\n✅ Auto-rater testing complete!")

if __name__ == "__main__":
    main() 