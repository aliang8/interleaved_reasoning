import sys
sys.path.append('.')

from rewards import conditional_reward_function, reward_tracker
import pytest

def test_conditional_reward_function():
    """Test conditional reward function with math expansion example."""
    
    # Reset reward tracker for clean test
    reward_tracker.batch_count = 0
    reward_tracker.previous_batch_accuracy = 0.0
    reward_tracker.current_batch_accuracy = 0.0
    
    # Example completion with interleaved reasoning
    generated_text = """<think>First, let's expand the product step by step. We start with the expression (x−2)²(x+2)². We can simplify this by recognizing that (x − 2)(x + 2) = x² − 4. So, the expression becomes (x² − 4)².</think>
<answer>(x² − 4)²</answer>
<think>Next, we need to expand (x² − 4)². This is a square of a binomial, which can be expanded using the formula (a − b)² = a² − 2ab + b².</think>
<answer>(x² − 4)² = x⁴ − 8x² + 16</answer>
<think>The expanded form of the expression is x⁴ − 8x² + 16. The nonzero coefficients are 1, -8, and 16. We need to find the product of these coefficients.</think>
<answer>The product of the coefficients 1, -8, and 16 is 1 × −8 × 16 = −128.</answer>
<think>The final answer is:</think>
<answer>-128</answer>"""

    # Format as completion (matches expected format from GRPO)
    completions = [
        [{"content": generated_text}]
    ]
    
    # Ground truth and intermediate truths
    ground_truth = ["-128"]
    intermediate_truths = [
        ["(x² − 4)²", "x squared minus 4 squared"],  # First intermediate step
        ["x⁴ − 8x² + 16", "x to the fourth minus 8x squared plus 16"],  # Second intermediate step  
        ["1 × −8 × 16 = −128", "product of coefficients is -128"]  # Third intermediate step
    ]
    
    # Test case 1: All-or-None strategy
    print("Testing All-or-None Strategy:")
    reward_all_or_none = conditional_reward_function(
        completions=completions,
        reward_type="all_or_none",
        base_reward=1.0,
        solution=ground_truth,
        intermediate_truths=intermediate_truths
    )
    print(f"All-or-None Reward: {reward_all_or_none[0]}")
    
    # Test case 2: Partial Credit strategy  
    print("\nTesting Partial Credit Strategy:")
    reward_partial = conditional_reward_function(
        completions=completions,
        reward_type="partial_credit", 
        base_reward=1.0,
        solution=ground_truth,
        intermediate_truths=intermediate_truths
    )
    print(f"Partial Credit Reward: {reward_partial[0]}")
    
    # Test case 3: Time-Discounted strategy
    print("\nTesting Time-Discounted Strategy:")
    reward_time_discounted = conditional_reward_function(
        completions=completions,
        reward_type="time_discounted",
        base_reward=1.0, 
        solution=ground_truth,
        intermediate_truths=intermediate_truths
    )
    print(f"Time-Discounted Reward: {reward_time_discounted[0]}")


def test_conditional_reward_incorrect_final_answer():
    """Test case where final answer is incorrect - should return 0 rewards."""
    
    # Reset reward tracker
    reward_tracker.batch_count = 0
    reward_tracker.previous_batch_accuracy = 0.0
    reward_tracker.current_batch_accuracy = 0.0
    
    # Same reasoning but wrong final answer
    generated_text_wrong = """<think>First, let's expand the product step by step. We start with the expression (x−2)²(x+2)². We can simplify this by recognizing that (x − 2)(x + 2) = x² − 4. So, the expression becomes (x² − 4)².</think>
<answer>(x² − 4)²</answer>
<think>Next, we need to expand (x² − 4)². This is a square of a binomial, which can be expanded using the formula (a − b)² = a² − 2ab + b².</think>
<answer>(x² − 4)² = x⁴ − 8x² + 16</answer>
<think>The expanded form of the expression is x⁴ − 8x² + 16. The nonzero coefficients are 1, -8, and 16. We need to find the product of these coefficients.</think>
<answer>The product of the coefficients 1, -8, and 16 is 1 × −8 × 16 = −128.</answer>
<think>The final answer is:</think>
<answer>128</answer>"""  # Wrong final answer!

    completions = [[{"content": generated_text_wrong}]]
    ground_truth = ["-128"] 
    intermediate_truths = [
        ["(x² − 4)²"],
        ["x⁴ − 8x² + 16"], 
        ["1 × −8 × 16 = −128"]
    ]
    
    print("\nTesting Incorrect Final Answer (should be 0 rewards):")
    
    for strategy in ["all_or_none", "partial_credit", "time_discounted"]:
        reward = conditional_reward_function(
            completions=completions,
            reward_type=strategy,
            base_reward=1.0,
            solution=ground_truth,
            intermediate_truths=intermediate_truths
        )
        print(f"{strategy.capitalize()} reward: {reward[0]}")
        assert reward[0] == -0.5, f"Expected format(1.0) + final(-1.5) + intermediate(0.0) = -0.5 for {strategy}, got {reward[0]}"


def test_conditional_reward_missing_intermediate_steps():
    """Test case where some intermediate steps are missing."""
    
    # Reset reward tracker
    reward_tracker.batch_count = 0
    reward_tracker.previous_batch_accuracy = 0.0
    reward_tracker.current_batch_accuracy = 0.0
    
    # Missing some intermediate steps
    generated_text_partial = """<think>I need to expand (x−2)²(x+2)².</think>
<answer>(x² − 4)²</answer>
<think>Final calculation gives us the answer.</think>
<answer>-128</answer>"""

    completions = [[{"content": generated_text_partial}]]
    ground_truth = ["-128"]
    intermediate_truths = [
        ["(x² − 4)²"],
        ["x⁴ − 8x² + 16"],  # This step is missing
        ["1 × −8 × 16 = −128"]  # This step is missing
    ]
    
    print("\nTesting Missing Intermediate Steps:")
    
    # All-or-None should give 0 intermediate reward (not all steps present)
    reward_all_or_none = conditional_reward_function(
        completions=completions,
        reward_type="all_or_none",
        base_reward=1.0,
        solution=ground_truth, 
        intermediate_truths=intermediate_truths
    )
    print(f"All-or-None (missing steps): {reward_all_or_none[0]}")
    
    # Partial Credit should give some reward (1/3 of intermediate steps correct)
    reward_partial = conditional_reward_function(
        completions=completions,
        reward_type="partial_credit",
        base_reward=1.0,
        solution=ground_truth,
        intermediate_truths=intermediate_truths
    )
    print(f"Partial Credit (missing steps): {reward_partial[0]}")
    
    # Time-Discounted should give partial reward
    reward_time_discounted = conditional_reward_function(
        completions=completions,
        reward_type="time_discounted", 
        base_reward=1.0,
        solution=ground_truth,
        intermediate_truths=intermediate_truths
    )
    print(f"Time-Discounted (missing steps): {reward_time_discounted[0]}")


def test_batch_accuracy_progression():
    """Test that rewards are only given when batch accuracy is progressing."""
    
    # Reset and simulate declining batch accuracy
    reward_tracker.batch_count = 2
    reward_tracker.previous_batch_accuracy = 0.8
    reward_tracker.current_batch_accuracy = 0.6  # Declining accuracy
    
    generated_text = """<think>Testing with declining accuracy.</think>
<answer>-128</answer>"""

    completions = [[{"content": generated_text}]]
    ground_truth = ["-128"]
    intermediate_truths = [["some step"]]
    
    print("\nTesting Declining Batch Accuracy (should give limited rewards):")
    
    reward = conditional_reward_function(
        completions=completions,
        reward_type="partial_credit",
        base_reward=1.0,
        solution=ground_truth,
        intermediate_truths=intermediate_truths
    )
    print(f"Reward with declining accuracy: {reward[0]}")
    

if __name__ == "__main__":
    print("Running Conditional Reward Function Tests\n")
    print("=" * 60)
    
    test_conditional_reward_function()
    print("\n" + "=" * 60)
    
    test_conditional_reward_incorrect_final_answer()
    print("\n" + "=" * 60)
    
    test_conditional_reward_missing_intermediate_steps()
    print("\n" + "=" * 60)
    
    test_batch_accuracy_progression()
    print("\n" + "=" * 60)
    
    print("\nAll tests completed!")
