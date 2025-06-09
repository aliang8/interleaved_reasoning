import re
from math_verify import LatexExtractionConfig, parse, verify

class ConditionalRewardTracker:
    def __init__(self, epsilon=0.05):
        self.previous_batch_accuracy = 0.0
        self.current_batch_accuracy = 0.0
        self.epsilon = epsilon
        self.batch_count = 0
    
    def update_batch_accuracy(self, accuracy):
        self.previous_batch_accuracy = self.current_batch_accuracy
        self.current_batch_accuracy = accuracy
        self.batch_count += 1
    
    def should_apply_intermediate_rewards(self):
        if self.batch_count <= 1:
            return True
        return self.current_batch_accuracy > (self.previous_batch_accuracy - self.epsilon)

reward_tracker = ConditionalRewardTracker()

def extract_thinking_and_answer(text):
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    
    thinking = think_match.group(1).strip() if think_match else ""
    final_answer = answer_match.group(1).strip() if answer_match else ""
    
    return thinking, final_answer

def extract_intermediate_answers(thinking_text):
    step_patterns = [
        r'Step \d+:?(.*?)(?=Step \d+|$)',
        r'\d+\.(.*?)(?=\d+\.|$)',
        r'First,?(.*?)(?=Second|Next|Then|Finally|$)',
        r'Second,?(.*?)(?=Third|Next|Then|Finally|$)',
        r'Then,?(.*?)(?=Next|Finally|$)',
        r'Finally,?(.*?)$'
    ]
    
    answers = []
    for pattern in step_patterns:
        matches = re.findall(pattern, thinking_text, re.DOTALL | re.IGNORECASE)
        if matches:
            answers.extend([answer.strip() for answer in matches if answer.strip()])
            break
    
    return answers if answers else [thinking_text.strip()]

def format_check_reward(generated_text):
    has_think = '<think>' in generated_text and '</think>' in generated_text
    has_answer = '<answer>' in generated_text and '</answer>' in generated_text
    return 1.0 if (has_think and has_answer) else 0.0

def final_answer_reward(generated_text, ground_truth, lambda_a=1.0):
    """
    Final answer reward function as per specification:
    rfinal(x, y) = λa · {
        2.0 if y^(N)_answer = g_N
        -1.5 if y^(N)_answer ≠ g_N  
        -2.0 if answer is not parseable
    }
    """
    try:
        thinking, final_answer = extract_thinking_and_answer(generated_text)
        
        if not final_answer or final_answer.strip() == "":
            return lambda_a * (-2.0)
        
        # Check if final answer matches ground truth (exact match)
        if final_answer.lower().strip() == ground_truth.lower().strip():
            return lambda_a * 2.0
        else:
            return lambda_a * (-1.5)
            
    except Exception:
        return lambda_a * (-2.0)

def check_answer_correctness(answer_text, ground_truth_answer):
    answer_lower = answer_text.lower().strip()
    gt_lower = ground_truth_answer.lower().strip()
    return gt_lower in answer_lower

def conditional_intermediate_reward_all_or_none(generated_text, ground_truth, intermediate_truths, base_reward=1.0):
    thinking, final_answer = extract_thinking_and_answer(generated_text)
    
    intermediate_answers = extract_intermediate_answers(thinking)
    if not intermediate_answers or not intermediate_truths:
        return 0.0
    
    all_correct = True
    for k in range(len(intermediate_truths)):
        found_correct = False
        for answer in intermediate_answers:
            if check_answer_correctness(answer, intermediate_truths[k]):
                found_correct = True
                break
        if not found_correct:
            all_correct = False
            break
    
    return base_reward if all_correct else 0.0

def conditional_intermediate_reward_partial_credit(generated_text, ground_truth, intermediate_truths, base_reward=1.0):
    thinking, final_answer = extract_thinking_and_answer(generated_text)
    
    intermediate_answers = extract_intermediate_answers(thinking)
    if not intermediate_answers or not intermediate_truths:
        return 0.0
    
    N = len(intermediate_truths)
    reward_sum = 0.0
    
    for k in range(N):
        for answer in intermediate_answers:
            if check_answer_correctness(answer, intermediate_truths[k]):
                reward_sum += base_reward / N
                break
    
    return reward_sum

def conditional_intermediate_reward_time_discounted(generated_text, ground_truth, intermediate_truths, base_reward=1.0):
    thinking, final_answer = extract_thinking_and_answer(generated_text)
    
    intermediate_answers = extract_intermediate_answers(thinking)
    if not intermediate_answers or not intermediate_truths:
        return 0.0
    
    correct_step = {}
    
    for step_idx, answer in enumerate(intermediate_answers, 1):
        for gt_idx, gt_answer in enumerate(intermediate_truths):
            if gt_idx not in correct_step and check_answer_correctness(answer, gt_answer):
                correct_step[gt_idx] = step_idx
    
    if len(correct_step) == len(intermediate_truths):
        return base_reward
    else:
        if not correct_step:
            return 0.0
        sum_weights = sum(1.0 / step for step in correct_step.values())
        return (sum_weights / len(intermediate_truths)) * base_reward

def conditional_reward_function(completions, reward_type="partial_credit", base_reward=1.0, lambda_a=1.0, **kwargs):
    generated_texts = [completion[0]["content"] for completion in completions]
    ground_truths = kwargs.get('solution', [])
    intermediate_truths_list = kwargs.get('intermediate_truths', [[] for _ in generated_texts])
    
    batch_size = len(generated_texts)
    rewards = []
    
    # Calculate batch accuracy for progression tracking (based on correct final answers)
    batch_final_accuracy = sum(
        1 for gen, gt in zip(generated_texts, ground_truths)
        if final_answer_reward(gen, gt, lambda_a) == lambda_a * 2.0  # Only count correct answers
    ) / batch_size
    
    reward_tracker.update_batch_accuracy(batch_final_accuracy)
    
    for generated_text, ground_truth, intermediate_truths in zip(generated_texts, ground_truths, intermediate_truths_list):
        r_format = format_check_reward(generated_text)
        r_final = final_answer_reward(generated_text, ground_truth, lambda_a)
        
        # Check the three conditions from Algorithm 1
        is_final_correct = r_final == lambda_a * 2.0  # Only positive (correct) final answers
        is_format_valid = r_format > 0
        is_progressing = reward_tracker.should_apply_intermediate_rewards()
        
        # Only compute intermediate rewards if all conditions are met
        if is_final_correct and is_format_valid and is_progressing:
            if reward_type == "all_or_none":
                r_intermediate = conditional_intermediate_reward_all_or_none(
                    generated_text, ground_truth, intermediate_truths, base_reward
                )
            elif reward_type == "time_discounted":
                r_intermediate = conditional_intermediate_reward_time_discounted(
                    generated_text, ground_truth, intermediate_truths, base_reward
                )
            else:
                r_intermediate = conditional_intermediate_reward_partial_credit(
                    generated_text, ground_truth, intermediate_truths, base_reward
                )
        else:
            r_intermediate = 0.0
        
        total_reward = r_format + r_final + r_intermediate
        rewards.append(total_reward)
    
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards