from kk_prompts import system_instruction_no_reason, system_instruction

def formatting_func_kk(example, mode="func"):
    """Formatting function specifically for K&K dataset using existing prompt functions."""
    if mode == "func":
        # Use the existing formatting_prompts_func for direct mode
        return formatting_prompts_func(example, eos_token="")

    elif mode == "cot":
        # Use the existing formatting_prompts_func_cot for chain of thought mode
        return formatting_prompts_func_cot(example, eos_token="")


def formatting_func_musique(example, mode="func"):
    """Formatting function specifically for MuSiQue dataset."""
    question = example["question"]
    answer = example["answer"]
    paragraphs = example.get("paragraphs", None)
    
    if mode == "func":
        # Direct function mode
        if paragraphs:
            context = " ".join([p.get("text", "") for p in paragraphs])
            text = (
                f"### Question: {question}\n"
                f"### Answer: <think>Based on the given information: {context} "
                f"I need to analyze this to find the answer.</think>\n"
                f"<answer>{answer}</answer>"
            )
        else:
            text = (
                f"### Question: {question}\n"
                f"### Answer: <think>Let me think about this question step by step.</think>\n"
                f"<answer>{answer}</answer>"
            )
    elif mode == "cot":
        # Chain of thought mode with more detailed reasoning
        if paragraphs:
            context = " ".join([p.get("text", "") for p in paragraphs])
            text = (
                f"### Question: {question}\n"
                f"### Answer: <think>Let me break this down step by step. "
                f"Given the information: {context} "
                f"I need to carefully analyze each piece of evidence to reach the correct conclusion.</think>\n"
                f"<answer>{answer}</answer>"
            )
        else:
            text = (
                f"### Question: {question}\n"
                f"### Answer: <think>Let me approach this systematically, step by step, "
                f"considering all aspects of the question to arrive at the correct answer.</think>\n"
                f"<answer>{answer}</answer>"
            )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'func' or 'cot'")
    
    return text


def formatting_func_general(example, mode="func"):
    """General formatting function that dispatches based on dataset_source."""
    dataset_source = example["dataset_source"]
    
    if dataset_source == "kk":
        return formatting_func_kk(example, mode)
    elif dataset_source == "musique":
        return formatting_func_musique(example, mode)
    else:
        # Default formatting for unknown datasets
        question_field = "question" if "question" in example else "quiz"
        answer_field = "answer" if "answer" in example else "solution"
        
        question = example[question_field]
        answer = example[answer_field]
        
        if mode == "func":
            text = (
                f"### Question: {question}\n"
                f"### Answer: <think>Let me think about this step by step.</think>\n"
                f"<answer>{answer}</answer>"
            )
        elif mode == "cot":
            text = (
                f"### Question: {question}\n"
                f"### Answer: <think>Let me carefully analyze this question step by step, "
                f"considering all relevant factors to arrive at the correct answer.</think>\n"
                f"<answer>{answer}</answer>"
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'func' or 'cot'")
        
        return text

# Formatting function for Knights-and-Knaves
def formatting_prompts_func(example, eos_token):
    text = (
        system_instruction_no_reason
        + f"\n\n### Question: {example['quiz']}\n### Answer:\nCONCLUSION:\n{example['solution_text_format']}"
    )
    text += eos_token
    return text


def formatting_prompts_func_cot(example, eos_token):
    cot_head = "Let's think step by step, by considering whether each person is lying and if that leads to contradiction."
    cot_steps = example["cot_repeat_steps"]
    cot_steps = " ".join(cot_steps)
    cot_foot = example["cot_foot"]
    text = (
        system_instruction
        + f"\n\n### Question: {example['quiz']}\n### Answer: {cot_head} {cot_steps} {cot_foot}\nCONCLUSION:\n{example['solution_text_format']}"
    )
    text += eos_token
    return text


def compute_ttft(generated_text, tokenizer, prompt_text=""):
    """
    Compute normalized Time to First Token (TTFT) metric.
    
    TTFT = Position of first answer token / Total response length
    
    Args:
        generated_text (str): The full generated text including prompt and response
        tokenizer: The tokenizer used for tokenization
        prompt_text (str): The original prompt text to exclude from TTFT calculation
        
    Returns:
        float: Normalized TTFT value between 0 and 1, where lower values indicate faster initial responses
    """
    # Remove prompt from generated text to get only the response
    if prompt_text and generated_text.startswith(prompt_text):
        response_text = generated_text[len(prompt_text):].strip()
    else:
        response_text = generated_text.strip()
    
    # Tokenize the response
    response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
    total_response_length = len(response_tokens)
    
    if total_response_length == 0:
        return 1.0  # If no response, TTFT is maximum (worst case)
    
    # Find the position of the first answer token
    # Look for <answer> tag or direct answer content
    answer_tag = "<answer>"
    if answer_tag in response_text:
        # Find position where <answer> tag ends and actual answer begins
        answer_start_idx = response_text.find(answer_tag) + len(answer_tag)
        text_before_answer = response_text[:answer_start_idx]
        tokens_before_answer = tokenizer.encode(text_before_answer, add_special_tokens=False)
        first_answer_token_position = len(tokens_before_answer)
    else:
        # If no explicit answer tag, assume answer starts after thinking section
        think_tag = "</think>"
        if think_tag in response_text:
            think_end_idx = response_text.find(think_tag) + len(think_tag)
            text_before_answer = response_text[:think_end_idx]
            tokens_before_answer = tokenizer.encode(text_before_answer, add_special_tokens=False)
            first_answer_token_position = len(tokens_before_answer)
        else:
            # If no structured format, assume first token is the answer start
            first_answer_token_position = 1
    
    # Ensure position is within bounds
    first_answer_token_position = min(first_answer_token_position, total_response_length)
    
    # Calculate normalized TTFT
    ttft = first_answer_token_position / total_response_length
    
    return ttft


def compute_ttft_from_tokens(input_token_ids, output_token_ids, tokenizer):
    """
    Compute normalized TTFT from token IDs directly.
    
    Args:
        input_token_ids: Input prompt token IDs
        output_token_ids: Full output token IDs (including input)
        tokenizer: The tokenizer used
        
    Returns:
        float: Normalized TTFT value between 0 and 1
    """
    # Extract only the generated tokens (excluding input)
    input_length = len(input_token_ids)
    generated_tokens = output_token_ids[input_length:]
    total_response_length = len(generated_tokens)
    
    if total_response_length == 0:
        return 1.0
    
    # Convert tokens back to text to find answer position
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    
    # Find answer start position in tokens
    answer_tag = "<answer>"
    if answer_tag in generated_text:
        # Find where answer content starts
        answer_start_idx = generated_text.find(answer_tag) + len(answer_tag)
        text_before_answer = generated_text[:answer_start_idx]
        tokens_before_answer = tokenizer.encode(text_before_answer, add_special_tokens=False)
        first_answer_token_position = len(tokens_before_answer)
    else:
        # Fallback logic
        think_tag = "</think>"
        if think_tag in generated_text:
            think_end_idx = generated_text.find(think_tag) + len(think_tag)
            text_before_answer = generated_text[:think_end_idx]
            tokens_before_answer = tokenizer.encode(text_before_answer, add_special_tokens=False)
            first_answer_token_position = len(tokens_before_answer)
        else:
            first_answer_token_position = 1
    
    # Ensure position is within bounds
    first_answer_token_position = min(first_answer_token_position, total_response_length)
    
    # Calculate normalized TTFT
    ttft = first_answer_token_position / total_response_length
    
    return ttft

# taken from https://github.com/AlphaPav/mem-kk-logic/blob/main/dataset/kk.py
def parse_cot_eval(pred_str, ans,
                   conclusion_patterns=['CONCLUSION:'],
                   verbose=False,
                   finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"],
                   reformat_gold_conditions=None):
    
    def judge_string(input_str, reformat_gold_conditions, wrong_reason, finish_patterns):
        correct_count = 0
        is_correct = False
        beyond_id = len(reformat_gold_conditions)+1
        beyond_id_pattern = f"({beyond_id})"

        for finish_pattern in finish_patterns:
            if finish_pattern in input_str:
                input_str = input_str.split(finish_pattern)[0]

        if beyond_id_pattern in input_str:
            is_correct = False
            wrong_reason = "beyond_list"
        elif "if" in input_str:
            is_correct = False
            wrong_reason = "contain_if"
        else:
            is_correct = True
            for gold_condition in reformat_gold_conditions:
                if gold_condition not in input_str:
                    is_correct = False
                    wrong_reason = "wrong_identity"
                else:
                    correct_count += 1
        correct_ratio = correct_count/len(reformat_gold_conditions)

        return is_correct, wrong_reason, correct_ratio

    def check_numbers_in_string(s, N):
        for i in range(1, N + 1):
            if f"({i})" not in s:
                return False
        return True
    
    original_str = pred_str
    pred_str = pred_str.split("### Question")[0]
    pred_answer = pred_str
    is_correct = False
    correct_ratio = 0
    if reformat_gold_conditions is None:
        gold = ans.replace(" and ", "").replace(".", "")
        gold_conditions = gold.split(",")
        reformat_gold_conditions = []
        for condition in gold_conditions:
            gold_condition = condition.strip()    # Remove leading and trailing spaces
            reformat_gold_conditions.append(gold_condition)

    wrong_reason = "no_conclusion_matched"
    for pattern in conclusion_patterns:
        pred = pred_str.split(pattern)
        if len(pred) > 1:
            if len(pred[1]) > 0:  # if the matched the answer is not empty
                pred_answer = pred[1]
                is_correct, wrong_reason, correct_ratio = judge_string(
                    pred_answer, reformat_gold_conditions, wrong_reason, finish_patterns)
                break
    if is_correct == False and wrong_reason == "no_conclusion_matched": 
        if check_numbers_in_string(pred_str, len(reformat_gold_conditions)): # the answer contains (1)..(2)..
            is_correct, wrong_reason, correct_ratio = judge_string(
                pred_str, reformat_gold_conditions, wrong_reason, finish_patterns)
    if is_correct == False and verbose == True:
        print("wrong_reason:",wrong_reason)
        print("********* \nprediction before parse:\n", original_str)
        print("********* \nprediction after parse:\n", pred_answer)

    return is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions

def save_evaluation_results(config, detailed_results: list, accuracy: float, avg_correct_ratio: float, avg_ttft: float, total_count: int):
    """Save detailed evaluation results to a file for qualitative analysis."""
    import os
    from datetime import datetime
    
    # Create eval directory if it doesn't exist
    eval_dir = os.path.join(config.output_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(eval_dir, f"eval_samples_{timestamp}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Write summary metrics
        f.write("SUMMARY METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.3f} ({int(accuracy * total_count)}/{total_count})\n")
        f.write(f"Average Correct Ratio: {avg_correct_ratio:.3f}\n")
        f.write(f"Average TTFT: {avg_ttft:.3f}\n")
        f.write(f"Formatting Mode: {config.formatting_mode}\n")
        f.write(f"Datasets: {', '.join(config.datasets)}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        # Write detailed sample results
        f.write("SAMPLE-BY-SAMPLE ANALYSIS:\n")
        f.write("="*80 + "\n\n")
        
        for result in detailed_results:
            f.write(f"SAMPLE #{result['sample_id'] + 1} ({result['dataset_source'].upper()})\n")
            f.write("-" * 50 + "\n")
            
            f.write("QUESTION:\n")
            f.write(f"{result['question']}\n\n")
            
            f.write("GROUND TRUTH:\n")
            f.write(f"{result['ground_truth']}\n\n")
            
            f.write("MODEL PREDICTION:\n")
            f.write(f"{result['prediction']}\n\n")
            
            f.write("EVALUATION RESULTS:\n")
            f.write(f"• Correct: {'✓' if result['is_correct'] else '✗'}\n")
            f.write(f"• Correct Ratio: {result['correct_ratio']:.3f}\n")
            f.write(f"• TTFT Score: {result['ttft']:.3f}\n")
            if not result['is_correct']:
                f.write(f"• Wrong Reason: {result['wrong_reason']}\n")
            f.write(f"• Expected Conditions: {result['gold_conditions']}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"Detailed evaluation results saved to: {filename}")