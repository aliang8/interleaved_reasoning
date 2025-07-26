#!/usr/bin/env python3
"""
Generate an interleaved code reasoning dataset from a list of prompts:
- For each prompt, use Qwen-128B to generate multiple code solutions AND a detailed rationale/explanation in the same response
- Extract code blocks and the thoughts section (all non-code text)
- Use Qwen-32B to split/attribute the thoughts to each code solution, outputting <think>...</think><answer>...</answer> for each
- Parse and save as JSONL and parquet
"""
import os
import re
import json
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from interleave_generator import InterleavedResponsesGenerator
from helpers import StandardizedRewardModel, save_to_parquet, save_jsonl
from verl.workers.code_evaluator import CodeEvaluator
from omegaconf import OmegaConf

# Prompt configurations for each step
MULTIPLE_SOLUTIONS_PROMPTS = {
    "think": "Think step by step about how to solve this problem. Begin with <think> and end with </think>.",
    "answer": "Now provide the code solution in <answer></answer> tags.",
    "function_signature": "You are an expert Python developer. Given a coding prompt, write only the Python function signature for a function named task_func that solves the problem. Only output the function signature, e.g., 'def task_func(x, y, z):'",
    "unit_test": "You are an expert Python developer. Given a coding prompt and several code solutions (def task_func is defined), write a minimal Python unittest.TestCase class that tests the described functionality. Do not define task_func in the unit test, just assume it is defined. Write several small test methods (e.g., test_case1, test_case2), each with a single assert, rather than one big test with many asserts. Make sure to use task_func in the test methods. The test should be self-contained and runnable. The asserts must be correct and match the actual outputs of the provided solutions.",
    "split_prompts": "If the following prompt asks for specific, distinct ways to solve a problem explicitly (for example, 'Provide an iterative solution and one using itertools' or 'Provide at least one in-place solution and one out-of-place solution'), split it into separate prompts, one for each approach. If not, but it generally just asks for a few solutions, return 'NO'.",
    "reword_split": "Given the original prompt and a split instruction or fragment, rewrite the split instruction as a full, standalone prompt for a code solution, making it clear and self-contained.",
    "single_solution": "If the prompt is asking for several, multiple, or a few solutions, rewrite it as a prompt for a single solution. If it is already a single-solution prompt, return it unchanged."
}

def simple_generate(prompt: str, generator: InterleavedResponsesGenerator, max_new_tokens: int = 128, 
                   temperature: float = 0.2, top_p: float = 0.7, enable_thinking: bool = False, 
                   do_sample: bool = True) -> str:
    """Simple function to generate text using the generator."""
    messages = [{"role": "user", "content": prompt}]
    text = generator.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=enable_thinking
    )
    model_inputs = generator.tokenizer([text], return_tensors="pt").to(generator.model.device)
    
    with torch.no_grad():
        output = generator.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=generator.tokenizer.eos_token_id
        )
    
    output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
    response = generator.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return response

def read_prompts(prompt_file: str) -> List[str]:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def extract_code_blocks(text: str) -> List[str]:
    code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    if not code_blocks:
        code_blocks = re.findall(r"(?m:^(    |\t).+$)", text)
    code_blocks = [block.strip() for block in code_blocks if block.strip()]
    return code_blocks

def extract_answers_from_interleaved(answer: str) -> List[str]:
    """Extract individual code answers from interleaved format."""
    # The interleaved format is: <think>1</think>\n<answer>code1</answer>\n<think>2</think>\n<answer>code2</answer>
    answers = []
    
    # Extract code from <answer> tags
    answer_matches = re.findall(r'<answer>(.*?)</answer>', answer, re.DOTALL)
    
    for match in answer_matches:
        code = match.strip()
        if code:
            answers.append(code)
    
    return answers

def evaluate_single_interleaved_entry(result: Dict[str, Any], generator: InterleavedResponsesGenerator, code_evaluator: CodeEvaluator) -> bool:
    """Evaluate a single interleaved entry and return True if it passes filtering."""
    # Extract individual code answers
    answer = result.get('answer', '')
    code_answers = extract_answers_from_interleaved(answer)
    
    if len(code_answers) == 0:
        print(f"    ❌ No code answers found, skipping")
        return False
    
    # Prepare reward model info for each solution
    reward_model = result.get('reward_model', {})
    unit_tests = reward_model.get('unit_tests', '')
    libs = reward_model.get('libs', [])
    
    # Create reward model info for each solution (same unit tests for all solutions)
    rm_infos = []
    for _ in range(len(code_answers)):
        rm_info = {
            "style": "code",
            "unit_tests": unit_tests[0],
            "libs": libs,
            "ground_truth": ""
        }
        rm_infos.append(rm_info)
    
    # Evaluate each solution
    all_passed = True
    for j, code_answer in enumerate(code_answers):
        eval_result = code_evaluator.evaluate_code(
            answers=[code_answer],
            prompts=[result.get('prompt', '')],
            rm_infos=[rm_infos[j]],
            batch_indices=[0]  # Single entry evaluation
        )
        
        unit_test_pass_rate = eval_result["unit_test_pass_rate"][0]
        print(f"    Solution {j+1} pass rate: {unit_test_pass_rate:.3f}")
        
        if unit_test_pass_rate == 0.0:
            all_passed = False
            print(f"    ❌ Solution {j+1} failed all tests, skipping sample")
            break
    
    if all_passed:
        print(f"    ✅ All solutions passed, keeping sample")
    else:
        print(f"    ❌ Sample filtered out due to failing solutions")
    
    return all_passed

def parse_numbered_list(text: str) -> list:
    prompts = []
    for line in text.split('\n'):
        m = re.match(r'^\s*\d+\.\s*(.+)', line)
        if m:
            prompts.append(m.group(1).strip())
    return prompts

def to_single_solution_prompt_llm(prompt: str, generator: InterleavedResponsesGenerator, max_new_tokens=128) -> str:
    full_prompt = f"{MULTIPLE_SOLUTIONS_PROMPTS['single_solution']}\n\nPrompt: {prompt}\n\nOutput: "
    return simple_generate(full_prompt, generator, max_new_tokens)

def split_if_specific_ways(prompt: str, generator: InterleavedResponsesGenerator, max_new_tokens=256) -> tuple[Optional[list], Optional[str]]:
    full_prompt = f"{MULTIPLE_SOLUTIONS_PROMPTS['split_prompts']}\n\nPrompt: {prompt}\n\nOutput: "
    response = simple_generate(full_prompt, generator, max_new_tokens)
    return parse_numbered_list(response), response

def reword_split_prompt(split_prompt: str, original_prompt: str, generator: InterleavedResponsesGenerator, max_new_tokens=128) -> str:
    full_prompt = f"{MULTIPLE_SOLUTIONS_PROMPTS['reword_split']}\n\nOriginal prompt:\n{original_prompt}\n\nSplit instruction:\n{split_prompt}"
    return simple_generate(full_prompt, generator, max_new_tokens)
    
def generate_code_and_thoughts(prompt, previous_solutions, generator: InterleavedResponsesGenerator, max_new_tokens=512):
    """Generate code using the InterleavedResponsesGenerator."""
    # Add context about previous solutions to the prompt
    context = prompt
    if previous_solutions:
        context += "\n\nPreviously generated solutions:\n"
        for idx, sol in enumerate(previous_solutions):
            context += f"Solution {idx+1}:\n{sol}\n"
    context += "\n\nJust implement the function and the imports, do not write any other code."
    
    # Create messages
    messages = [{"role": "user", "content": context}]
    
    # Generate thinking
    messages.append({"role": "user", "content": MULTIPLE_SOLUTIONS_PROMPTS["think"]})
    
    thought = generator.generate_thoughts(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.7,
        previous_solutions=previous_solutions if previous_solutions else None
    )
    messages.append({"role": "assistant", "content": thought})
    
    # Generate answer
    messages.append({"role": "user", "content": MULTIPLE_SOLUTIONS_PROMPTS["answer"]})
    
    answer = generator.generate_answer(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.7,
        previous_solutions=previous_solutions if previous_solutions else None
    )
    messages.append({"role": "assistant", "content": answer})
    
    # Extract code from answer
    code = generator.extract_code_from_answer(answer)
    
    thought = thought.replace("<think>", "").replace("</think>", "").strip()    
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip()
    return thought, code, answer

def generate_unit_test(prompt: str, all_solutions_code: str, generator: InterleavedResponsesGenerator, max_new_tokens=512) -> str:
    user_prompt = f"Prompt:\n{prompt}\n\nCode Solutions:\n{all_solutions_code}\n"
    full_prompt = f"{MULTIPLE_SOLUTIONS_PROMPTS['unit_test']}\n\n{user_prompt}"
    response = simple_generate(full_prompt, generator, max_new_tokens, do_sample=False)
    code = generator.extract_code_from_answer(response)
    return code

# --- Add function to generate function signature using the code model ---
def generate_function_signature(prompt: str, generator: InterleavedResponsesGenerator, max_new_tokens=64) -> str:
    full_prompt = f"{MULTIPLE_SOLUTIONS_PROMPTS['function_signature']}\n\nPrompt: {prompt}"
    response = simple_generate(full_prompt, generator, max_new_tokens, do_sample=False)
    # Clean up: only keep the first line that starts with 'def task_func'
    for line in response.splitlines():
        if line.strip().startswith('def task_func'):
            return line.strip()
    # fallback: return the first line
    return response.splitlines()[0].strip() if response else 'def task_func():'

def main():
    parser = argparse.ArgumentParser(description="Generate interleaved code reasoning dataset from prompts")
    parser.add_argument('--prompt_file', type=str, default='data_gen/code_list_prompts.txt', help='Text file with one prompt per line')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--num_code_solutions', type=int, default=2, help='Number of code solutions per prompt (if available)')
    parser.add_argument('--code_model', type=str, default='Qwen/Qwen3-32B', help='Model name for generation')
    parser.add_argument('--device_map', type=str, default='auto', help='Device map for model loading')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prompts = read_prompts(args.prompt_file)

    print(f"\n🚀 Loading generator: {args.code_model}")
    generator = InterleavedResponsesGenerator(
        model_name=args.code_model,
        device_map=args.device_map
    )

    interleaved_results = []
    
    # Initialize code evaluator once
    config = OmegaConf.create({
        "max_concurrent": 4,
        "execute_sequential": False
    })
    code_evaluator = CodeEvaluator(config=config, tokenizer=generator.tokenizer)
    
    for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        print(f"\n---\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
        # --- Generate function signature for this prompt ---
        func_signature = generate_function_signature(prompt, generator)
        print(f"  [Function signature] {func_signature}")
        # --- Prepend required context to prompt as a code block ---
        preamble = (
            "You should write self-contained code starting with:\n"
            "```python\n"
            f"{func_signature}\n"
            "```\n"
        )
        prompt_with_preamble = prompt + "\n\n" + preamble
        split_prompts, split_response = split_if_specific_ways(prompt, generator)
        code_solutions = []
        if split_prompts:
            print(f"Detected specific approaches, split into {len(split_prompts)} prompts.")
            for idx, sp in enumerate(split_prompts):
                print(f"  {idx+1}. {sp}")
            for j, split_prompt in enumerate(split_prompts):
                reworded_prompt = reword_split_prompt(split_prompt, prompt, generator)
                print(f"  Reworded split prompt: {reworded_prompt}")
                print(f"Generating code for split prompt {j+1}: {reworded_prompt}")
                reworded_prompt = reworded_prompt + "\n\n" + preamble
                thought, code, code_response = generate_code_and_thoughts(reworded_prompt, [], generator)
                print(f"  Thought: {thought[:100]}...")
                code_blocks = extract_code_blocks(code_response)
                code_to_save = code_blocks[0] if code_blocks else code
                code_solutions.append({
                    "split_prompt": split_prompt,
                    "reworded_prompt": reworded_prompt,
                    "thought": thought,
                    "code": code_to_save,
                    "raw_response": code_response
                })
            # Build interleaved answer string
            interleaved = ""
            for sol in code_solutions:
                interleaved += f"<think>{sol['thought']}</think>\n<answer>{sol['code']}</answer>\n"
            interleaved_results.append({
                "data_source": "multiple_solutions_interleave",
                "prompt": prompt,
                "answer": interleaved.strip(),
                "extra_info": {
                    "split": "train",
                    "index": i,
                    "question": [prompt],
                    "answer": [interleaved.strip()]
                }
                # reward_model will be added below
            })
        else:
            single_prompt = to_single_solution_prompt_llm(prompt, generator)
            print(f"Single-solution prompt: {single_prompt}")
            previous_codes = []
            for j in range(args.num_code_solutions):
                if j == 0:
                    context_prompt = single_prompt
                else:
                    context_prompt = (
                        f"{single_prompt}\n\nPreviously generated solution(s):\n"
                        + "\n\n".join([f"Solution {k+1}:\n{code}" for k, code in enumerate(previous_codes)])
                        + "\n\nPlease think of a different way to solve the problem."
                    )
                context_prompt = context_prompt + "\n\n" + preamble
                thought, code, code_response = generate_code_and_thoughts(context_prompt, previous_codes, generator)
                print(f"  Thought: {thought[:100]}...")
                code_blocks = extract_code_blocks(code_response)
                code_to_save = code_blocks[0] if code_blocks else code
                code_solutions.append({
                    "single_prompt": single_prompt,
                    "context_prompt": context_prompt,
                    "thought": thought,
                    "code": code_to_save,
                    "raw_response": code_response
                })
                previous_codes.append(code_to_save)

            # Build interleaved answer string
            interleaved = ""
            for sol in code_solutions:
                interleaved += f"<think>{sol['thought']}</think>\n<answer>{sol['code']}</answer>\n"
            interleaved_results.append({
                "data_source": "multiple_solutions_interleave",
                "prompt": prompt_with_preamble,
                "answer": interleaved.strip(),
                "extra_info": {
                    "split": "train",
                    "index": i,
                    "question": [prompt],
                    "answer": [interleaved.strip()]
                }
            })

        # --- Generate unit test for all code solutions ---
        all_solutions_code = "\n\n".join(
            f"# Solution {idx+1}:\n{sol['code']}" for idx, sol in enumerate(code_solutions)
        )
        unit_test = generate_unit_test(prompt, all_solutions_code, generator)
        
        # Create standardized reward model
        reward_model = StandardizedRewardModel(
            ground_truth="",  # No single ground truth for multiple solutions
            style="code",
            unit_tests=[unit_test],
            libs=[]
        )
        
        # Add reward_model to the last interleaved_results entry
        interleaved_results[-1]["reward_model"] = reward_model.to_dict()
        
        # Evaluate the entry immediately
        print(f"  🔍 Evaluating generated entry...")
        if evaluate_single_interleaved_entry(interleaved_results[-1], generator, code_evaluator):
            print(f"  ✅ Entry {i+1} passed evaluation and will be saved")
        else:
            print(f"  ❌ Entry {i+1} failed evaluation, removing from results")
            interleaved_results.pop()  # Remove the failed entry

    # Save successful interleaved results
    filename_prefix = "sft/multiple_solutions_interleave"
    save_to_parquet(interleaved_results, "", args.output_dir, filename_prefix)
    save_jsonl(interleaved_results, os.path.join(args.output_dir, f"{filename_prefix}.jsonl"))
    print(f"\n✅ All done! {len(interleaved_results)} interleaved prompts processed and saved.")

if __name__ == "__main__":
    main() 