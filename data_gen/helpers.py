import os
import pandas as pd
import json
from typing import List, Dict, Any
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standardized reward model structure
class StandardizedRewardModel:
    """Standardized reward model structure for all datasets."""
    
    def __init__(self, ground_truth, style="rule", unit_tests=None, libs=None, **kwargs):
        self.ground_truth = ground_truth
        self.style = style
        self.unit_tests = unit_tests or []
        self.libs = libs or []
        # Store any additional fields as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary format."""
        result = {
            "ground_truth": self.ground_truth,
            "style": self.style,
            "unit_tests": self.unit_tests,
            "libs": self.libs
        }
        # Add any additional fields
        for key, value in self.__dict__.items():
            if key not in ["ground_truth", "style", "unit_tests", "libs"]:
                result[key] = value
        return result


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data as JSONL for debugging."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Saved {len(data)} samples to {file_path}")

def save_to_parquet(data, suffix, local_dir, output_filename):
    """Save data to parquet file."""
    os.makedirs(local_dir, exist_ok=True)
    if suffix:
        output_path = os.path.join(local_dir, f"{output_filename}_{suffix}.parquet")
    else:
        output_path = os.path.join(local_dir, f"{output_filename}.parquet")
    pd.DataFrame(data).to_parquet(output_path)
    print(f"Saved {len(data)} samples to {output_path}")

def save_to_parquet_all(all_datasets, local_dir, output_filename):
    """Save all datasets to individual parquet files and a combined parquet file."""
    os.makedirs(local_dir, exist_ok=True)
    
    # Save individual parquet files
    print(f"\nSaving individual parquet files to {local_dir}:")
    for dataset_name, (train_data, test_data) in all_datasets.items():
        if train_data is not None:
            save_to_parquet(train_data, "train", local_dir, f"{dataset_name}")
        
        if test_data is not None:
            save_to_parquet(test_data, "test", local_dir, f"{dataset_name}")
    
    # Combine all datasets for combined parquet file
    all_data = []
    for dataset_name, (train_data, test_data) in all_datasets.items():
        if train_data is not None:
            all_data.extend(train_data)
        if test_data is not None:
            all_data.extend(test_data)
    
    # Save to combined parquet file
    save_to_parquet(all_data, "all", local_dir, output_filename)
    
    # Print summary
    print(f"\nDataset Summary:")
    for dataset_name, (train_data, test_data) in all_datasets.items():
        train_count = len(train_data) if train_data is not None else 0
        test_count = len(test_data) if test_data is not None else 0
        print(f"  {dataset_name}: {train_count} train, {test_count} test")
    
    # Show example from first dataset
    if all_data:
        print(f"\nExample from combined dataset:")
        print(all_data[0])


LLM_MODEL = None
LLM_TOKENIZER = None
LLM_MODEL_NAME = None
LLM_DEVICE_MAP = None


def merge_prompts_with_llm(prompts, model_name=None, device_map=None):
    """
    Use Qwen LLM to merge prompts into a single, natural question.
    """
    global LLM_MODEL, LLM_TOKENIZER, LLM_MODEL_NAME, LLM_DEVICE_MAP
    if LLM_MODEL is None or LLM_TOKENIZER is None or model_name != LLM_MODEL_NAME or device_map != LLM_DEVICE_MAP:
        print(f"Loading LLM for prompt merging: {model_name} (device_map={device_map})")
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        LLM_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        LLM_MODEL_NAME = model_name
        LLM_DEVICE_MAP = device_map
    # Compose a system prompt
    system_prompt = "You are an expert at combining multiple short questions into a single, natural, clear question."
    user_prompt = "Combine the following questions into a single, natural question that asks for all the information together.\n\n" + "\n".join(f"- {p}" for p in prompts)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = LLM_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = LLM_TOKENIZER([text], return_tensors="pt").to(LLM_MODEL.device)
    with torch.no_grad():
        output = LLM_MODEL.generate(
            **model_inputs,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.7,
            do_sample=True,
            pad_token_id=LLM_TOKENIZER.eos_token_id
        )
    output_ids = output[0][len(model_inputs.input_ids[0]):].tolist()
    response = LLM_TOKENIZER.decode(output_ids, skip_special_tokens=True).strip()
    return response


def combine_examples(examples, prompt_key="question", answer_key="answer", sep=" ", answer_sep=", ", 
                    prompt_combine_mode="space", llm_model_name=None, llm_device_map=None,
                    prompt_prefix=None):
    """
    Combine a list of examples into a single prompt and a single answer.
    - prompt_combine_mode: 'space', 'and', 'llm', or 'numbered_list'
    - prompt_prefix: Optional prefix to add to the beginning of the combined prompt
    """
    if isinstance(examples, dict):
        prompts = examples[prompt_key]
    else:
        prompts = [example[prompt_key] for example in examples]

    # Remove trailing punctuation from all but the last prompt if combining with 'and' or 'space', and lowercase the second and later prompts
    if prompt_combine_mode in ("and", "space") and len(prompts) > 1:
        cleaned_prompts = []
        for i, p in enumerate(prompts):
            cleaned = p.strip()
            if i < len(prompts) - 1:
                cleaned = re.sub(r'[\.,!?;:]+$', '', cleaned)
            if i > 0:
                cleaned = cleaned.lower()
            cleaned_prompts.append(cleaned)
        prompts = cleaned_prompts
    
    # Format prompts based on combine mode
    if prompt_combine_mode == "numbered_list" and len(prompts) > 1:
        formatted_prompts = []
        for i, prompt in enumerate(prompts, 1):
            formatted_prompts.append(f"{i}) {prompt}")
        combined_prompt = "\n".join(formatted_prompts)
    elif prompt_combine_mode == "and":
        combined_prompt = " and ".join(prompts)
    elif prompt_combine_mode == "llm":
        combined_prompt = merge_prompts_with_llm(prompts, model_name=llm_model_name, device_map=llm_device_map)
    else:
        combined_prompt = sep.join(prompts)
    
    # Add prefix if provided
    if prompt_prefix:
        combined_prompt = f"{prompt_prefix}\n{combined_prompt}"

    if isinstance(examples, dict):
        answers = examples[answer_key]
    else:
        answers = [example[answer_key] for example in examples]
    combined_answer = answer_sep.join(answers)
    return {"prompt": combined_prompt, "answer": combined_answer}