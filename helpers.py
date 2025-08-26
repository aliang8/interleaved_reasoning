#!/usr/bin/env python3
"""
Helper functions for various utilities across the project.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import re
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils.templates import format_system_message

def create_prompts_dataproto(
    tokenizer,
    questions,
    max_prompt_length=1024,
    template_type="default",
    explicit_tasks=None,
    enable_thinking=True,
):
    """
    Create a DataProto object with prompts formatted for ActorRolloutRefWorker.

    Args:
        tokenizer: HuggingFace tokenizer
        questions: List of question strings
        max_prompt_length: Maximum length for prompt padding
        template_type: Type of system template to use
        explicit_tasks: Optional list of explicit task descriptions (stored in non_tensor_batch)
        enable_thinking: Whether to enable thinking in the chat template (default: True)

    Returns:
        DataProto object ready for generate_sequences()
    """
    batch_size = len(questions)

    # Apply chat template to format questions properly for instruction model
    formatted_prompts = []
    for question in questions:
        # Format as a conversation with configurable system template
        messages = [
            format_system_message(template_type),
            {"role": "user", "content": question},
        ]

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        formatted_prompts.append(formatted_prompt)

        # Debug output for first few prompts
        if len(formatted_prompts) <= 2:
            print(f"    Formatted prompt {len(formatted_prompts)}:")
            print(f"    {formatted_prompt}")
            print()

    # Tokenize the formatted prompts with left padding (vLLM requirement)
    tokenizer.padding_side = "left"

    encoded = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Create position_ids - this accounts for left padding
    position_ids = torch.zeros_like(input_ids)
    for i in range(batch_size):
        non_pad_mask = input_ids[i] != tokenizer.pad_token_id
        if non_pad_mask.any():
            first_token_pos = non_pad_mask.nonzero(as_tuple=False)[0][0]
            seq_len = max_prompt_length - first_token_pos
            position_ids[i, first_token_pos:] = torch.arange(seq_len)

    # Create batch TensorDict
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )

    # Meta info required by ActorRolloutRefWorker
    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "validate": False,
    }

    meta_info["original_prompt"] = questions

    if explicit_tasks:
        meta_info["explicit_tasks"] = explicit_tasks

    return DataProto(batch=batch, meta_info=meta_info)

def compute_completion_rate(
    response_text: str, template_type: str = "default", enable_thinking: bool = True
) -> float:
    task_completed = False
    if not enable_thinking:
        task_completed = bool(response_text.strip())
    elif template_type == "default":
        task_completed = "</think>" in response_text
    else:
        answer_pattern = r"<answer>.*?</answer>"
        answer_matches = re.findall(
            answer_pattern, response_text, re.DOTALL | re.IGNORECASE
        )
        task_completed = len(answer_matches) >= 2

    return task_completed


def extract_solution_from_response(
    response_text: str, template_type: str = "default", enable_thinking: bool = False
) -> str:
    """
    Extract solution from response text.
    
    Args:
        response_text: The full response text
        template_type: Template type used for generation
        
    Returns:
        Extracted solution or empty string if not found
    """
    if template_type == "plan_first":
        # For plan_first template, extract the last answer block
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_matches = re.findall(
            answer_pattern, response_text, re.DOTALL | re.IGNORECASE
        )
        
        if answer_matches:
            # Use the last answer block
            last_answer = answer_matches[-1].strip()
            return last_answer.strip()
    
    # For other template types, use the original logic
    # First, extract everything after </think> if it exists
    if "</think>" in response_text:
        parts = response_text.split("</think>")
        if len(parts) > 1:
            response_text = parts[1].strip()
    
        return response_text.strip()

    if not enable_thinking:
        return response_text.strip()

    # Return empty string if model didn't finish generating the answer
    return ""


def compute_ttft_ratio(response: str, template_type: str = "default", max_tokens: int = 4096) -> float:
    """
    Compute Time to First Token (TTFT) ratio.

    TTFT measures how quickly the model gets to the answer relative to the maximum context length.
    Lower values indicate the model gets to the answer faster.
    
    The ratio is computed as: tokens_to_answer / max_tokens
    This gives a standardized measure across different response lengths.

    Args:
        response: Response string to analyze
        template_type: Template type used for generation
        max_tokens: Maximum total tokens (default: 4096)

    Returns:
        TTFT ratio (value between 0 and 1)
    """
    if not response or not response.strip():
        return 1.0

    # Tokenize by splitting on whitespace (simple approximation)
    tokens = response.strip().split()
    response_tokens = len(tokens)

    if response_tokens == 0:
        return 1.0

    if template_type == "default":
        # For default template: look for </think> tag
        think_tag_pos = response.find("</think>")

        if think_tag_pos == -1:
            # </think> not found, set TTFT to 1.0 (worst case)
            return 1.0

        # Find the first token after </think>
        text_before_think = response[: think_tag_pos + len("</think>")]
        tokens_to_think = len(text_before_think.split())
        # Normalize: tokens to think / max_tokens
        ttft_ratio = tokens_to_think / max_tokens if max_tokens > 0 else 1.0
    else:
        # For other templates: look for <answer> tags
        answer_pattern = r"<answer>.*?</answer>"
        answer_matches = re.findall(answer_pattern, response, re.DOTALL | re.IGNORECASE)

        if not answer_matches:
            # No answer tags found, set TTFT to 1.0 (worst case)
            return 1.0

        # Find the first <answer> tag
        first_answer_pos = response.find("<answer>")
        if first_answer_pos == -1:
            return 1.0

        # Calculate tokens before first answer
        text_before_answer = response[:first_answer_pos]
        tokens_to_answer = len(text_before_answer.split())
        # Normalize: tokens to answer / max_tokens
        ttft_ratio = tokens_to_answer / max_tokens if max_tokens > 0 else 1.0

    # Ensure it's between 0 and 1
    return max(0.0, min(ttft_ratio, 1.0))


def compute_tokens_to_first_answer(response: str, template_type: str = "default") -> int:
    """
    Compute the number of tokens to the first answer.

    This function counts the actual number of tokens (not ratio) from the start
    of the response to the first answer or thinking completion.

    Args:
        response: Response string to analyze
        template_type: Template type used for generation

    Returns:
        Number of tokens to first answer (integer)
    """
    if not response or not response.strip():
        return 0

    # Tokenize by splitting on whitespace (simple approximation)
    tokens = response.strip().split()
    response_tokens = len(tokens)

    if response_tokens == 0:
        return 0

    if template_type == "default":
        # For default template: look for </think> tag
        think_tag_pos = response.find("</think>")

        if think_tag_pos == -1:
            # </think> not found, return total response length
            return response_tokens

        # Find the first token after </think>
        text_before_think = response[: think_tag_pos + len("</think>")]
        tokens_to_think = len(text_before_think.split())
        return tokens_to_think
    else:
        # For other templates: look for <answer> tags
        answer_pattern = r"<answer>.*?</answer>"
        answer_matches = re.findall(answer_pattern, response, re.DOTALL | re.IGNORECASE)

        if not answer_matches:
            # No answer tags found, return total response length
            return response_tokens

        # Find the first <answer> tag
        first_answer_pos = response.find("<answer>")
        if first_answer_pos == -1:
            return response_tokens

        # Calculate tokens before first answer
        text_before_answer = response[:first_answer_pos]
        tokens_to_answer = len(text_before_answer.split())
        return tokens_to_answer


def parse_interleaved_components(response: str) -> List[Dict[str, Any]]:
    """
    Parse interleaved think and answer components from a response string.
    
    Args:
        response: Response string that may contain <think></think> and <answer></answer> tags
        
    Returns:
        List of component dictionaries with 'type', 'index', and 'content' keys
    """
    import re
    
    if not isinstance(response, str) or not response.strip():
        return []
    
    # Find all think and answer tags with their positions
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"
    
    # Find all matches with their start positions
    think_matches = [
        (m.start(), "think", m.group(1).strip(), i + 1)
        for i, m in enumerate(
            re.finditer(think_pattern, response, re.DOTALL | re.IGNORECASE)
        )
    ]
    answer_matches = [
        (m.start(), "answer", m.group(1).strip(), i + 1)
        for i, m in enumerate(
            re.finditer(answer_pattern, response, re.DOTALL | re.IGNORECASE)
        )
    ]
    
    # If no <answer></answer> tags found, extract answer as everything after the last </think>
    if not answer_matches and think_matches:
        # Find the last </think> position
        last_think_end = (
            think_matches[-1][0] + len(think_matches[-1][2]) + 8
        )  # 8 = len("</think>")
        
        # Extract everything after the last </think> as the answer
        answer_content = response[last_think_end:].strip()
        if answer_content:
            # Add the answer component
            answer_matches = [
                (last_think_end, "answer", answer_content, len(think_matches) + 1)
            ]
    
    # Combine and sort by position to maintain chronological order
    all_matches = think_matches + answer_matches
    all_matches.sort(key=lambda x: x[0])
    
    # Create interleaved sections with additional whitespace cleaning
    interleaved_sections = []
    for _, section_type, content, index in all_matches:
        if content and content.strip():
            # Additional whitespace cleaning: remove extra newlines and normalize spacing
            cleaned_content = content.strip()
            # Remove excessive newlines (more than 2 consecutive)
            cleaned_content = re.sub(r"\n{3,}", "\n\n", cleaned_content)
            # Remove leading/trailing whitespace from each line
            cleaned_content = "\n".join(
                line.strip() for line in cleaned_content.split("\n")
            )
            
            interleaved_sections.append(
                {"type": section_type, "index": index, "content": cleaned_content}
            )
    
    return interleaved_sections


def save_to_parquet(data: List[Dict], prefix: str, output_dir: str, filename: str):
    """
    Save data to parquet file.
    
    Args:
        data: List of dictionaries to save
        prefix: Prefix for the filename
        output_dir: Output directory
        filename: Base filename
    """
    import pandas as pd
    
    df = pd.DataFrame(data)
    output_path = Path(output_dir) / f"{prefix}_{filename}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(data)} entries to {output_path}")


def save_jsonl(data: List[Dict], output_file: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        output_file: Output file path
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} entries to {output_file}")


class StandardizedRewardModel:
    """
    Standardized reward model for evaluation.
    """
    
    def __init__(
        self,
        ground_truth: List[str],
        style: str,
        unit_tests: List[str],
        libs: List[str],
    ):
        self.ground_truth = ground_truth
        self.style = style
        self.unit_tests = unit_tests
        self.libs = libs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "ground_truth": self.ground_truth,
            "style": self.style,
            "unit_tests": self.unit_tests,
            "libs": self.libs,
        }
