import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Optional
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
import pandas as pd
import subprocess
import tempfile
from datasets import load_dataset


class InterleavedResponsesGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-32B", device_map: str = "auto"):
        """Initialize the model and tokenizer."""
        print(f"Loading model: {model_name}")
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )

        print(f"Model loaded successfully")

    def generate_thoughts(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        previous_solutions: List[str] = None,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate thinking response with automatic tag fixing."""

        # Condition on previous solutions if provided
        if previous_solutions:
            # Find the last user message and append previous solutions context
            last_user_msg_idx = -1
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    last_user_msg_idx = i

            if last_user_msg_idx != -1:
                context = messages[last_user_msg_idx]["content"]
                context += "\n\nPreviously generated solutions:\n"
                for idx, sol in enumerate(previous_solutions):
                    context += f"Solution {idx + 1}:\n{sol}\n"
                context += (
                    "\n\nPlease think of a different approach to solve this problem."
                )
                messages[last_user_msg_idx]["content"] = context

        # Apply chat template with thinking enabled
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate response
        with torch.no_grad():
            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "stop_strings": ["</think>", "</answer>"],
                "tokenizer": self.tokenizer,
                "num_return_sequences": num_return_sequences,
            }

            generated_ids = self.model.generate(**generate_kwargs)
            
        # Extract only the generated part (remove input)
        responses = []
        for ids in generated_ids:
            output_ids = ids[len(model_inputs.input_ids[0]) :].tolist()
            response = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()

            # Fix incomplete thinking tag
            if "<think>" in response and "</think>" not in response:
                think_start = response.find("<think>")
                if think_start != -1:
                    content_after_think = response[think_start + 7 :]
                    cleaned_content = self.clean_incomplete_content(content_after_think)
                    response = (
                        response[: think_start + 7] + cleaned_content + "</think>"
                    )
                else:
                    response += "</think>"

            responses.append(response)
        return responses

    def generate_answer(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        previous_solutions: List[str] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate answer response with automatic tag fixing."""

        # Condition on previous solutions if provided
        if previous_solutions:
            # Find the last user message and append previous solutions context
            last_user_msg_idx = -1
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    last_user_msg_idx = i

            if last_user_msg_idx != -1:
                context = messages[last_user_msg_idx]["content"]
                context += "\n\nPreviously generated solutions:\n"
                for idx, sol in enumerate(previous_solutions):
                    context += f"Solution {idx + 1}:\n{sol}\n"
                context += "\n\nPlease provide a different solution approach."
                messages[last_user_msg_idx]["content"] = context

        # Apply chat template without thinking
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate response
        with torch.no_grad():
            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": num_return_sequences,
            }

            generated_ids = self.model.generate(**generate_kwargs)

        # Extract only the generated part (remove input)
        responses = []
        for ids in generated_ids:
            output_ids = ids[len(model_inputs.input_ids[0]) :].tolist()
            response = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()

            # Fix incomplete answer tag
            if "<answer>" in response and "</answer>" not in response:
                answer_start = response.find("<answer>")
                if answer_start != -1:
                    content_after_answer = response[answer_start + 8 :]
                    cleaned_content = self.clean_incomplete_content(
                        content_after_answer
                    )
                    response = (
                        response[: answer_start + 8] + cleaned_content + "</answer>"
                    )
                else:
                    response += "</answer>"

            responses.append(response)

        return responses

    def clean_incomplete_content(self, content: str) -> str:
        """Remove the last sentence if it appears to be incomplete."""
        if not content.strip():
            return content

        # Split into sentences (improved approach)
        sentences = re.split(r"(?<=[.!?])\s+", content.strip())

        # If no proper sentences found, try splitting by line breaks as backup
        if len(sentences) == 1 and "\n" in content:
            sentences = [s.strip() for s in content.split("\n") if s.strip()]

        # If still only one piece and it doesn't end with punctuation, consider it incomplete
        if len(sentences) == 1:
            if not content.strip().endswith((".", "!", "?", ":", ";")):
                return ""  # Remove incomplete single sentence
        else:
            # Remove last sentence if it doesn't end with proper punctuation
            last_sentence = sentences[-1]
            if not last_sentence.endswith((".", "!", "?", ":", ";")):
                cleaned = " ".join(sentences[:-1])
                return cleaned

        return content

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.7,
        previous_solutions: List[str] = None,
    ) -> str:
        """Generate a complete response (thinking + solution)."""

        # Condition on previous solutions if provided
        if previous_solutions:
            # Find the last user message and append previous solutions context
            last_user_msg_idx = -1
            for i, msg in enumerate(messages):
                if msg["role"] == "user":
                    last_user_msg_idx = i

            if last_user_msg_idx != -1:
                context = messages[last_user_msg_idx]["content"]
                context += "\n\nPreviously generated solutions:\n"
                for idx, sol in enumerate(previous_solutions):
                    context += f"Solution {idx + 1}:\n{sol}\n"
                context += (
                    "\n\nPlease think of a different approach to solve this problem."
                )
                messages[last_user_msg_idx]["content"] = context

        # Apply chat template with thinking enabled
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate response
        with torch.no_grad():
            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "tokenizer": self.tokenizer,
            }

            generated_ids = self.model.generate(**generate_kwargs)

        # Extract only the generated part (remove input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return response

    def extract_solution_from_response(self, response: str) -> str:
        """Extract the solution part from a response (everything after </think>)."""
        # Find the </think> token
        think_end = response.find("</think>")
        if think_end != -1:
            # Extract everything after </think>
            solution = response[think_end + 8 :].strip()
            return solution
        else:
            # If no </think> found, return the entire response
            return response.strip()

    def extract_code_from_answer(self, answer: str) -> str:
        """Extract code from an answer response."""
        # First try to extract from <answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
        if answer_match:
            code = answer_match.group(1).strip()
        else:
            # If no tags, use the entire answer
            code = answer.strip()

        # Extract code from markdown code blocks if present
        if "```python" in code:
            code = code.split("```python")[-1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[-1].split("```")[0].strip()

        return code
