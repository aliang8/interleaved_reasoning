#!/usr/bin/env python3
"""
Unified script to generate reasoning datasets using Qwen:
1. Ask Qwen to generate examples of prompts for different reasoning categories (3-5 items for listing)
2. Use those prompts with interleaved system message to generate responses
3. Save as parquet files with train/test split for SFT training

Supports both listing and interleaved reasoning categories:
- listing: General listing prompts requiring multi-step reasoning (3-5 items)
- historical_chronological: Temporal progressions
- procedural_instructional: Step-by-step procedures
- cause_effect: Causal chains and domino effects
- narrative_storytelling: Plot and character progressions
- ordered_preference: Ranked lists with specific criteria

Default output: Parquet files with train/test split for SFT training

Usage: python generate_listing_prompts_with_qwen.py --output_dir reasoning_data --category listing
"""

import json
import argparse
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
import pandas as pd
from sklearn.model_selection import train_test_split
from category_generation_prompts import get_generation_prompt, get_available_categories


@contextmanager
def timer(name: str, verbose: bool = True):
    """Context manager for timing operations."""
    start_time = time.time()
    if verbose:
        print(f"    Starting {name}...")
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print(f"    {name} completed in {duration:.2f}s")


# Interleaved system prompt template
# INTERLEAVED_SYSTEM_PROMPT = """You are a helpful assistant that provides step-by-step reasoning for complex problems. 

# When solving problems that involve multiple steps or listing items, please structure your response as follows:
# 1. Use <answer>content</answer> tags to wrap each discrete step, item, or conclusion
# 2. Show your thinking process between <think>content</think> tags
# 3. For listing tasks, each bullet point should be wrapped in <answer> tags
# 4. Ensure each <answer> represents a complete, helpful piece of information

# Example format:
# I need to think about this step by step.

# <think>Thinking content...</think>
# <answer>First key point or step</answer>

# <think>Thinking content...</think>
# <answer>Second key point or step</answer>

# And so on for each item in the list or step in the reasoning process."""
INTERLEAVED_SYSTEM_PROMPT = """You are a helpful assistant that thinks compositionally about complex problems. When working with lists or multi-part problems, you break them down into individual components and reason about each one separately. You conduct your reasoning within <think></think> tags, focusing on just one item or component at a time. You then provide that specific item within <answer></answer> tags, including both the item itself and a brief one-sentence summary explaining why this item qualifies or was chosen. You think deeply about each component - considering what criteria it meets, what research or analysis is needed, and how it relates to the overall problem. This compositional approach helps ensure thorough, accurate reasoning for each element."""

class QwenListingGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-32B", device_map: str = "auto"):
        """Initialize the Qwen model and tokenizer."""
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        
        print(f"Model loaded successfully")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_new_tokens: int = 4096,
                         temperature: float = 0.8,
                         top_p: float = 0.95,
                         enable_thinking: bool = False) -> str:
        """Generate a response for given messages."""
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Disable thinking for prompt generation
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract only the generated part (remove input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_single_turn(self, messages: List[Dict[str, str]], 
                            max_new_tokens: int = 512,
                            temperature: float = 0.2,
                            top_p: float = 0.7,
                            enable_thinking: bool = False) -> str:
        """Generate a single turn response."""
        
        # Apply chat template with thinking enabled
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Enable thinking mode
        )
        
        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Set up stop tokens - will stop at either thinking or answer end
        stop_strings = None
        if enable_thinking:
            stop_strings = ["</think>", "</answer>"]
        
        # Generate response
        with torch.no_grad():
            generate_kwargs = {
                **model_inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add stop strings and tokenizer if thinking is enabled
            if enable_thinking and stop_strings:
                generate_kwargs["stop_strings"] = stop_strings
                generate_kwargs["tokenizer"] = self.tokenizer
            
            generated_ids = self.model.generate(**generate_kwargs)
        
        # Extract only the generated part (remove input)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return response

    def extract_expected_count(self, prompt: str) -> int:
        """Extract expected number of items from prompt."""
        import re
        
        # Look for patterns like "List 7", "List 10", etc.
        match = re.search(r'list\s+(\d+)', prompt.lower())
        if match:
            return int(match.group(1))
        
        # Look for other patterns like "7 ways", "10 steps", etc.
        match = re.search(r'(\d+)\s+(?:ways|steps|items|things|points|stages|factors|methods)', prompt.lower())
        if match:
            return int(match.group(1))
            
                # Default fallback
        return 8

    def clean_incomplete_content(self, content: str) -> str:
        """Remove the last sentence if it appears to be incomplete."""
        if not content.strip():
            return content
        
        # Split into sentences (improved approach)
        # First split by proper sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', content.strip())
        
        # If no proper sentences found, try splitting by line breaks as backup
        if len(sentences) == 1 and '\n' in content:
            sentences = [s.strip() for s in content.split('\n') if s.strip()]
        
        if len(sentences) <= 1:
            return content
        
        last_sentence = sentences[-1].strip()
        
        # Check if last sentence looks incomplete
        incomplete_indicators = [
            not re.search(r'[.!?]$', last_sentence),  # Doesn't end with punctuation
            len(last_sentence) < 8,  # Very short
            last_sentence.endswith(','),  # Ends with comma
            last_sentence.endswith('...'),  # Ends with ellipsis (might be cut off)
            re.search(r'\b(the|and|or|but|because|since|when|if|that|which|who|so|then|now|first|next|also)$', last_sentence, re.IGNORECASE),  # Ends with connecting word
            re.search(r'\b(I|we|he|she|it|they|this|that|there|here)\s*(am|is|are|was|were|will|would|should|could|might|need|want|think)?\s*$', last_sentence, re.IGNORECASE),  # Ends with incomplete subject-verb
            re.search(r'\b(to|for|from|with|by|in|on|at|of|about)\s*$', last_sentence, re.IGNORECASE),  # Ends with preposition
        ]
        
        if any(incomplete_indicators):
            # Remove the last sentence
            cleaned = ' '.join(sentences[:-1])
            print(f"        Cleaned incomplete sentence: '{last_sentence[:30]}...'")
            return cleaned
        
        return content

    def generate_thinking_response(self, prompt: str,
                                  max_new_tokens_per_turn: int = 512,
                                  temperature: float = 0.2,
                                  top_p: float = 0.7) -> Dict[str, str]:
        """Generate multi-turn interleaved thinking response with simplified two-step approach."""
        
        # Extract expected number of items
        expected_count = self.extract_expected_count(prompt)
        print(f"    Expected items: {expected_count}")
        
        print(f"    🎯 Generating {expected_count} items with interleaved reasoning")
        with timer(f"Full trace generation ({expected_count} items)", verbose=True):
            interleaved_parts = []
            individual_thinking = []
            individual_answers = []
            
            # Initialize conversation
            messages = [
                {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            for turn in range(expected_count):
                print(f"      🔄 Starting turn {turn + 1}/{expected_count}")
                with timer(f"Turn {turn + 1}/{expected_count} generation", verbose=True):
                    
                    if turn == 0:
                        thinking_prompt = f"Let me work through this step by step. First, focus only on identifying the first item for this list. Think compositionally about what criteria need to be evaluated and what makes this item qualify. Start with <think> and end with </think>."
                    else:
                        prev_items = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(individual_answers)])
                        thinking_prompt = f"Great! So far we have:\n{prev_items}\n\nNow think about item #{turn + 1} specifically. Start with <think> and end with </think>."
                    
                    # Step 1: Generate thinking
                    print(f"        🧠 Generating thinking for item {turn + 1}")
                    messages.append({"role": "user", "content": thinking_prompt})
                    
                    # Generate thinking with 256 token limit and </think> as stop token
                    thinking_response = self.generate_single_turn(
                        messages,
                        max_new_tokens=256,  # Hard limit for thinking
                        temperature=temperature,
                        top_p=top_p,
                        enable_thinking=True
                    )
                    print(f"        📝 Raw thinking response length: {len(thinking_response)} chars")
                    
                    # Fix incomplete thinking tag
                    if '<think>' in thinking_response and '</think>' not in thinking_response:
                        # Extract content after <think> and clean it
                        think_start = thinking_response.find('<think>')
                        if think_start != -1:
                            content_after_think = thinking_response[think_start + 7:]  # Everything after <think>
                            cleaned_content = self.clean_incomplete_content(content_after_think)
                            thinking_response = thinking_response[:think_start + 7] + cleaned_content + "</think>"
                        else:
                            thinking_response += "</think>"
                        print(f"        Fixed and cleaned incomplete thinking tag")
                    
                    messages.append({"role": "assistant", "content": thinking_response})
                    
                    # Step 2: Generate answer
                    print(f"        💡 Generating answer for item {turn + 1}")
                    answer_prompt = f"Now provide your answer for item #{turn + 1} in <answer></answer> tags. Include both the item and a brief explanation."
                    messages.append({"role": "user", "content": answer_prompt})
                    
                    # Generate answer with 128 token limit and </answer> as stop token  
                    answer_response = self.generate_single_turn(
                        messages,
                        max_new_tokens=128,  # Answers should be shorter than thinking
                        temperature=temperature,
                        top_p=top_p,
                        enable_thinking=False
                    )
                    print(f"        📝 Raw answer response length: {len(answer_response)} chars")
                    
                    # Fix incomplete answer tag
                    if '<answer>' in answer_response and '</answer>' not in answer_response:
                        # Extract content after <answer> and clean it
                        answer_start = answer_response.find('<answer>')
                        if answer_start != -1:
                            content_after_answer = answer_response[answer_start + 8:]  # Everything after <answer>
                            cleaned_content = self.clean_incomplete_content(content_after_answer)
                            answer_response = answer_response[:answer_start + 8] + cleaned_content + "</answer>"
                        else:
                            answer_response += "</answer>"
                        print(f"        Fixed and cleaned incomplete answer tag")
                    
                    messages.append({"role": "assistant", "content": answer_response})
                    
                    # Extract content
                    think_match = re.search(r'<think>(.*?)</think>', thinking_response, re.DOTALL)
                    thinking_content = think_match.group(1).strip() if think_match else ""
                    
                    answer_match = re.search(r'<answer>(.*?)</answer>', answer_response, re.DOTALL)
                    answer_content = answer_match.group(1).strip() if answer_match else ""
                    
                    # Debug output for empty content
                    if not thinking_content:
                        print(f"        ⚠️  Warning: No thinking content extracted from response")
                    if not answer_content:
                        print(f"        ⚠️  Warning: No answer content extracted from response")
                    
                    # Store parts
                    individual_thinking.append(thinking_content)
                    individual_answers.append(answer_content)
                    
                    # Build interleaved response
                    if thinking_content or answer_content:  # Only add if we have some content
                        turn_interleaved = f"<think>\n{thinking_content}\n</think>\n\n<answer>{answer_content}</answer>"
                        interleaved_parts.append(turn_interleaved)
                        
                        # Count approximate tokens (rough estimate: ~4 chars per token)
                        thinking_tokens = len(thinking_content) // 4
                        answer_tokens = len(answer_content) // 4
                        print(f"        ✅ Completed turn {turn + 1}: thinking={len(thinking_content)} chars (~{thinking_tokens} tokens), answer={len(answer_content)} chars (~{answer_tokens} tokens)")
                        
                        # Warn if thinking exceeds expected limit
                        if thinking_tokens > 256:
                            print(f"        ⚠️  Warning: Thinking section has ~{thinking_tokens} tokens, exceeds 256 token limit")
                    else:
                        print(f"        ❌ Skipping turn {turn + 1}: no valid content extracted")
            
            # Combine into final interleaved response
            full_interleaved = "\n\n".join(interleaved_parts)
            
            # Generation summary
            successful_turns = len(interleaved_parts)
            print(f"    📊 Generation summary: {successful_turns}/{expected_count} turns completed successfully")
            if successful_turns < expected_count:
                print(f"    ⚠️  Warning: Expected {expected_count} items but only generated {successful_turns}")
            
            return {
                "thinking": "",  # Not used
                "answer": "",    # Not used
                "full_response": full_interleaved,
                "turns": interleaved_parts,
                "individual_thinking": individual_thinking,
                "individual_answers": individual_answers
            }

    def step1_generate_category_prompts(self, category: str) -> List[str]:
        """Step 1: Generate prompts for a specific category."""
        print(f"Step 1: Generating {category} prompt examples...")
        
        # Get the generation prompt for this category
        try:
            generation_prompt = get_generation_prompt(category)
        except ValueError as e:
            print(f"Error: {e}")
            return []
        
        messages = [
            {"role": "user", "content": generation_prompt}
        ]
        
        response = self.generate_response(messages, max_new_tokens=4096, temperature=0.8)
        
        # Extract prompts from the response
        prompts = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering and extract the actual prompt
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            
            # For listing category, filter for "List" prompts, for others just filter by length
            if category == "listing":
                if cleaned_line.lower().startswith('list ') and len(cleaned_line) > 10:
                    prompts.append(cleaned_line)
            else:
                if len(cleaned_line) > 20:  # Filter out very short lines
                    prompts.append(cleaned_line)
        
        print(f"Extracted {len(prompts)} {category} prompts")
        return prompts
    
    def step2_generate_interleaved_responses(self, prompts: List[str], category: str) -> List[Dict[str, Any]]:
        """Step 2: Generate interleaved responses for prompts from a specific category."""
        print(f"Step 2: Generating interleaved responses for {category}...")
        
        results = []
        
        for i, prompt in enumerate(tqdm(prompts, desc=f"Generating {category} responses")):
            try:
                # Generate interleaved response
                response_data = self.generate_thinking_response(prompt)
                
                # Count how many <answer> tags are in the response
                answer_count = response_data["full_response"].count("<answer>")
                
                # Create result entry
                result = {
                    "prompt": prompt,
                    "messages": [
                        {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "generated_thinking": response_data["thinking"],
                    "generated_answer": response_data["answer"],
                    "generated_full_response": response_data["full_response"],
                    "answer_tag_count": answer_count,
                    "model_name": self.model_name,
                    "category": category,
                                                "generation_params": {
                                "max_new_tokens_per_turn": args.max_tokens_per_turn,
                                "temperature": 0.2,
                                "top_p": 0.7
                            },
                    "turns": response_data.get("turns", []),
                    "individual_thinking": response_data.get("individual_thinking", []),
                    "individual_answers": response_data.get("individual_answers", [])
                }
                
                results.append(result)
                
                # Print sample for first few results
                if i < 2:
                    print(f"\n--- {category.title()} Sample {i+1} ---")
                    print(f"Prompt: {prompt}")
                    print(f"Answer tags found: {answer_count}")
                    print(f"Number of turns: {len(response_data.get('turns', []))}")
                    print(f"Individual answers: {response_data.get('individual_answers', [])[:3]}...")  # Show first 3
                    print(f"Combined answer preview:\n{response_data['answer'][:500]}...")
                
            except Exception as e:
                print(f"Error processing {category} prompt {i}: {str(e)}")
                continue
        
        return results
    
    def _infer_category(self, prompt: str) -> str:
        """Infer category from prompt content."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['program', 'code', 'software', 'api', 'debug', 'test']):
            return "programming"
        elif any(word in prompt_lower for word in ['business', 'startup', 'marketing', 'entrepreneur']):
            return "business"
        elif any(word in prompt_lower for word in ['green card', 'visa', 'legal', 'law', 'court']):
            return "legal"
        elif any(word in prompt_lower for word in ['health', 'medical', 'doctor', 'medicine']):
            return "health"
        elif any(word in prompt_lower for word in ['environment', 'climate', 'carbon', 'green']):
            return "environmental"
        elif any(word in prompt_lower for word in ['security', 'vulnerability', 'cyber', 'hack']):
            return "cybersecurity"
        elif any(word in prompt_lower for word in ['finance', 'invest', 'money', 'stock', 'financial']):
            return "finance"
        elif any(word in prompt_lower for word in ['education', 'learn', 'teach', 'school']):
            return "education"
        elif any(word in prompt_lower for word in ['travel', 'country', 'geography', 'place']):
            return "travel"
        else:
            return "general"


def save_parquet_with_split(data: List[Dict[str, Any]], output_dir: str, filename_prefix: str, test_size: float = 0.1):
    """Save data to parquet files with train/test split."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Split into train and test
    if len(data) > 1:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    else:
        train_df = df
        test_df = df.iloc[:0]  # Empty DataFrame with same structure
    
    # Save to parquet files
    train_path = os.path.join(output_dir, f"{filename_prefix}_train.parquet")
    test_path = os.path.join(output_dir, f"{filename_prefix}_test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    if not test_df.empty:
        test_df.to_parquet(test_path, index=False)
    
    print(f"Saved {len(train_df)} train samples to {train_path}")
    if not test_df.empty:
        print(f"Saved {len(test_df)} test samples to {test_path}")
    
    return train_path, test_path


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file (legacy function)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning prompts and interleaved responses with Qwen")
    parser.add_argument("--output_dir", type=str, default="reasoning_data",
                       help="Output directory for generated data (default: reasoning_data)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B",
                       help="Model name to use (default: Qwen/Qwen3-32B)")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map for model loading (default: auto)")
    parser.add_argument("--max_prompts", type=int, default=None,
                       help="Maximum number of prompts to process (default: all)")
    parser.add_argument("--skip_step1", action="store_true",
                       help="Skip step 1 and load prompts from existing file")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="File to load prompts from if skipping step 1")
    parser.add_argument("--category", type=str, default="listing",
                       choices=get_available_categories(),
                       help=f"Category of prompts to generate (choices: {', '.join(get_available_categories())})")
    parser.add_argument("--max_tokens_per_turn", type=int, default=768,
                       help="Maximum tokens per turn for multi-turn generation (default: 768)")
    parser.add_argument("--save_format", type=str, default="parquet", choices=["parquet", "jsonl"],
                       help="Output format (default: parquet)")
    parser.add_argument("--test_size", type=float, default=0.1,
                       help="Test set size ratio (default: 0.1)")
    parser.add_argument("--samples_per_prompt", type=int, default=5,
                       help="Number of reasoning traces to sample per prompt (default: 5)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = QwenListingGenerator(
        model_name=args.model_name,
        device_map=args.device_map
    )
    
    # Step 1: Generate prompts for selected category
    if args.skip_step1 and args.prompts_file:
        print(f"Loading prompts from {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = generator.step1_generate_category_prompts(args.category)
        
        # Save the generated prompts
        prompts_file = os.path.join(args.output_dir, f"generated_{args.category}_prompts.txt")
        with open(prompts_file, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        print(f"Saved {len(prompts)} prompts to {prompts_file}")
    
    # Limit prompts if requested
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Processing first {len(prompts)} prompts")
    
    # Step 2: Generate multiple interleaved responses per prompt
    results = []
    total_samples = len(prompts) * args.samples_per_prompt
    
    print(f"\n🚀 Starting generation of {total_samples} interleaved reasoning traces...")
    generation_start_time = time.time()
    
    with tqdm(total=total_samples, desc=f"Generating {args.category} responses") as pbar:
        for i, prompt in enumerate(prompts):
            for sample_idx in range(args.samples_per_prompt):
                try:
                    sample_id = f"{i+1}.{sample_idx+1}"
                    print(f"\n🔄 Starting generation for sample {sample_id}")
                    with timer(f"Complete interleaved trace for sample {sample_id}", verbose=True):
                        response_data = generator.generate_thinking_response(
                            prompt, 
                            max_new_tokens_per_turn=args.max_tokens_per_turn
                        )
                        
                        answer_count = response_data["full_response"].count("<answer>")
                        
                        result = {
                            "prompt": prompt,
                            "prompt_id": i,
                            "sample_id": sample_idx,
                            "unique_id": f"prompt_{i}_sample_{sample_idx}",
                            "messages": [
                                {"role": "system", "content": INTERLEAVED_SYSTEM_PROMPT},
                                {"role": "user", "content": prompt}
                            ],
                            "generated_thinking": response_data["thinking"],
                            "generated_answer": response_data["answer"],
                            "generated_full_response": response_data["full_response"],
                            "answer_tag_count": answer_count,
                            "model_name": generator.model_name,
                            "category": args.category,
                            "generation_params": {
                                "max_new_tokens_per_turn": args.max_tokens_per_turn,
                                "temperature": 0.2,
                                "top_p": 0.7
                            },
                            "turns": response_data.get("turns", []),
                            "individual_thinking": response_data.get("individual_thinking", []),
                            "individual_answers": response_data.get("individual_answers", [])
                        }
                        
                        results.append(result)
                        
                        print(f"    ✅ Successfully generated sample {sample_id} with {answer_count} answer tags")
                        
                        # Print each generated trace to see what it looks like
                        print(f"\n{'='*80}")
                        print(f"SAMPLE {sample_id}: {args.category.upper()} INTERLEAVED TRACE")
                        print(f"{'='*80}")
                        print(f"PROMPT: {prompt}")
                        print(f"\nANSWER TAGS: {answer_count}")
                        print(f"TURNS: {len(response_data.get('turns', []))}")
                        print(f"\nINTERLEAVED RESPONSE:")
                        print("-" * 80)
                        
                        # Create clean interleaved format from individual parts
                        individual_thinking = response_data.get('individual_thinking', [])
                        individual_answers = response_data.get('individual_answers', [])
                        
                        for i in range(len(individual_answers)):
                            if i < len(individual_thinking) and individual_thinking[i]:
                                print(f"<think>\n{individual_thinking[i]}\n</think>\n")
                            if individual_answers[i]:
                                print(f"<answer>{individual_answers[i]}</answer>\n")
                        
                        print("-" * 80)
                        
                        pbar.update(1)
                
                except Exception as e:
                    print(f"❌ Error processing {args.category} prompt {i} sample {sample_idx}: {str(e)}")
                    pbar.update(1)
                    continue
    
    generation_end_time = time.time()
    total_generation_time = generation_end_time - generation_start_time
    avg_time_per_sample = total_generation_time / len(results) if results else 0
    
    print(f"\n⏱️  GENERATION PERFORMANCE SUMMARY")
    print(f"    Total generation time: {total_generation_time:.1f}s ({total_generation_time/60:.1f} minutes)")
    print(f"    Successful samples: {len(results)}/{total_samples}")
    print(f"    Average time per sample: {avg_time_per_sample:.1f}s")
    if results:
        total_answer_tags = sum(r.get('answer_tag_count', 0) for r in results)
        avg_tags_per_sample = total_answer_tags / len(results)
        avg_time_per_tag = total_generation_time / total_answer_tags if total_answer_tags > 0 else 0
        print(f"    Total answer tags generated: {total_answer_tags}")
        print(f"    Average tags per sample: {avg_tags_per_sample:.1f}")
        print(f"    Average time per answer tag: {avg_time_per_tag:.1f}s")
    
    # Prepare data for SFT training with clean interleaved responses
    print(f"\n📝 Preparing SFT training data...")
    with timer("Data preparation and cleaning", verbose=True):
        sft_data = []
        for result in results:
            # Create clean interleaved trace
            individual_thinking = result.get('individual_thinking', [])
            individual_answers = result.get('individual_answers', [])
            
            interleaved_trace = ""
            for i in range(len(individual_answers)):
                if i < len(individual_thinking) and individual_thinking[i]:
                    interleaved_trace += f"<think>\n{individual_thinking[i]}\n</think>\n\n"
                if individual_answers[i]:
                    interleaved_trace += f"<answer>{individual_answers[i]}</answer>\n\n"
            
            interleaved_trace = interleaved_trace.strip()  # Remove trailing whitespace
            
            sft_entry = {
                "question": result["prompt"],
                "answer": interleaved_trace
            }
            sft_data.append(sft_entry)
    
    # Save results in chosen format
    print(f"\n💾 Saving data in {args.save_format} format...")
    with timer(f"Data saving ({args.save_format} format)", verbose=True):
        if args.save_format == "parquet":
            # Save as parquet with train/test split
            filename_prefix = f"interleaved_{args.category}_dataset"
            train_path, test_path = save_parquet_with_split(sft_data, args.output_dir, filename_prefix, args.test_size)
            output_files = [train_path, test_path]
        else:
            # Legacy JSONL format
            output_file = os.path.join(args.output_dir, f"interleaved_{args.category}_dataset.jsonl")
            save_jsonl(sft_data, output_file)
            output_files = [output_file]
    
    # Print summary
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total prompts generated in step 1: {len(prompts)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Expected total samples: {len(prompts) * args.samples_per_prompt}")
    print(f"Successfully processed samples: {len(results)}")
    print(f"Output format: {args.save_format}")
    if args.save_format == "parquet":
        print(f"SFT-ready files: {output_files}")
        print(f"Train/test split ratio: {1-args.test_size:.1f}/{args.test_size:.1f}")
    else:
        print(f"Output file: {output_files[0]}")
    
    # Show sample from final data
    if sft_data:
        print(f"\n{'='*30}")
        print("SAMPLE SFT DATA")
        print(f"{'='*30}")
        sample = sft_data[0]
        print(f"Question: {sample['question']}")
        print(f"Answer (first 500 chars):\n{sample['answer'][:500]}...")
        if len(sample['answer']) > 500:
            print("... (truncated)")
        
        # Count answer tags in the cleaned data
        answer_counts = [entry['answer'].count('<answer>') for entry in sft_data]
        avg_answers = sum(answer_counts) / len(answer_counts)
        print(f"\nAverage <answer> tags per response: {avg_answers:.1f}")
        print(f"Min/Max answer tags: {min(answer_counts)}/{max(answer_counts)}")


if __name__ == "__main__":
    main() 