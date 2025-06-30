#!/usr/bin/env python3
"""
Trip planning interleaved reasoning dataset generation:
Generate multi-turn conversations where an agent first creates a high-level trip overview,
then iteratively thinks about and improves each day of the itinerary.

Usage: python generate_trip_planning_interleaved.py --output_dir trip_planning_data --model Qwen/Qwen3-32B
"""

import json
import argparse
import torch
import re
import pandas as pd
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple
import os
from tqdm import tqdm
import time
from contextlib import contextmanager
from category_generation_prompts import get_generation_prompt


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


# System prompt for trip planning with interleaved reasoning
TRIP_PLANNING_SYSTEM_PROMPT = """You are a travel planning expert assistant. You help create detailed, practical trip itineraries through an iterative planning process.

When planning trips, you follow this process:
1. First, think about the high-level overview and provide a general plan outline
2. Then, iteratively think about and refine each day of the trip with detailed planning

Always structure your responses with:
- Your thinking process within <think></think> tags
- Your answer/output within <answer></answer> tags

Be practical, consider real-world constraints like transportation schedules, booking requirements, budgets, and local factors."""


class TripPlanningGenerator:
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-32B",
                 device_map: str = "auto"):
        """Initialize the model."""
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True
        )
        print(f"Model loaded successfully")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         max_new_tokens: int = 2048,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         enable_thinking: bool = True) -> str:
        """Generate a response using the model."""
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
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
        """Generate a single turn response with stop tokens."""
        
        # Apply chat template with thinking enabled
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
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
    
    def extract_thinking_and_answer(self, response: str) -> Tuple[str, str]:
        """Extract thinking content and answer from response."""
        
        # Extract thinking content
        thinking_match = re.search('<think>(.*?)</think>', response, re.DOTALL)
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        
        # Extract answer content
        answer_match = re.search('<answer>(.*?)</answer>', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else ""
        
        # If no answer tags found, try to extract from end of response
        if not answer and thinking:
            # Everything after </think> could be the answer
            thinking_end = response.find('</think>')
            if thinking_end != -1:
                answer = response[thinking_end + 11:].strip()
        
        return thinking, answer
    
    def estimate_trip_days(self, prompt: str) -> int:
        """Estimate number of days in the trip from the prompt (constrained to 2-4 days)."""
        # Look for explicit day mentions
        day_patterns = [
            r'(\d+)[-\s]*day',
            r'(\d+)[-\s]*week',
            r'for\s+(\d+)\s+days',
            r'(\d+)\s+days?'
        ]
        
        for pattern in day_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                days = int(match.group(1))
                if 'week' in pattern:
                    days *= 7
                # Constrain to 2-4 days for manageable interleaved conversations
                return max(2, min(days, 4))
        
        # Default estimation based on trip type (constrained to 2-4 days)
        if any(word in prompt.lower() for word in ['weekend', 'short', '2']):
            return 2
        elif any(word in prompt.lower() for word in ['3', 'three']):
            return 3
        elif any(word in prompt.lower() for word in ['4', 'four', 'long', 'extended']):
            return 4
        else:
            return 3  # Default to 3 days
    
    def generate_trip_planning_conversation(self, prompt: str) -> Dict[str, Any]:
        """Generate a multi-turn trip planning conversation with alternating thinking/answer pattern."""
        
        estimated_days = self.estimate_trip_days(prompt)
        print(f"        Estimated trip duration: {estimated_days} days")
        
        conversation = []
        interleaved_parts = []
        
        # Initialize messages
        messages = [
            {"role": "system", "content": TRIP_PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Step 1: Generate thinking about the overview
        with timer("Generate overview thinking"):
            overview_thinking_prompt = f"""Please think about a high-level overview and plan for this trip request in <think></think> tags. Consider the overall structure, key considerations, and general approach.

{prompt}

Think about: overall trip structure, main destinations, key logistical considerations, and general timeline. Don't get into daily details yet - just the big picture planning."""
            
            think_messages = messages + [{"role": "user", "content": overview_thinking_prompt}]
            
            thinking_response = self.generate_single_turn(
                think_messages,
                max_new_tokens=512,
                temperature=0.2,
                enable_thinking=True
            )
            
            # Clean thinking content
            thinking_content = thinking_response.replace("<think>", "").replace("</think>", "").strip()
            thinking_content = self.clean_incomplete_content(thinking_content)
            
            print(f"          Overview thinking: {len(thinking_content)} chars")
        
        # Step 2: Generate overview answer
        with timer("Generate overview answer"):
            # Add thinking to conversation context
            updated_messages = think_messages + [{"role": "assistant", "content": f"<think>\n{thinking_content}\n</think>"}]
            
            answer_messages = updated_messages + [{"role": "user", "content": "Now provide a concise trip overview in <answer></answer> tags based on your thinking. Format it as a list of days with activities and timings:"}]
            
            answer_response = self.generate_single_turn(
                answer_messages,
                max_new_tokens=512,
                temperature=0.2,
                enable_thinking=False
            )
            
            # Clean answer content
            answer_content = answer_response.replace("<answer>", "").replace("</answer>", "").strip()
            answer_content = self.clean_incomplete_content(answer_content)
            
            print(f"          Overview answer: {len(answer_content)} chars")
            
            # Build the full response
            full_overview_response = f"<think>\n{thinking_content}\n</think>\n\n<answer>\n{answer_content}\n</answer>"
            
            # Add to conversation
            conversation.append({
                "role": "user",
                "content": overview_thinking_prompt
            })
            conversation.append({
                "role": "assistant",
                "content": full_overview_response,
                "thinking": thinking_content,
                "answer": answer_content,
                "turn_type": "high_level_planning"
            })
            
            interleaved_parts.append({
                "type": "thinking",
                "content": thinking_content,
                "turn": "overview"
            })
            interleaved_parts.append({
                "type": "answer", 
                "content": answer_content,
                "turn": "overview"
            })
        
        # Update base messages for subsequent days
        base_messages = messages + [{"role": "assistant", "content": full_overview_response}]
        
        # Step 3: Generate day-by-day planning with thinking/answer alternation
        for day in range(1, estimated_days + 1):
            with timer(f"Generate day {day} thinking"):
                day_thinking_prompt = f"""Now think about Day {day} of the trip in <think></think> tags. Consider the specific activities, logistics, timing, and practical considerations for this day. Think about what was planned in the overview and how to refine/improve the day with detailed planning.

Consider for Day {day}:
- Specific activities and timing
- Transportation logistics  
- Meal considerations
- Budget implications
- Any booking requirements or considerations
- Backup options if needed

Make this day practical and well-coordinated with the overall trip plan."""
                
                day_think_messages = base_messages + [{"role": "user", "content": day_thinking_prompt}]
                
                day_thinking_response = self.generate_single_turn(
                    day_think_messages,
                    max_new_tokens=512,
                    temperature=0.2,
                    enable_thinking=True
                )
                
                # Clean thinking content
                day_thinking_content = day_thinking_response.replace("<think>", "").replace("</think>", "").strip()
                day_thinking_content = self.clean_incomplete_content(day_thinking_content)
                
                print(f"          Day {day} thinking: {len(day_thinking_content)} chars")
            
            with timer(f"Generate day {day} answer"):
                # Add thinking to context
                day_updated_messages = day_think_messages + [{"role": "assistant", "content": f"<think>\n{day_thinking_content}\n</think>"}]
                
                day_answer_messages = day_updated_messages + [{"role": "user", "content": f"Now provide a detailed plan for Day {day} in <answer></answer> tags based on your thinking:"}]
                
                day_answer_response = self.generate_single_turn(
                    day_answer_messages,
                    max_new_tokens=512,
                    temperature=0.2,
                    enable_thinking=False
                )
                
                # Clean answer content
                day_answer_content = day_answer_response.replace("<answer>", "").replace("</answer>", "").strip()
                day_answer_content = self.clean_incomplete_content(day_answer_content)
                
                print(f"          Day {day} answer: {len(day_answer_content)} chars")
                
                # Build the full day response
                full_day_response = f"<think>\n{day_thinking_content}\n</think>\n\n<answer>\n{day_answer_content}\n</answer>"
                
                # Add to conversation
                conversation.append({
                    "role": "user",
                    "content": day_thinking_prompt
                })
                conversation.append({
                    "role": "assistant",
                    "content": full_day_response,
                    "thinking": day_thinking_content,
                    "answer": day_answer_content,
                    "turn_type": f"day_{day}_planning"
                })
                
                interleaved_parts.append({
                    "type": "thinking",
                    "content": day_thinking_content,
                    "turn": f"day_{day}"
                })
                interleaved_parts.append({
                    "type": "answer",
                    "content": day_answer_content,
                    "turn": f"day_{day}"
                })
                
                # Update base messages for next day
                base_messages.append({
                    "role": "user",
                    "content": day_thinking_prompt
                })
                base_messages.append({
                    "role": "assistant",
                    "content": full_day_response
                })
        
        return {
            "conversation": conversation,
            "interleaved_parts": interleaved_parts,
            "estimated_days": estimated_days,
            "total_turns": len([msg for msg in conversation if msg["role"] == "assistant"])
        }
    
    def generate_trip_planning_prompts(self) -> List[str]:
        """Generate trip planning prompts using the category system."""
        print("Generating trip planning prompts...")
        
        generation_prompt = get_generation_prompt("trip_planning_itinerary")
        
        messages = [
            {"role": "user", "content": generation_prompt}
        ]
        
        response = self.generate_response(
            messages, 
            max_new_tokens=1024, 
            temperature=0.8,
            enable_thinking=False
        )
        
        # Extract prompts from the response
        prompts = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering and extract the actual prompt
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            
            # Filter for substantial trip planning prompts
            if len(cleaned_line) > 50 and any(word in cleaned_line.lower() for word in 
                                            ['trip', 'travel', 'itinerary', 'vacation', 'visit', 'plan']):
                prompts.append(cleaned_line)
        
        print(f"Extracted {len(prompts)} trip planning prompts")
        return prompts


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def convert_to_parquet_format(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert results to format suitable for parquet saving."""
    parquet_data = []
    
    for result in results:
        # Create a training example for each conversation
        data = {
            "data_source": "trip_planning_interleaved",
            "prompt": result["conversation"],  # Full conversation as prompt
            "ability": "trip_planning",
            "reward_model": {"style": "conversational", "interleaved": True},
            "extra_info": {
                "original_prompt": result["original_prompt"],
                "estimated_days": result["estimated_days"],
                "total_assistant_turns": result["total_assistant_turns"],
                "total_interleaved_parts": len(result["interleaved_parts"]),
                "model_name": result["model_name"],
                "category": result["category"],
                "generation_method": result["generation_method"],
                "interleaved_parts": result["interleaved_parts"]
            }
        }
        parquet_data.append(data)
    
    return parquet_data


def main():
    parser = argparse.ArgumentParser(description="Generate trip planning interleaved conversations")
    parser.add_argument("--output_dir", type=str, default="trip_planning_data",
                       help="Output directory for generated data")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                       help="Model for generating conversations")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device map for model loading")
    parser.add_argument("--max_prompts", type=int, default=None,
                       help="Maximum number of prompts to process")
    parser.add_argument("--skip_prompt_generation", action="store_true",
                       help="Skip prompt generation and load from existing file")
    parser.add_argument("--prompts_file", type=str, default=None,
                       help="File to load prompts from if skipping generation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = TripPlanningGenerator(
        model_name=args.model,
        device_map=args.device_map
    )
    
    # Step 1: Generate or load prompts
    if args.skip_prompt_generation and args.prompts_file:
        print(f"Loading prompts from {args.prompts_file}")
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = generator.generate_trip_planning_prompts()
        
        # Save the generated prompts
        prompts_file = os.path.join(args.output_dir, "generated_trip_planning_prompts.txt")
        with open(prompts_file, 'w', encoding='utf-8') as f:
            for prompt in prompts:
                f.write(prompt + '\n')
        print(f"Saved {len(prompts)} prompts to {prompts_file}")
    
    # Limit prompts if requested
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
        print(f"Processing first {len(prompts)} prompts")
    
    # Step 2: Generate trip planning conversations
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating trip planning conversations")):
        try:
            with timer(f"Prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'"):
                conversation_data = generator.generate_trip_planning_conversation(prompt)
                
                result = {
                    "original_prompt": prompt,
                    "conversation": conversation_data["conversation"],
                    "interleaved_parts": conversation_data["interleaved_parts"],
                    "estimated_days": conversation_data["estimated_days"],
                    "total_assistant_turns": conversation_data["total_turns"],
                    "model_name": generator.model_name,
                    "category": "trip_planning_interleaved",
                    "generation_method": "alternating_think_answer"
                }
                
                results.append(result)
                
                # Print sample for first few results
                if i < 2:
                    print(f"\n--- Trip Planning Sample {i+1} ---")
                    print(f"Original prompt: {prompt}")
                    print(f"Estimated days: {conversation_data['estimated_days']}")
                    print(f"Total assistant turns: {conversation_data['total_turns']}")
                    print(f"Total interleaved parts: {len(conversation_data['interleaved_parts'])}")
                    
                    # Show interleaved parts structure  
                    parts = conversation_data['interleaved_parts']
                    if parts:
                        print(f"\nInterleaved parts breakdown:")
                        for j, part in enumerate(parts[:6]):  # Show first 6 parts
                            print(f"  Part {j+1}: {part['type']} ({part['turn']}) - {len(part['content'])} chars")
                        if len(parts) > 6:
                            print(f"  ... and {len(parts) - 6} more parts")
                    
                    # Show detailed conversation structure
                    conversation = conversation_data['conversation']
                    print(f"\nConversation structure:")
                    for j, turn in enumerate(conversation):
                        if turn['role'] == 'user':
                            print(f"  User Turn {j//2 + 1}: {turn['content']}...")
                        elif turn['role'] == 'assistant':
                            turn_type = turn.get('turn_type', 'unknown')
                            thinking_len = len(turn.get('thinking', ''))
                            answer_len = len(turn.get('answer', ''))
                            print(f"  Assistant Turn {j//2 + 1} ({turn_type}):")
                            print(f"    Thinking: {thinking_len} chars")
                            print(f"    Answer: {answer_len} chars")
                            
                            # Show thinking preview for first turn
                            if j == 1 and turn.get('thinking'):
                                print(f"    Thinking preview: {turn['thinking']}...")
                            
                            # Show answer preview for first turn  
                            if j == 1 and turn.get('answer'):
                                print(f"    Answer preview: {turn['answer']}...")
                
                # Print very detailed example for first result
                if i == 0:
                    print(f"\n{'='*50}")
                    print("DETAILED FIRST EXAMPLE")
                    print(f"{'='*50}")
                    sample_conv = conversation_data['conversation']
                    
                    # Show first user prompt
                    print(f"First User Prompt:")
                    print(f"{sample_conv[0]['content']}")
                    
                    # Show first assistant response in detail
                    if len(sample_conv) > 1:
                        first_response = sample_conv[1]
                        print(f"\nFirst Assistant Response ({first_response.get('turn_type', 'unknown')}):")
                        print(f"Full response content:")
                        print(f"{first_response['content']}...")
                        
                        if first_response.get('thinking'):
                            print(f"\nExtracted thinking:")
                            print(f"{first_response['thinking']}...")
                        
                        if first_response.get('answer'):
                            print(f"\nExtracted answer:")
                            print(f"{first_response['answer']}...")
                    
                    # Show structure of subsequent turns
                    if len(sample_conv) > 3:
                        print(f"\nSubsequent turns structure:")
                        for k in range(2, min(6, len(sample_conv)), 2):
                            if k < len(sample_conv) and k+1 < len(sample_conv):
                                user_turn = sample_conv[k]
                                assistant_turn = sample_conv[k+1]
                                turn_num = k//2 + 1
                                turn_type = assistant_turn.get('turn_type', 'unknown')
                                print(f"Turn {turn_num} ({turn_type}):")
                                print(f"  User: {user_turn['content']}...")
                                print(f"  Assistant thinking: {len(assistant_turn.get('thinking', ''))} chars")
                                print(f"  Assistant answer: {len(assistant_turn.get('answer', ''))} chars")
        
        except Exception as e:
            print(f"Error processing trip planning prompt {i}: {str(e)}")
            continue
    
    # Save results in both formats
    output_file_jsonl = os.path.join(args.output_dir, "trip_planning_interleaved_dataset.jsonl")
    output_file_parquet = os.path.join(args.output_dir, "trip_planning_interleaved_dataset.parquet")
    
    # Save JSONL
    save_jsonl(results, output_file_jsonl)
    
    # Convert and save parquet
    if results:
        parquet_data = convert_to_parquet_format(results)
        
        # Create a dataset and save as parquet
        dataset = datasets.Dataset.from_list(parquet_data)
        dataset.to_parquet(output_file_parquet)
        
        print(f"Saved parquet dataset with {len(dataset)} examples")
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRIP PLANNING GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total prompts generated: {len(prompts)}")
    print(f"Successfully processed: {len(results)}")
    print(f"Output files:")
    print(f"  JSONL: {output_file_jsonl}")
    print(f"  Parquet: {output_file_parquet}")
    
    # Show statistics
    if results:
        total_turns = [r["total_assistant_turns"] for r in results]
        avg_turns = sum(total_turns) / len(total_turns)
        print(f"\nAverage assistant turns per conversation: {avg_turns:.1f}")
        print(f"Min/Max turns: {min(total_turns)}/{max(total_turns)}")
        
        estimated_days = [r["estimated_days"] for r in results]
        avg_days = sum(estimated_days) / len(estimated_days)
        print(f"Average trip duration: {avg_days:.1f} days")
    
    # Show sample conversation structure
    if results:
        print(f"\n{'='*40}")
        print("SAMPLE CONVERSATION STRUCTURE")
        print(f"{'='*40}")
        sample = results[0]
        print(f"Original prompt: {sample['original_prompt']}")
        print(f"Estimated days: {sample['estimated_days']}")
        print(f"Total turns: {sample['total_assistant_turns']}")
        
        print("\nConversation flow:")
        for i, turn in enumerate(sample['conversation']):
            if turn['role'] == 'assistant':
                turn_type = turn.get('turn_type', 'unknown')
                thinking_len = len(turn.get('thinking', ''))
                answer_len = len(turn.get('answer', ''))
                print(f"  Turn {i//2 + 1} ({turn_type}): {thinking_len} chars thinking, {answer_len} chars answer")
    
    # Show format examples
    print(f"\n{'='*50}")
    print("DATA FORMAT EXAMPLES")
    print(f"{'='*50}")
    
    if results:
        print("Example JSONL record structure:")
        example_record = {
            "original_prompt": results[0]["original_prompt"][:100] + "...",
            "conversation": [
                {"role": "user", "content": "...user message..."},
                {"role": "assistant", "content": "...full response...", "thinking": "...extracted thinking...", "answer": "...extracted answer...", "turn_type": "high_level_planning"}
            ],
            "estimated_days": results[0]["estimated_days"],
            "total_assistant_turns": results[0]["total_assistant_turns"],
            "model_name": results[0]["model_name"],
            "category": results[0]["category"],
            "generation_method": results[0]["generation_method"]
        }
        print(json.dumps(example_record, indent=2)[:1000] + "...")
        
        print(f"\nExample Parquet record structure:")
        if results:
            parquet_example = convert_to_parquet_format(results[:1])[0]
            parquet_preview = {
                "data_source": parquet_example["data_source"],
                "prompt": "[Full conversation array...]",
                "ability": parquet_example["ability"], 
                "reward_model": parquet_example["reward_model"],
                "extra_info": parquet_example["extra_info"]
            }
            print(json.dumps(parquet_preview, indent=2))
    
    print(f"\n{'='*50}")
    print("INTERLEAVED REASONING FORMAT EXPLANATION")
    print(f"{'='*50}")
    print("""
Expected conversation format (Alternating Think/Answer Pattern):
1. User asks for trip overview planning
2. Assistant generates <think>...overview reasoning...</think>
3. Assistant generates <answer>...overview plan...</answer>
4. User asks for Day 1 details
5. Assistant generates <think>...Day 1 reasoning...</think>
6. Assistant generates <answer>...Day 1 plan...</answer>
7. User asks for Day 2 details  
8. Assistant generates <think>...Day 2 reasoning...</think>
9. Assistant generates <answer>...Day 2 plan...</answer>
... continues for each day

Each assistant response contains both parts but generated separately:
- <think>: Internal reasoning about logistics, costs, timing, etc.
- <answer>: The actual itinerary/plan output for that component

This creates true interleaved reasoning where thinking and answers are generated
in alternating turns, with stop tokens ensuring clean separation between reasoning
and output. Each thinking turn informs the subsequent answer turn.
""")
    
    print(f"{'='*50}")
    print("FILES CREATED")
    print(f"{'='*50}")
    if results:
        print(f"1. {output_file_jsonl}")
        print(f"   - Raw conversation data with full structure")
        print(f"   - {len(results)} trip planning conversations")
        print(f"   - Each record contains full multi-turn conversation")
        
        print(f"\n2. {output_file_parquet}")
        print(f"   - Training-ready format compatible with VERL")
        print(f"   - {len(results)} examples")
        print(f"   - Conversations stored in 'prompt' field as message arrays")
        
        prompts_file = os.path.join(args.output_dir, "generated_trip_planning_prompts.txt")
        if os.path.exists(prompts_file):
            print(f"\n3. {prompts_file}")
            print(f"   - Generated trip planning prompts")
            print(f"   - Can be reused with --skip_prompt_generation flag")


if __name__ == "__main__":
    main() 