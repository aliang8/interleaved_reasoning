#!/usr/bin/env python3
"""
Simplified example of using ActorRolloutRefWorker for vLLM rollout.

This script shows the basic pattern for:
1. Setting up the configuration following FSDP workers pattern
2. Creating input data in the correct format  
3. Calling generate_sequences through the worker
4. Processing the output

Run with: python simple_vllm_rollout_demo.py
"""

import os
import torch
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from omegaconf import DictConfig
from tensordict import TensorDict

# Set environment variables for distributed setup (single GPU)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1") 
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto
from verl.utils.templates import format_system_message

# Global configuration for ActorRolloutRefWorker
ACTOR_ROLLOUT_CONFIG = {
    # Model configuration
    "model": {
        # "path": "Qwen/Qwen3-8B",
        "path": "checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave",
        "tokenizer_path": "Qwen/Qwen3-8B",
        "trust_remote_code": False,
        "use_remove_padding": False,
        "use_fused_kernels": False,
        "enable_gradient_checkpointing": False,
        "use_liger": False,
        "enable_activation_offload": False,
        "lora_rank": 0,
        "override_config": {},
        "external_lib": None,
        "use_shm": False,
    },
    
    # Actor configuration (minimal since we're only doing rollout)
    "actor": {
        "strategy": "fsdp",
        "fsdp_config": {
            "fsdp_size": -1,
            "param_offload": False,
            "optimizer_offload": False,
            "wrap_policy": None,
            "forward_prefetch": True,
            "model_dtype": "bfloat16",
            "mixed_precision": {
                "param_dtype": "bfloat16",
                "reduce_dtype": "float32", 
                "buffer_dtype": "float32"
            }
        },
        "optim": {
            "lr": 1e-6,
            "betas": [0.9, 0.999],
            "weight_decay": 1e-2,
            "total_training_steps": 1000,
            "lr_warmup_steps": 0,
            "warmup_style": "constant"
        },
        "ppo_mini_batch_size": 1,
        "ppo_micro_batch_size_per_gpu": 1,
        "checkpoint": {
            "contents": ["model", "optimizer", "lr_scheduler"]
        }
    },
    
    # Rollout configuration - this is where vLLM settings go
    "rollout": {
        "name": "vllm_autorater",
        "mode": "sync",
        
        # Basic rollout settings
        "prompt_length": 1024,
        "response_length": 4096,
        "max_model_len": None,
        "n": 1,  # Number of responses per prompt
        
        # Template configuration
        "template_type": "interleave",
        
        # vLLM specific settings
        "tensor_model_parallel_size": 1,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.8,
        "enforce_eager": False,
        "free_cache_engine": False,
        "load_format": "auto",
        "disable_log_stats": True,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": 2,
        "enable_chunked_prefill": False,
        "disable_custom_all_reduce": True,
        "seed": 42,
        
        # Sampling parameters
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": -1,
        "do_sample": True,
        
        # Log probability settings
        "log_prob_micro_batch_size": 2,
        "log_prob_micro_batch_size_per_gpu": 2,
        "log_prob_max_token_len_per_gpu": 1024,
        "log_prob_use_dynamic_bsz": False,

        # mcp settings
        "mcp_mode": "direct_article",
        "mcp_timeout": 10,
        "mcp_max_article_tokens": 2000,
        
        # Validation settings
        "val_kwargs": {
            "top_k": -1,
            "top_p": 1.0,
            "temperature": 0,
            "n": 1,
            "do_sample": True,
        },
        
        # Engine kwargs
        "engine_kwargs": {
            "vllm": {
                "swap_space": 4,
            }
        },

        "max_turns": 5,
    },
    
    # Reference policy config (minimal, not used in this demo)
    "ref": {
        "fsdp_config": {
            "param_offload": True,
            "wrap_policy": None,
            "forward_prefetch": True,
        },
        "log_prob_micro_batch_size_per_gpu": 1,
        "log_prob_max_token_len_per_gpu": 1024,
        "log_prob_use_dynamic_bsz": False,
    }
}


def create_prompts_dataproto(tokenizer, questions, max_prompt_length=1024, template_type="default"):
    """
    Create a DataProto object with prompts formatted for ActorRolloutRefWorker.
    
    Args:
        tokenizer: HuggingFace tokenizer
        questions: List of question strings
        max_prompt_length: Maximum length for prompt padding
        template_type: Type of system template to use
        
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
            {"role": "user", "content": question}
        ]
        
        # Apply the chat template
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True
            )
            formatted_prompts.append(formatted_prompt)
            
            # Debug output for first few prompts
            if len(formatted_prompts) <= 2:
                print(f"    Formatted prompt {len(formatted_prompts)}:")
                print(f"    {formatted_prompt[:200]}...")
                print()
                
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
            # Fallback to simple format
            formatted_prompt = f"System: {messages[0]['content']}\n\nUser: {messages[1]['content']}\n\nAssistant:"
            formatted_prompts.append(formatted_prompt)
    
    # Tokenize the formatted prompts with left padding (vLLM requirement)
    tokenizer.padding_side = "left"
    
    encoded = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Create position_ids - this accounts for left padding
    position_ids = torch.zeros_like(input_ids)
    for i in range(batch_size):
        non_pad_mask = (input_ids[i] != tokenizer.pad_token_id)
        if non_pad_mask.any():
            first_token_pos = non_pad_mask.nonzero(as_tuple=False)[0][0]
            seq_len = max_prompt_length - first_token_pos
            position_ids[i, first_token_pos:] = torch.arange(seq_len)
    
    # Create batch TensorDict
    batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask, 
        "position_ids": position_ids,
    }, batch_size=batch_size)
    
    # Meta info required by ActorRolloutRefWorker
    meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "validate": False,
    }
    
    # Non-tensor batch (can be empty for basic usage)
    non_tensor_batch = {}
    
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)





def load_validation_data(file_paths: List[str]) -> List[str]:
    """Load validation prompts from files."""
    prompts = []
    
    for file_path in file_paths:
        path = Path(file_path)
        print(f"    Loading file: {file_path}")
        
        try:
            if path.suffix.lower() == '.txt':
                # Load from text file (one prompt per line)
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_prompts = []
                for line in lines:
                    line = line.strip()
                    if line:
                        file_prompts.append(line)
                
                prompts.extend(file_prompts)
                print(f"    ✓ Loaded {len(file_prompts)} prompts from text file")
                
            elif path.suffix.lower() == '.jsonl':
                # Load from JSONL file
                with open(path, 'r', encoding='utf-8') as f:
                    file_prompts = []
                    for line in f:
                        data = json.loads(line.strip())
                        if 'prompt' in data:
                            file_prompts.append(data['prompt'])
                        elif 'question' in data:
                            file_prompts.append(data['question'])
                        elif 'content' in data:
                            file_prompts.append(data['content'])
                
                prompts.extend(file_prompts)
                print(f"    ✓ Loaded {len(file_prompts)} prompts from JSONL file")
                
            elif path.suffix.lower() == '.parquet':
                # Load from parquet file
                import pandas as pd
                df = pd.read_parquet(path)
                print(f"    Columns: {list(df.columns)}")
                
                file_prompts = []
                if 'prompt' in df.columns:
                    file_prompts = df['prompt'].tolist()
                elif 'question' in df.columns:
                    file_prompts = df['question'].tolist()
                elif 'content' in df.columns:
                    file_prompts = df['content'].tolist()
                else:
                    # Use first column if standard columns not found
                    first_col = df.columns[0]
                    print(f"    Warning: Using first column '{first_col}' as prompts")
                    file_prompts = df[first_col].tolist()
                
                # Filter out None/NaN values
                file_prompts = [str(p).strip() for p in file_prompts if pd.notna(p) and str(p).strip()]
                prompts.extend(file_prompts)
                print(f"    ✓ Loaded {len(file_prompts)} prompts from parquet file")
                
            else:
                print(f"    Warning: Unsupported file format {path.suffix}")
                
        except Exception as e:
            print(f"    ✗ Failed to load file {file_path}: {e}")
            continue
    
    return prompts


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="ActorRolloutRefWorker vLLM Demo")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B",
                       help="Path to the model")
    parser.add_argument("--template_type", type=str, default="default",
                       help="System template type")
    parser.add_argument("--val_data", type=str, nargs='+', 
                       help="Validation data file(s)")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=4096,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    print("=== ActorRolloutRefWorker vLLM Demo ===\n")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This demo requires a GPU.")
        return False
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Model setup
    print(f"Loading model: {args.model_path}")
    print(f"Using template type: {args.template_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if tokenizer supports thinking
    has_thinking_support = hasattr(tokenizer, 'apply_chat_template') and 'enable_thinking' in tokenizer.apply_chat_template.__code__.co_varnames
    print(f"Tokenizer thinking support: {has_thinking_support}")
    if not has_thinking_support:
        print("Warning: Tokenizer may not support enable_thinking parameter")
    
    # Load validation data
    if args.val_data:
        print(f"Loading validation data from {len(args.val_data)} file(s):")
        questions = load_validation_data(args.val_data)
        
        if not questions:
            print("❌ No prompts loaded from validation files!")
            return False
            
        print(f"✓ Successfully loaded {len(questions)} prompts from validation files")
    else:
        # Default sample questions
        questions = [
            "Explain the concept of machine learning in simple terms.",
            "Write a short poem about the ocean.",
            "What are the benefits of renewable energy?",
            "How does photosynthesis work?",
            "What is the capital of France?"
        ]
        print(f"Using {len(questions)} default sample questions")
    
    # Limit number of samples
    if len(questions) > args.num_samples:
        questions = questions[:args.num_samples]
        print(f"Limited to {len(questions)} samples")
    else:
        print(f"Processing all {len(questions)} available questions")
    
    # Use global configuration and update with command line arguments
    config = ACTOR_ROLLOUT_CONFIG.copy()
    config["model"]["path"] = args.model_path
    config["model"]["tokenizer_path"] = args.model_path
    config["rollout"]["template_type"] = args.template_type
    config["rollout"]["temperature"] = args.temperature
    config["rollout"]["response_length"] = args.max_tokens
    
    print(f"Config: prompt_length={config['rollout']['prompt_length']}, response_length={config['rollout']['response_length']}")
    print(f"Generation: temperature={config['rollout']['temperature']}, top_p={config['rollout']['top_p']}")
    
    # Initialize ActorRolloutRefWorker
    print("\nInitializing ActorRolloutRefWorker...")
    try:
        worker = ActorRolloutRefWorker(config=DictConfig(config), role="rollout")
        print("✓ ActorRolloutRefWorker initialized successfully!")
        
        worker.init_model()
        print("✓ Model initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing ActorRolloutRefWorker: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Prepare prompts
    print(f"\nPreparing {len(questions)} prompts...")
    prompts = create_prompts_dataproto(
        tokenizer, 
        questions, 
        config["rollout"]["prompt_length"],
        template_type=config["rollout"]["template_type"]
    )
    
    print(f"Prompt batch shape: {prompts.batch['input_ids'].shape}")
    
    # Generate responses
    print("\n🚀 Generating responses...")
    try:
        torch.cuda.empty_cache()
        
        output = worker.generate_sequences(prompts)
        print("✅ Generation completed!")
        
        # Process results
        results = []
        for i, question in enumerate(questions):
            # Get response tokens (excluding padding)
            response_tokens = output.batch['responses'][i]
            response_tokens = response_tokens[response_tokens != tokenizer.pad_token_id]
            
            # Decode response
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            result = {
                'question': question,
                'response': response_text,
                'num_tokens': len(response_tokens)
            }
            results.append(result)
            
            print(f"\n--- Example {i+1} ---")
            print(f"Q: {question}")
            print(f"A: {response_text}")
            print(f"   ({len(response_tokens)} tokens)")
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "generation_results.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"\n✅ Results saved to {output_file}")
        print(f"✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 