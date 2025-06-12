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
import numpy as np
from transformers import AutoTokenizer, AutoConfig
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


def create_prompts_dataproto(tokenizer, questions, max_prompt_length=256):
    """
    Create a DataProto object with prompts formatted for ActorRolloutRefWorker.
    
    Args:
        tokenizer: HuggingFace tokenizer
        questions: List of question strings
        max_prompt_length: Maximum length for prompt padding
        
    Returns:
        DataProto object ready for generate_sequences()
    """
    batch_size = len(questions)
    
    # Tokenize the questions with left padding (vLLM requirement)
    tokenizer.padding_side = "left"
    
    # First tokenize without padding to check lengths
    encoded_no_pad = tokenizer(questions, return_tensors="pt", add_special_tokens=True)
    
    # Check if any sequence is longer than max_prompt_length
    max_actual_length = encoded_no_pad["input_ids"].shape[1]
    if max_actual_length > max_prompt_length:
        print(f"Warning: Some sequences ({max_actual_length} tokens) are longer than max_prompt_length ({max_prompt_length})")
        print("Truncating to fit max_prompt_length...")
    
    # Now tokenize with proper padding and truncation
    encoded = tokenizer(
        questions,
        padding="max_length",
        max_length=max_prompt_length,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    print(f"Tokenized input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    print(f"Padding token ID: {tokenizer.pad_token_id}")
    
    # Verify padding is correct (should be left-padded)
    for i in range(batch_size):
        non_pad_tokens = (input_ids[i] != tokenizer.pad_token_id).sum().item()
        first_non_pad = (input_ids[i] != tokenizer.pad_token_id).nonzero(as_tuple=False)[0][0].item()
        print(f"Sequence {i}: {non_pad_tokens} non-pad tokens, first non-pad at position {first_non_pad}")
    
    # Create position_ids - this accounts for left padding
    position_ids = torch.zeros_like(input_ids)
    for i in range(batch_size):
        # Find the first non-pad token
        non_pad_mask = (input_ids[i] != tokenizer.pad_token_id)
        if non_pad_mask.any():
            first_token_pos = non_pad_mask.nonzero(as_tuple=False)[0][0]
            seq_len = max_prompt_length - first_token_pos
            position_ids[i, first_token_pos:] = torch.arange(seq_len)
        else:
            # All tokens are padding (shouldn't happen with real input)
            print(f"Warning: Sequence {i} is all padding tokens!")
    
    # Verify position_ids are correct
    print(f"Position IDs shape: {position_ids.shape}")
    print(f"Sample position_ids[0]: {position_ids[0]}")
    
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


def setup_actor_rollout_config(model_path, prompt_length=256, response_length=256):
    """
    Create configuration for ActorRolloutRefWorker following FSDP workers pattern.
    
    Args:
        model_path: Path to the model
        prompt_length: Maximum prompt length
        response_length: Maximum response length
        
    Returns:
        DictConfig object with proper structure for ActorRolloutRefWorker
    """
    return DictConfig({
        # Model configuration
        "model": {
            "path": model_path,
            "tokenizer_path": model_path,  # Use same path for tokenizer
            "trust_remote_code": False,
            "use_remove_padding": False,
            "use_fused_kernels": False,
            "enable_gradient_checkpointing": False,
            "use_liger": False,
            "enable_activation_offload": False,
            "lora_rank": 0,  # No LoRA for this demo
            "override_config": {},
            "external_lib": None,
            "use_shm": False,
        },
        
        # Actor configuration (minimal since we're only doing rollout)
        "actor": {
            "strategy": "fsdp",
            "fsdp_config": {
                "fsdp_size": -1,  # Use full world size
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
            "name": "vllm_with_tool",
            "mode": "sync",
            
            # Basic rollout settings
            "prompt_length": prompt_length,
            "response_length": response_length,
            "max_model_len": None,
            "n": 5,  # Number of responses per prompt
            
            # vLLM specific settings
            "tensor_model_parallel_size": 1,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.3,
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
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "do_sample": True,
            
            # Log probability settings
            "log_prob_micro_batch_size": 2,
            "log_prob_micro_batch_size_per_gpu": 2,
            "log_prob_max_token_len_per_gpu": 1024,
            "log_prob_use_dynamic_bsz": False,
            
            # Validation settings
            "val_kwargs": {
                "top_k": -1,
                "top_p": 1.0,
                "temperature": 0,
                "n": 1,
                "do_sample": False,
            },
            
            # Engine kwargs
            "engine_kwargs": {
                "vllm": {
                    "swap_space": 4,
                }
            },

            "max_turns": 10,
        },
        
        # Reference policy config (minimal, not used in this demo)
        "ref": {
            "fsdp_config": {
                "param_offload": True,  # Offload ref to save memory
                "wrap_policy": None,
                "forward_prefetch": True,
            },
            "log_prob_micro_batch_size_per_gpu": 1,
            "log_prob_max_token_len_per_gpu": 1024,
            "log_prob_use_dynamic_bsz": False,
        }
    })


def main():
    """Main demo function."""
    print("=== ActorRolloutRefWorker vLLM Demo ===\n")
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Show GPU memory before starting
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {memory_total:.2f}GB")
        print()
    else:
        print("❌ CUDA not available! This demo requires a GPU.")
        return False
    
    # Model setup
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_path}")
    
    # Load tokenizer for prompt preparation
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create configuration for ActorRolloutRefWorker
    config = setup_actor_rollout_config(model_path)
    print(f"Config: prompt_length={config.rollout.prompt_length}, response_length={config.rollout.response_length}")
    print(f"Rollout config: tensor_parallel_size={config.rollout.tensor_model_parallel_size}, dtype={config.rollout.dtype}")
    
    # Initialize ActorRolloutRefWorker
    print("\nInitializing ActorRolloutRefWorker...")
    try:
        # Create worker with rollout role (we only need rollout functionality)
        worker = ActorRolloutRefWorker(config=config, role="rollout")
        print("ActorRolloutRefWorker initialized successfully!")
        
        # Initialize the model
        print("Initializing model...")
        worker.init_model()
        print("Model initialized successfully!")
        
        # Show GPU memory after model loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory after model loading - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
    except Exception as e:
        print(f"❌ Error initializing ActorRolloutRefWorker: {e}")
        print("\nTip: Make sure you have:")
        print("- A GPU available")
        print("- vLLM properly installed")
        print("- Sufficient GPU memory")
        print("- Compatible CUDA version")
        import traceback
        traceback.print_exc()
        raise
    
    # Prepare sample questions
    questions = [
        "You have access to search tools. Please search for information about the capital of France using <tool_call>query</tool_call> format.",
        # "Explain artificial intelligence in one sentence.",
        # "What is 2 + 2?",
    ]
    
    print(f"\nPreparing {len(questions)} prompts...")
    prompts = create_prompts_dataproto(tokenizer, questions, config.rollout.prompt_length)
    
    print(f"Prompt batch shape: {prompts.batch['input_ids'].shape}")
    print(f"Expected shape: ({len(questions)}, {config.rollout.prompt_length})")
    
    # Show detailed tokenization info for debugging
    for i, question in enumerate(questions):
        input_ids = prompts.batch['input_ids'][i]
        attention_mask = prompts.batch['attention_mask'][i]
        position_ids = prompts.batch['position_ids'][i]
        
        # Find actual content (non-padding)
        non_pad_mask = input_ids != tokenizer.pad_token_id
        actual_tokens = input_ids[non_pad_mask]
        
        print(f"\n--- Question {i+1} Tokenization ---")
        print(f"Original: {question}")
        print(f"Tokenized length: {len(actual_tokens)} tokens")
        print(f"Padded length: {len(input_ids)} tokens")
        print(f"First 10 tokens: {input_ids[:10].tolist()}")
        print(f"Last 10 tokens: {input_ids[-10:].tolist()}")
        print(f"Decoded (first 50 chars): {tokenizer.decode(actual_tokens)[:50]}...")
        print(f"Attention mask sum: {attention_mask.sum().item()}")
        print(f"Position IDs range: {position_ids.min().item()} to {position_ids.max().item()}")
    
    print(f"\nAll prompts properly padded to length {config.rollout.prompt_length}")
    print(f"Ready for vLLM generation...")
    
    # Generate responses using ActorRolloutRefWorker
    print("\n🚀 Generating responses...")
    try:
        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate sequences through the worker
        output = worker.generate_sequences(prompts)
        print("✅ Generation completed!")
        
        # Show results
        print(f"\nResults:")
        print(f"- Batch size: {len(output)}")
        print(f"- Response tensor shape: {output.batch['responses'].shape}")
        
        # Decode and display each result
        for i, question in enumerate(questions):
            print(f"\n--- Example {i+1} ---")
            print(f"Q: {question}")
            
            # Get response tokens (excluding padding)
            response_tokens = output.batch['responses'][i]
            response_tokens = response_tokens[response_tokens != tokenizer.pad_token_id]
            
            # Decode response
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            print(f"A: {response_text}")
            print(f"   ({len(response_tokens)} tokens)")
            
            # Show log probabilities if available
            if 'rollout_log_probs' in output.batch:
                valid_logprobs = output.batch['rollout_log_probs'][i][:len(response_tokens)]
                avg_logprob = valid_logprobs.mean().item()
                print(f"   (avg log prob: {avg_logprob:.3f})")
        
        print(f"\n✨ Demo completed successfully!")
        
        # Final memory check
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Final GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        print("\nThis could be due to:")
        print("- Insufficient GPU memory")
        print("- CUDA compatibility issues")
        print("- vLLM configuration problems")
        print("- Model loading issues")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def show_dataproto_structure():
    """
    Helper function to show the expected DataProto structure.
    """
    print("=== DataProto Structure for ActorRolloutRefWorker ===\n")
    
    print("Required batch fields:")
    print("  - input_ids: torch.Tensor of shape (batch_size, prompt_length)")
    print("  - attention_mask: torch.Tensor of shape (batch_size, prompt_length)")  
    print("  - position_ids: torch.Tensor of shape (batch_size, prompt_length)")
    
    print("\nRequired meta_info fields:")
    print("  - eos_token_id: int (end of sequence token)")
    print("  - do_sample: bool (enable sampling)")
    print("  - validate: bool (validation mode flag)")
    
    print("\nOptional non_tensor_batch fields:")
    print("  - raw_prompt_ids: numpy array of token lists (auto-generated if missing)")
    print("  - multi_modal_data: for multimodal inputs")
    
    print("\nKey points:")
    print("  - Use left padding for input_ids and attention_mask")
    print("  - position_ids should account for padding")
    print("  - All tensors in batch must have same batch_size")
    print("  - Configuration follows ActorRolloutRefWorker structure")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ActorRolloutRefWorker vLLM Demo")
    parser.add_argument("--show-structure", action="store_true", 
                       help="Show expected DataProto structure")
    args = parser.parse_args()
    
    if args.show_structure:
        show_dataproto_structure()
    else:
        main() 