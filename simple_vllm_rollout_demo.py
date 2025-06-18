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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import re
import string

# Set environment variables for distributed setup (single GPU)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1") 
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl import DataProto
from verl.verl.utils.templates import format_system_message


def create_prompts_dataproto(tokenizer, questions, max_prompt_length=1024, template_type="confidence"):
    """
    Create a DataProto object with prompts formatted for ActorRolloutRefWorker.
    
    Args:
        tokenizer: HuggingFace tokenizer
        questions: List of question strings
        max_prompt_length: Maximum length for prompt padding
        template_type: Type of system template to use ("tool", "tool_interleaved", "confidence", "default")
        
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
                add_generation_prompt=True  # Add assistant prompt for generation
            )
            formatted_prompts.append(formatted_prompt)
            print(f"Original: {question}")
            print(f"Template: {template_type}")
            print(f"Formatted: {formatted_prompt}")
            print("---")
        except Exception as e:
            print(f"Warning: Could not apply chat template: {e}")
            print("Falling back to raw question format")
            formatted_prompts.append(question)
    
    # Tokenize the formatted prompts with left padding (vLLM requirement)
    tokenizer.padding_side = "left"
    
    # Now tokenize with proper padding and truncation
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


def setup_actor_rollout_config(model_path, prompt_length=256, response_length=256, template_type="confidence"):
    """
    Create configuration for ActorRolloutRefWorker following FSDP workers pattern.
    
    Args:
        model_path: Path to the model
        prompt_length: Maximum prompt length
        response_length: Maximum response length
        template_type: Type of system template to use ("tool", "tool_interleaved", "confidence", "default")
        
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
            "name": "vllm",
            "mode": "sync",
            
            # Basic rollout settings
            "prompt_length": prompt_length,
            "response_length": 5000,
            "max_model_len": None,
            "n": 1,  # Number of responses per prompt
            
            # Template configuration
            "template_type": template_type,
            
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
            "temperature": 1.0,
            "top_p": 1.0,
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


def segment_text_by_sentences(token_texts, token_logprobs, tokenizer):
    """
    Segment tokens into sentences/punctuated segments and compute metrics for each segment.
    
    Args:
        token_texts: List of decoded token strings
        token_logprobs: List of log probabilities for each token
        tokenizer: HuggingFace tokenizer
        
    Returns:
        List of dictionaries with segment info including text, tokens, metrics
    """
    segments = []
    current_segment = {
        'tokens': [],
        'token_texts': [],
        'logprobs': [],
        'start_idx': 0
    }
    
    # Sentence ending punctuation
    sentence_endings = {'.', '\n'}
    # Additional segment breaks for better granularity
    segment_breaks = set()  # Remove other punctuation marks
    
    for i, (token_text, logprob) in enumerate(zip(token_texts, token_logprobs)):
        current_segment['tokens'].append(i)
        current_segment['token_texts'].append(token_text)
        current_segment['logprobs'].append(logprob.item())
        
        # Check if this token ends a sentence/segment
        stripped_token = token_text.strip()
        should_break = False
        
        # Strong sentence endings
        if any(ending in stripped_token for ending in sentence_endings):
            should_break = True
        # Weaker segment breaks (only if segment is getting long)
        elif len(current_segment['tokens']) > 10 and any(break_char in stripped_token for break_char in segment_breaks):
            should_break = True
        # Force break if segment gets too long
        elif len(current_segment['tokens']) > 30:
            should_break = True
        
        if should_break and len(current_segment['tokens']) > 1:  # Don't create single-token segments
            # Compute metrics for this segment
            segment_logprobs = torch.tensor(current_segment['logprobs'])
            segment_text = ''.join(current_segment['token_texts']).strip()
            
            if len(segment_text) > 0:  # Only add non-empty segments
                segment_info = {
                    'text': segment_text,
                    'tokens': current_segment['tokens'].copy(),
                    'token_texts': current_segment['token_texts'].copy(),
                    'logprobs': current_segment['logprobs'].copy(),
                    'start_idx': current_segment['start_idx'],
                    'end_idx': i,
                    'length': len(current_segment['tokens']),
                    'normalized_logprob': segment_logprobs.mean().item(),
                    'total_logprob': segment_logprobs.sum().item(),
                    'min_logprob': segment_logprobs.min().item(),
                    'max_logprob': segment_logprobs.max().item(),
                    'std_logprob': segment_logprobs.std().item(),
                }
                
                # Compute entropy (uncertainty)
                # For entropy, we need to convert log probabilities to probabilities
                probs = torch.exp(segment_logprobs)
                # Clip to avoid numerical issues
                probs = torch.clamp(probs, min=1e-10, max=1.0)
                entropy = -(probs * torch.log(probs)).sum() / len(probs)
                segment_info['mean_entropy'] = entropy.item()
                
                segments.append(segment_info)
            
            # Start new segment
            current_segment = {
                'tokens': [],
                'token_texts': [],
                'logprobs': [],
                'start_idx': i + 1
            }
    
    # Add final segment if it has content
    if len(current_segment['tokens']) > 0:
        segment_logprobs = torch.tensor(current_segment['logprobs'])
        segment_text = ''.join(current_segment['token_texts']).strip()
        
        if len(segment_text) > 0:
            segment_info = {
                'text': segment_text,
                'tokens': current_segment['tokens'].copy(),
                'token_texts': current_segment['token_texts'].copy(),
                'logprobs': current_segment['logprobs'].copy(),
                'start_idx': current_segment['start_idx'],
                'end_idx': len(token_texts) - 1,
                'length': len(current_segment['tokens']),
                'normalized_logprob': segment_logprobs.mean().item(),
                'total_logprob': segment_logprobs.sum().item(),
                'min_logprob': segment_logprobs.min().item(),
                'max_logprob': segment_logprobs.max().item(),
                'std_logprob': segment_logprobs.std().item(),
            }
            
            probs = torch.exp(segment_logprobs)
            probs = torch.clamp(probs, min=1e-10, max=1.0)
            entropy = -(probs * torch.log(probs)).sum() / len(probs)
            segment_info['mean_entropy'] = entropy.item()
            
            segments.append(segment_info)
    
    return segments


def compute_segment_colors(segments, metric='normalized_logprob'):
    """
    Assign colors to segments based on their metric values using quantiles.
    
    Args:
        segments: List of segment dictionaries
        metric: Which metric to use ('normalized_logprob' or 'mean_entropy')
        
    Returns:
        List of color assignments for each segment
    """
    if not segments:
        return []
    
    # Extract metric values
    values = [seg[metric] for seg in segments]
    values_tensor = torch.tensor(values)
    
    # Compute quantiles for color assignment
    if metric == 'normalized_logprob':
        # For log probabilities: higher = better (green), lower = worse (red)
        q25 = torch.quantile(values_tensor, 0.25)
        q75 = torch.quantile(values_tensor, 0.75)
        
        colors = []
        for value in values:
            if value >= q75:
                colors.append('high')  # Green - high confidence
            elif value <= q25:
                colors.append('low')   # Red - low confidence  
            else:
                colors.append('medium')  # Yellow - medium confidence
                
    else:  # mean_entropy
        # For entropy: lower = better (green), higher = worse (red)
        q25 = torch.quantile(values_tensor, 0.25)
        q75 = torch.quantile(values_tensor, 0.75)
        
        colors = []
        for value in values:
            if value <= q25:
                colors.append('high')  # Green - low entropy (high confidence)
            elif value >= q75:
                colors.append('low')   # Red - high entropy (low confidence)
            else:
                colors.append('medium')  # Yellow - medium entropy
    
    return colors


def plot_and_save_logprobs(logprobs, tokenizer, questions, responses, results_dir="results"):
    """
    Create and save log probability plots for each rollout with text highlighting.
    
    Args:
        logprobs: torch.Tensor of log probabilities (batch_size, sequence_length)
        tokenizer: HuggingFace tokenizer for decoding tokens
        questions: List of input questions
        responses: torch.Tensor of response token IDs
        results_dir: Directory to save plots
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    batch_size = logprobs.shape[0]
    
    for i in range(batch_size):
        # Get log probabilities for this sample (remove padding)
        sample_logprobs = logprobs[i]
        sample_responses = responses[i]
        
        # Remove padding tokens (-1 values in logprobs, pad_token_id in responses)
        non_pad_mask = sample_logprobs != -1
        valid_logprobs = sample_logprobs[non_pad_mask]
        valid_responses = sample_responses[non_pad_mask]
        
        if len(valid_logprobs) == 0:
            continue
        
        # Create single panel figure for log probabilities only
        plt.figure(figsize=(14, 6))
        
        # Plot log probabilities over token positions
        plt.plot(range(len(valid_logprobs)), valid_logprobs.cpu().numpy(), 'b-', alpha=0.8, linewidth=1.5)
        plt.title(f'Log Probabilities - Sample {i+1}', fontsize=16, fontweight='bold')
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Log Probability', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_logprob = valid_logprobs.mean().item()
        min_logprob = valid_logprobs.min().item()
        max_logprob = valid_logprobs.max().item()
        std_logprob = valid_logprobs.std().item()
        
        # Add threshold lines
        plt.axhline(y=mean_logprob, color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_logprob:.3f}')
        
        # Add threshold line for "low" log probabilities (mean - 1 std)
        low_threshold = mean_logprob - std_logprob
        plt.axhline(y=low_threshold, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Low threshold: {low_threshold:.3f}')
        plt.legend()
        
        # Add text with statistics
        stats_text = f'Mean: {mean_logprob:.3f}\nStd: {std_logprob:.3f}\nMin: {min_logprob:.3f}\nMax: {max_logprob:.3f}\nTokens: {len(valid_logprobs)}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
        
        # Prepare data for HTML visualization
        token_texts = []
        token_colors = []
        
        for j, (token_id, logprob) in enumerate(zip(valid_responses, valid_logprobs)):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            token_texts.append(token_text)
            
            # Color based on log probability relative to threshold
            if logprob < low_threshold:
                color = 'red'  # Low confidence
            elif logprob < mean_logprob:
                color = 'orange'  # Medium-low confidence
            else:
                color = 'green'  # High confidence
            
            token_colors.append(color)
        
        # Compute sentence-level segments and metrics
        segments = segment_text_by_sentences(token_texts, valid_logprobs, tokenizer)
        
        # Compute colors for both metrics
        logprob_colors = compute_segment_colors(segments, 'normalized_logprob')
        entropy_colors = compute_segment_colors(segments, 'mean_entropy')
        
        # Add segment color information to each segment
        for seg, lp_color, ent_color in zip(segments, logprob_colors, entropy_colors):
            seg['logprob_color'] = lp_color
            seg['entropy_color'] = ent_color
        
        plt.tight_layout()
        
        # Save the plot
        filename = f'logprob_analysis_sample_{i+1}.png'
        filepath = os.path.join(results_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        # Create both token-level and sentence-level HTML visualizations
        html_filename = f'logprob_analysis_sample_{i+1}.html'
        html_filepath = os.path.join(results_dir, html_filename)
        _create_enhanced_html_visualization(html_filepath, token_texts, token_colors, valid_logprobs, 
                                          mean_logprob, low_threshold, segments,
                                          questions[i] if i < len(questions) else "Sample")
        
        print(f"✅ Saved log probability plot: {filepath}")
        print(f"✅ Saved enhanced HTML visualization: {html_filepath}")
    
    print(f"\n📊 All log probability analyses saved to '{results_dir}/' directory")


def _create_enhanced_html_visualization(filepath, token_texts, token_colors, logprobs, mean_logprob, low_threshold, segments, question):
    """Create an enhanced HTML file with both token-level and segment-level highlighting."""
    
    # Create HTML header and styles
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Log Probability Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
            line-height: 1.6;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }}
        .stats {{
            background-color: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 5px solid #007bff;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            text-align: center;
        }}
        .tabs {{
            margin-bottom: 20px;
        }}
        .tab-button {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 5px 5px 0 0;
            margin-right: 5px;
        }}
        .tab-button.active {{
            background-color: white;
            border-bottom: 1px solid white;
        }}
        .tab-content {{
            background-color: white;
            padding: 25px;
            border-radius: 0 8px 8px 8px;
            border: 1px solid #dee2e6;
            min-height: 400px;
        }}
        .text-container {{
            font-family: 'Courier New', monospace;
            line-height: 1.8;
            font-size: 14px;
        }}
        .high-confidence {{
            background-color: #d4edda;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #c3e6cb;
        }}
        .medium-confidence {{
            background-color: #fff3cd;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #ffeaa7;
        }}
        .low-confidence {{
            background-color: #f8d7da;
            padding: 2px 4px;
            border-radius: 3px;
            border: 1px solid #f5c6cb;
        }}
        .segment {{
            display: inline;
            padding: 2px 4px;
            border-radius: 3px;
            margin: 1px;
        }}
        .segment-high {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
        }}
        .segment-medium {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
        }}
        .segment-low {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
        }}
        .tooltip {{
            position: relative;
            cursor: help;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }}
        .legend-item {{
            display: inline-block;
            margin: 5px 15px 5px 0;
            padding: 8px 12px;
            border-radius: 5px;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Enhanced Log Probability Analysis</h1>
        <p><strong>Question:</strong> {question}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <h4>Token Analysis</h4>
            <p><strong>Total Tokens:</strong> {len(token_texts)}</p>
            <p><strong>Mean LogProb:</strong> {mean_logprob:.3f}</p>
        </div>
        <div class="metric-card">
            <h4>Segment Analysis</h4>
            <p><strong>Total Segments:</strong> {len(segments)}</p>
            <p><strong>Avg Segment Length:</strong> {sum(seg['length'] for seg in segments) / len(segments):.1f} tokens</p>
        </div>
        <div class="metric-card">
            <h4>Confidence Distribution</h4>
            <p><strong>High:</strong> {sum(1 for c in token_colors if c == 'green')} tokens</p>
            <p><strong>Medium:</strong> {sum(1 for c in token_colors if c == 'orange')}</p>
            <p><strong>Low:</strong> {sum(1 for c in token_colors if c == 'red')}</p>
        </div>
    </div>
    
    <div class="tabs">
        <button class="tab-button active" onclick="showTab('token-level')">Token-Level Analysis</button>
        <button class="tab-button" onclick="showTab('segment-logprob')">Segment-Level (LogProb)</button>
        <button class="tab-button" onclick="showTab('segment-entropy')">Segment-Level (Entropy)</button>
    </div>
    
    <div id="token-level" class="tab-content">
        <h3>Token-Level Highlighting</h3>
        <div class="legend">
            <span class="legend-item high-confidence">High Confidence (> mean)</span>
            <span class="legend-item medium-confidence">Medium Confidence (< mean)</span>
            <span class="legend-item low-confidence">Low Confidence (< mean - std)</span>
        </div>
        <div class="text-container">
            <p style="margin-bottom: 15px; font-style: italic;">Hover over tokens to see exact log probability values.</p>
"""
    
    # Add token-level highlighting
    for token_text, color, logprob in zip(token_texts, token_colors, logprobs):
        clean_token = token_text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        clean_token = clean_token.replace('\n', '<br>').replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
        
        css_class = {
            'green': 'high-confidence',
            'orange': 'medium-confidence', 
            'red': 'low-confidence'
        }.get(color, 'high-confidence')
        
        html_content += f'<span class="tooltip {css_class}">{clean_token}<span class="tooltiptext">LogProb: {logprob:.3f}</span></span>'
    
    html_content += """
        </div>
    </div>
    
    <div id="segment-logprob" class="tab-content hidden">
        <h3>Segment-Level Analysis: Normalized Log Probability</h3>
        <div class="legend">
            <span class="legend-item segment-high">High Confidence (Top 25%)</span>
            <span class="legend-item segment-medium">Medium Confidence (Middle 50%)</span>
            <span class="legend-item segment-low">Low Confidence (Bottom 25%)</span>
        </div>
"""
    
    # Add segment-level highlighting for log probabilities
    for seg in segments:
        css_class = f"segment-{seg['logprob_color']}"
        html_content += f"""
        <span class="tooltip {css_class}">{seg['text']}<span class="tooltiptext">
            Normalized LogProb: {seg['normalized_logprob']:.3f}<br>
            Length: {seg['length']} tokens<br>
            Total LogProb: {seg['total_logprob']:.3f}<br>
            Std Dev: {seg['std_logprob']:.3f}
        </span></span>"""
    
    html_content += """
    </div>
    
    <div id="segment-entropy" class="tab-content hidden">
        <h3>Segment-Level Analysis: Mean Token Entropy</h3>
        <div class="legend">
            <span class="legend-item segment-high">Low Entropy (High Certainty)</span>
            <span class="legend-item segment-medium">Medium Entropy</span>
            <span class="legend-item segment-low">High Entropy (High Uncertainty)</span>
        </div>
"""
    
    # Add segment-level highlighting for entropy
    for seg in segments:
        css_class = f"segment-{seg['entropy_color']}"
        html_content += f"""
        <span class="tooltip {css_class}">{seg['text']}<span class="tooltiptext">
            Mean Entropy: {seg['mean_entropy']:.3f}<br>
            Length: {seg['length']} tokens<br>
            Normalized LogProb: {seg['normalized_logprob']:.3f}<br>
            Std Dev: {seg['std_logprob']:.3f}
        </span></span>"""
    
    html_content += """
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.add('hidden'));
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.remove('hidden');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)


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
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    template_type = "tool"  # Change this to "tool_interleaved" for interleaved reasoning
    print(f"Loading model: {model_path}")
    print(f"Using template type: {template_type}")
    
    # Load tokenizer for prompt preparation
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create configuration for ActorRolloutRefWorker
    config = setup_actor_rollout_config(model_path, template_type=template_type)
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
        "Who was president of the United States in the year that Citibank was founded?",
        "Which cities hosted the Olympics in 1988, and where were the opening ceremonies held in each city?",
        "Which actor in the movie Nadja has a Golden Palm Star on the Walk of Stars in Palm Springs, California?",
        "How many years old was The Real Housewives of New York City franchise when Jenna Lyons premiered on the show?"
        # "Explain the concept of machine learning in simple terms.",
        # "Write a short poem about the ocean.",
    ]
    
    print(f"\nPreparing {len(questions)} prompts...")
    prompts = create_prompts_dataproto(
        tokenizer, 
        questions, 
        config.rollout.prompt_length,
        template_type=config.rollout.template_type
    )
    
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
        print(f"Decoded (first 50 chars): {tokenizer.decode(actual_tokens)}...")
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
        
        output = worker.generate_sequences(prompts)
    
        logprobs = output.batch['rollout_log_probs']
        print("✅ Generation completed!")
        
        # Show results
        print(f"\nResults:")
        print(f"- Batch size: {len(output)}")
        print(f"- Response tensor shape: {output.batch['responses'].shape}")
        
        # Create and save log probability plots
        print(f"\n📊 Creating log probability plots...")
        plot_and_save_logprobs(
            logprobs=logprobs,
            tokenizer=tokenizer,
            questions=questions,
            responses=output.batch['responses'],
            results_dir="results"
        )
        
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