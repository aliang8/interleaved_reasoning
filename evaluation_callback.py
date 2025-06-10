"""
Custom evaluation callback for RL training that measures pass@1 accuracy and TTFT.
"""

import wandb
import torch
import time
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from utils import parse_cot_eval, compute_ttft
from data_utils import DATASET_CONFIGS
from datasets import concatenate_datasets


def evaluate_completions(prompts, generated_completions, ground_truths, dataset_sources, tokenizer):
    """
    Evaluates generated completions for accuracy and TTFT.
    
    Args:
        prompts: List of input prompts
        generated_completions: List of generated text completions
        ground_truths: List of ground truth answers
        dataset_sources: List of dataset sources (e.g., "kk", "musique")
        tokenizer: Tokenizer for TTFT computation
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "eval/pass_at_1": 0.0,
        "eval/avg_correct_ratio": 0.0,
        "eval/avg_ttft": 0.0,
        "eval/total_samples": len(generated_completions)
    }

    correct_count = 0
    total_correct_ratio = 0
    total_ttft = 0
    valid_ttft_count = 0

    # For logging sample table to W&B
    eval_table_data = []
    
    for i, (prompt, completion, ground_truth, dataset_source) in enumerate(
        zip(prompts, generated_completions, ground_truths, dataset_sources)
    ):
        # Use existing parse_cot_eval function for accuracy assessment
        is_correct, pred_answer, wrong_reason, correct_ratio, _ = parse_cot_eval(
            completion, ground_truth
        )
        
        ttft = compute_ttft(completion, tokenizer, prompt_text=prompt)
        total_ttft += ttft
        valid_ttft_count += 1

        # Update counters
        if is_correct:
            correct_count += 1
        total_correct_ratio += correct_ratio
        
        # Add to evaluation table (truncate long texts for readability)
        eval_table_data.append([
            prompt[:200] + "..." if len(prompt) > 200 else prompt,
            completion[:300] + "..." if len(completion) > 300 else completion,
            ground_truth[:100] + "..." if len(ground_truth) > 100 else ground_truth,
            dataset_source,
            is_correct,
            f"{correct_ratio:.3f}",
            f"{ttft:.4f}" if ttft is not None else "N/A",
            wrong_reason if not is_correct else "N/A"
        ])

    # Calculate final metrics
    if metrics["eval/total_samples"] > 0:
        metrics["eval/pass_at_1"] = correct_count / metrics["eval/total_samples"]
        metrics["eval/avg_correct_ratio"] = total_correct_ratio / metrics["eval/total_samples"]
    
    if valid_ttft_count > 0:
        metrics["eval/avg_ttft"] = total_ttft / valid_ttft_count
    
    metrics["eval/correct_count"] = correct_count
    metrics["eval/valid_ttft_samples"] = valid_ttft_count

    return metrics, eval_table_data


def extract_prompt_from_sample(sample):
    """Extract prompt text from a dataset sample."""
    if 'prompt' in sample and isinstance(sample['prompt'], list):
        # Convert conversation format to text
        conversation_text = ""
        for message in sample['prompt']:
            conversation_text += f"{message['content']}\n"
        return conversation_text.strip()
    else:
        # Fallback to raw question
        dataset_source = sample['dataset_source']
        dataset_config = DATASET_CONFIGS[dataset_source]
        question = sample[dataset_config['question_field']]
        return f"### Question: {question}\n### Answer:"


def collate_eval_batch(batch):
    """Custom collate function for evaluation batches."""
    prompts = []
    ground_truths = []
    dataset_sources = []
    
    for sample in batch:
        # Extract prompt
        prompt = extract_prompt_from_sample(sample)
        prompts.append(prompt)
        
        # Extract ground truth
        dataset_source = sample['dataset_source']
        dataset_config = DATASET_CONFIGS[dataset_source]
        ground_truth = sample[dataset_config['answer_field']]
        
        ground_truths.append(ground_truth)
        dataset_sources.append(dataset_source)
    
    return {
        'prompts': prompts,
        'ground_truths': ground_truths,
        'dataset_sources': dataset_sources
    }


class CustomEvaluationCallback(TrainerCallback):
    """
    Custom evaluation callback that measures pass@1 and TTFT on eval dataset.
    """
    
    def __init__(self, eval_dataset=None, datasets_dict=None, max_eval_samples=50, batch_size=4):
        self.eval_dataset = eval_dataset 
        self.datasets_dict = datasets_dict or {}  # Dictionary of separate datasets
        self.max_eval_samples = max_eval_samples
        self.batch_size = batch_size

    def evaluate_per_dataset(self, processing_class, model, args, state):
        """Evaluate each dataset separately and return per-dataset metrics."""
        per_dataset_metrics = {}
        all_eval_table_data = []  # Collect data from all datasets
        
        # Set padding side to left for decoder-only models
        original_padding_side = processing_class.padding_side
        processing_class.padding_side = "left"
        
        for dataset_name, dataset in self.datasets_dict.items():
            print(f"\n{'='*50}")
            print(f"EVALUATING {dataset_name.upper()} DATASET")
            print(f"{'='*50}")
            
            # Sample from this specific dataset
            dataset_samples = dataset.select(range(min(10, len(dataset))))  # Smaller sample per dataset
            
            # Create DataLoader for this dataset
            eval_dataloader = DataLoader(
                dataset_samples, 
                batch_size=min(self.batch_size, len(dataset_samples)), 
                shuffle=False, 
                collate_fn=collate_eval_batch
            )
            
            # Generate and evaluate for this dataset
            all_prompts = []
            all_generated_completions = []
            all_ground_truths = []
            all_dataset_sources = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_dataloader):
                    batch_prompts = batch['prompts']
                    batch_ground_truths = batch['ground_truths']
                    batch_dataset_sources = batch['dataset_sources']
                    
                    # Generate completions
                    inputs = processing_class(
                        batch_prompts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=args.max_prompt_length
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_completion_length,
                        do_sample=True,
                        temperature=args.temperature,
                        pad_token_id=processing_class.pad_token_id,
                        eos_token_id=processing_class.eos_token_id,
                    )
                    
                    # Decode completions
                    batch_completions = []
                    for i in range(len(batch_prompts)):
                        input_length = inputs['input_ids'][i].shape[0]
                        generated_ids = outputs[i][input_length:]
                        completion = processing_class.decode(generated_ids, skip_special_tokens=True)
                        batch_completions.append(completion)
                    
                    # Store results
                    all_prompts.extend(batch_prompts)
                    all_generated_completions.extend(batch_completions)
                    all_ground_truths.extend(batch_ground_truths)
                    all_dataset_sources.extend(batch_dataset_sources)
            
            # Evaluate this dataset
            dataset_metrics, eval_table_data = evaluate_completions(
                all_prompts, all_generated_completions, all_ground_truths, all_dataset_sources, processing_class
            )
            
            # Add dataset name to eval table data and collect
            for row in eval_table_data[:5]:  # Limit to 5 samples per dataset
                all_eval_table_data.append(row)
            
            # Store with dataset prefix
            for key, value in dataset_metrics.items():
                if key.startswith("eval/"):
                    new_key = key.replace("eval/", f"eval/{dataset_name}_")
                    per_dataset_metrics[new_key] = value
            
            print(f"{dataset_name.upper()} Results:")
            print(f"  Pass@1: {dataset_metrics['eval/pass_at_1']:.3f}")
            print(f"  Avg Correct Ratio: {dataset_metrics['eval/avg_correct_ratio']:.3f}")
            print(f"  Avg TTFT: {dataset_metrics['eval/avg_ttft']:.4f}")
        
        # Restore original padding side
        processing_class.padding_side = original_padding_side
        
        return per_dataset_metrics, all_eval_table_data

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """Called during evaluation to compute custom metrics."""

        if model is None or (tokenizer is None and "processing_class" not in kwargs):
            print("Warning: Model or tokenizer not available for custom evaluation")
            return
        
        processing_class = kwargs["processing_class"]
        
        # Set padding side to left for decoder-only models
        original_padding_side = processing_class.padding_side
        processing_class.padding_side = "left"
            
        print(f"\n{'='*60}")
        print(f"RUNNING CUSTOM EVALUATION AT STEP {state.global_step}")
        print(f"{'='*60}")
        
        # Only run per-dataset evaluation if we have separate datasets
        if not self.datasets_dict:
            print("Warning: No datasets_dict provided, skipping evaluation")
            return
            
        # Run per-dataset evaluation
        per_dataset_metrics, eval_table_data = self.evaluate_per_dataset(processing_class, model, args, state)
        
        # Calculate overall metrics by aggregating across datasets
        overall_metrics = {}
        total_correct = 0
        total_samples = 0
        total_correct_ratio = 0
        total_ttft = 0
        dataset_count = 0
        
        for dataset_name in self.datasets_dict.keys():
            dataset_prefix = f"eval/{dataset_name}_"
            if f"{dataset_prefix}correct_count" in per_dataset_metrics:
                total_correct += per_dataset_metrics[f"{dataset_prefix}correct_count"]
                total_samples += per_dataset_metrics[f"{dataset_prefix}total_samples"]
                total_correct_ratio += per_dataset_metrics[f"{dataset_prefix}avg_correct_ratio"]
                total_ttft += per_dataset_metrics[f"{dataset_prefix}avg_ttft"]
                dataset_count += 1
        
        # Calculate aggregated metrics
        if total_samples > 0:
            overall_metrics["eval/pass_at_1"] = total_correct / total_samples
            overall_metrics["eval/correct_count"] = total_correct
            overall_metrics["eval/total_samples"] = total_samples
        
        if dataset_count > 0:
            overall_metrics["eval/avg_correct_ratio"] = total_correct_ratio / dataset_count
            overall_metrics["eval/avg_ttft"] = total_ttft / dataset_count
        
        overall_metrics["eval/global_step"] = state.global_step
        overall_metrics["eval/batch_size"] = self.batch_size
        
        # Combine with per-dataset metrics
        all_metrics = {**overall_metrics, **per_dataset_metrics}
        
        # Log to wandb
        if wandb.run:
            wandb.log(all_metrics)
            
            # Log sample table with predictions from all datasets
            if eval_table_data:
                eval_table_columns = [
                    "Prompt", "Generated", "Ground Truth", "Dataset", 
                    "Correct", "Correct Ratio", "TTFT", "Error Reason"
                ]
                wandb_table = wandb.Table(columns=eval_table_columns, data=eval_table_data)
                wandb.log({"eval/sample_predictions": wandb_table})
        
        # Print summary
        print(f"\nOverall Evaluation Results (Aggregated):")
        if "eval/pass_at_1" in overall_metrics:
            print(f"  Pass@1: {overall_metrics['eval/pass_at_1']:.3f}")
            print(f"  Avg Correct Ratio: {overall_metrics['eval/avg_correct_ratio']:.3f}")
            print(f"  Avg TTFT: {overall_metrics['eval/avg_ttft']:.4f}")
            print(f"  Correct: {overall_metrics['eval/correct_count']}/{overall_metrics['eval/total_samples']}")
        
        # Print per-dataset summary
        print(f"\nPer-Dataset Summary:")
        for dataset_name in self.datasets_dict.keys():
            pass_at_1_key = f"eval/{dataset_name}_pass_at_1"
            if pass_at_1_key in per_dataset_metrics:
                print(f"  {dataset_name.upper()}: {per_dataset_metrics[pass_at_1_key]:.3f} pass@1")
        
        print(f"{'='*60}\n")
        
        model.train()  # Return to training mode
        
        # Restore original padding side
        processing_class.padding_side = original_padding_side 