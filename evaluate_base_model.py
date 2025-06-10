#!/usr/bin/env python3
"""
Evaluate base model performance on all datasets before training.
Outputs results to wandb and files with detailed per-dataset analysis.
"""

import os
import sys
import json
import torch
import wandb
import rich
import pyrallis
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from torch.utils.data import Dataset, DataLoader

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils import DATASET_CONFIGS, load_single_dataset, make_conversation
from rewards import (
    accuracy_reward, format_reward, 
    kk_exact_reward, kk_partial_reward,
    parse_kk_assignments
)

class EvaluationDataset(Dataset):    
    def __init__(self, dataset, dataset_key, tokenizer):
        self.dataset = dataset
        self.dataset_key = dataset_key
        self.tokenizer = tokenizer
        self.dataset_config = DATASET_CONFIGS[dataset_key]
        self.question_field = self.dataset_config["question_field"]
        self.answer_field = self.dataset_config["answer_field"]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        question = example[self.question_field]
        ground_truth = example[self.answer_field]
        
        # Create conversation format
        conversation_example = make_conversation(example)
        
        # Format as chat prompt
        prompt = self.tokenizer.apply_chat_template(
            conversation_example["prompt"], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return {
            "example_id": idx,
            "prompt": prompt,
            "question": question,
            "ground_truth": ground_truth,
            "dataset": self.dataset_key
        }

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    prompts = [item["prompt"] for item in batch]
    questions = [item["question"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    example_ids = [item["example_id"] for item in batch]
    datasets = [item["dataset"] for item in batch]
    
    return {
        "prompts": prompts,
        "questions": questions,
        "ground_truths": ground_truths,
        "example_ids": example_ids,
        "datasets": datasets
    }

@dataclass
class EvaluationConfig:
    """Configuration for base model evaluation"""
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Datasets to evaluate
    # datasets: List[str] = field(default_factory=lambda: ["kk", "math500", "gpqa"])
    datasets: List[str] = field(default_factory=lambda: ["gpqa"])
    
    # Evaluation subset sizes
    kk_subset_size: Optional[int] = 50
    math500_subset_size: Optional[int] = 50
    gpqa_subset_size: Optional[int] = 50
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    batch_size: int = 8  # Batch size for generation
    
    # DataLoader parameters
    num_workers: int = 0  # Number of workers for data loading
    
    # Environment
    hf_cache_dir: str = "/scr/aliang80/hf_cache"
    output_dir: str = None
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "base-model-evaluation"
    wandb_entity: str = "clvr"
    wandb_tags: List[str] = field(default_factory=lambda: ["base_model", "evaluation"])
    
    def __post_init__(self):
        # Generate output directory name if not specified
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_id.split("/")[-1]
            datasets_str = "_".join(sorted(self.datasets))
            self.output_dir = f"base_eval_{model_name}_{datasets_str}_{timestamp}"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        rich.print(f"[blue]Output directory:[/blue] {self.output_dir}")

def generate_batch_responses(model, tokenizer, prompts, config):
    """Generate responses for a batch of prompts."""
    # Tokenize all prompts in the batch
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048  # Set a reasonable max length
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate responses for the batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.do_sample,
            top_p=config.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode responses (remove input prompts)
    input_lengths = inputs['input_ids'].shape[1]
    responses = []
    
    for i in range(len(prompts)):
        generated_tokens = outputs[i][input_lengths:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses

def evaluate_dataset(model, tokenizer, dataset_key, dataset, config):
    """Evaluate model on a single dataset using DataLoader for batching."""
    rich.print(f"\n[bold cyan]Evaluating {dataset_key.upper()} dataset...[/bold cyan]")
    
    # Create PyTorch dataset and dataloader
    eval_dataset = EvaluationDataset(dataset, dataset_key, tokenizer)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=collate_fn
    )
    
    rich.print(f"[blue]Processing {len(dataset)} examples in {len(dataloader)} batches of size {config.batch_size}[/blue]")
    
    # Print one example from the dataset
    if len(eval_dataset) > 0:
        sample = eval_dataset[0]
        rich.print(f"\n[yellow]Sample from {dataset_key.upper()} dataset:[/yellow]")
        rich.print(f"[cyan]Question:[/cyan] {sample['question']}")
        rich.print(f"[cyan]Ground Truth:[/cyan] {sample['ground_truth']}")
        rich.print(f"[cyan]Formatted Prompt:[/cyan] {sample['prompt']}")
        rich.print()
    
    results = []
    all_predictions = []
    all_ground_truths = []
    
    # Process batches with rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task(f"Processing {dataset_key}", total=len(dataloader))
        
        for batch in dataloader:
            # Generate responses for the batch
            batch_predictions = generate_batch_responses(model, tokenizer, batch["prompts"], config)
            
            # Store results for this batch
            for i, (example_id, question, ground_truth, prediction) in enumerate(
                zip(batch["example_ids"], batch["questions"], batch["ground_truths"], batch_predictions)
            ):
                all_predictions.append(prediction)
                all_ground_truths.append(ground_truth)
                
                # Store detailed result
                result = {
                    "example_id": example_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "dataset": dataset_key
                }
                results.append(result)
            
            progress.update(task, advance=1)
    
    # Calculate rewards/metrics
    completions = [[[{"content": pred}]] for pred in all_predictions]
    completions = [comp[0] for comp in completions]  
    kwargs = {"solution": all_ground_truths}
    
    # Apply appropriate reward functions based on dataset
    metrics = {}
    
    if dataset_key == "kk":
        # Knights and Knaves specific rewards
        exact_rewards = kk_exact_reward(completions, **kwargs)
        partial_rewards = kk_partial_reward(completions, **kwargs)
        format_rewards = format_reward(completions, **kwargs)
        
        metrics = {
            "exact_accuracy": sum(exact_rewards) / len(exact_rewards),
            "partial_accuracy": sum(partial_rewards) / len(partial_rewards), 
            "format_accuracy": sum(format_rewards) / len(format_rewards),
        }
        
        # Add individual scores to results
        for i, (exact, partial, fmt) in enumerate(zip(exact_rewards, partial_rewards, format_rewards)):
            results[i].update({
                "exact_reward": exact,
                "partial_reward": partial,
                "format_reward": fmt
            })
    
    else:
        # Math/Science datasets - use general accuracy
        accuracy_rewards = accuracy_reward(completions, **kwargs)
        format_rewards = format_reward(completions, **kwargs)
        
        metrics = {
            "accuracy": sum(accuracy_rewards) / len(accuracy_rewards),
            "format_accuracy": sum(format_rewards) / len(format_rewards),
        }
        
        # Add individual scores to results
        for i, (acc, fmt) in enumerate(zip(accuracy_rewards, format_rewards)):
            results[i].update({
                "accuracy_reward": acc,
                "format_reward": fmt
            })
    
    return results, metrics

def save_results(results_by_dataset, overall_metrics, config):
    """Save detailed results to files."""
    
    # Save detailed results
    detailed_file = os.path.join(config.output_dir, "detailed_results.json")
    with open(detailed_file, 'w') as f:
        json.dump(results_by_dataset, f, indent=2)
    
    # Save summary metrics
    summary_file = os.path.join(config.output_dir, "summary_metrics.json")
    with open(summary_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    # Save human-readable report
    report_file = os.path.join(config.output_dir, "evaluation_report.txt")
    with open(report_file, 'w') as f:
        f.write("BASE MODEL EVALUATION REPORT\n")        
        f.write(f"Model: {config.model_id}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datasets: {', '.join(config.datasets)}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in overall_metrics.items():
            if isinstance(value, dict):
                f.write(f"{metric}:\n")
                for sub_metric, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        f.write(f"  {sub_metric}:\n")
                        for ssm, ssv in sub_value.items():
                            f.write(f"    {ssm}: {ssv:.3f}\n")
                    else:
                        f.write(f"  {sub_metric}: {sub_value:.3f}\n")
            else:
                f.write(f"{metric}: {value:.3f}\n")
        f.write("\n")
        
        f.write("PER-DATASET BREAKDOWN:\n")
        f.write("-" * 30 + "\n")
        for dataset_key, results in results_by_dataset.items():
            f.write(f"\n{dataset_key.upper()}:\n")
            f.write(f"  Examples: {len(results)}\n")
            
            # Calculate and display metrics for this dataset
            if dataset_key == "kk":
                exact_scores = [r.get("exact_reward", 0) for r in results]
                partial_scores = [r.get("partial_reward", 0) for r in results]
                format_scores = [r.get("format_reward", 0) for r in results]
                
                f.write(f"  Exact Accuracy: {sum(exact_scores)/len(exact_scores):.3f}\n")
                f.write(f"  Partial Accuracy: {sum(partial_scores)/len(partial_scores):.3f}\n") 
                f.write(f"  Format Accuracy: {sum(format_scores)/len(format_scores):.3f}\n")
            else:
                acc_scores = [r.get("accuracy_reward", 0) for r in results]
                format_scores = [r.get("format_reward", 0) for r in results]
                
                f.write(f"  Accuracy: {sum(acc_scores)/len(acc_scores):.3f}\n")
                f.write(f"  Format Accuracy: {sum(format_scores)/len(format_scores):.3f}\n")
    
    rich.print(f"\n[green]Results saved to:[/green] {config.output_dir}")
    rich.print(f"  • Detailed results: [blue]{detailed_file}[/blue]")
    rich.print(f"  • Summary metrics: [blue]{summary_file}[/blue]")
    rich.print(f"  • Human report: [blue]{report_file}[/blue]")

@pyrallis.wrap()
def evaluate(config: EvaluationConfig):
    """Main evaluation function with pyrallis configuration management."""
    rich.print(config)
    
    rich.print("[bold cyan]BASE MODEL EVALUATION[/bold cyan]")
    rich.print(f"[blue]Model:[/blue] {config.model_id}")
    rich.print(f"[blue]Datasets:[/blue] {', '.join(config.datasets)}")
    
    # Initialize wandb
    if config.use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=f"base_eval_{config.model_id.split('/')[-1]}",
            config=config.__dict__,
            tags=config.wandb_tags
        )
    
    # Load model and tokenizer
    rich.print(f"\n[yellow]Loading model:[/yellow] {config.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        load_in_4bit=True,
        device_map="auto",
        cache_dir=config.hf_cache_dir
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, cache_dir=config.hf_cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rich.print("[green]✓[/green] Model and tokenizer loaded successfully")
    
    # Evaluate each dataset
    results_by_dataset = {}
    metrics_by_dataset = {}
    
    for dataset_key in config.datasets:
        # Load dataset
        train_data, test_data, info = load_single_dataset(dataset_key, config)
        
        # Use test data, fallback to train if no test
        eval_data = test_data if test_data is not None else train_data
        if eval_data is None:
            rich.print(f"[red]No data available for {dataset_key}, skipping...[/red]")
            continue
        
        rich.print(f"\n[blue]{info}[/blue]")
        
        # Evaluate
        results, metrics = evaluate_dataset(model, tokenizer, dataset_key, eval_data, config)
        
        results_by_dataset[dataset_key] = results
        metrics_by_dataset[dataset_key] = metrics
        
        # Log to wandb
        if config.use_wandb:
            for metric_name, metric_value in metrics.items():
                wandb.log({f"{dataset_key}/{metric_name}": metric_value})
        
        # Display results with rich formatting
        rich.print(f"\n[bold green]{dataset_key.upper()} Results:[/bold green]")
        for metric_name, metric_value in metrics.items():
            rich.print(f"  [cyan]{metric_name}:[/cyan] {metric_value:.3f}")
    
    # Calculate overall metrics
    all_format_scores = []
    all_accuracy_scores = []
    
    for dataset_key, results in results_by_dataset.items():
        format_scores = [r.get("format_reward", 0) for r in results]
        all_format_scores.extend(format_scores)
        
        if dataset_key == "kk":
            # Use exact accuracy for overall KK score
            acc_scores = [r.get("exact_reward", 0) for r in results]
        else:
            acc_scores = [r.get("accuracy_reward", 0) for r in results]
        all_accuracy_scores.extend(acc_scores)
    
    overall_metrics = {
        "overall_accuracy": sum(all_accuracy_scores) / len(all_accuracy_scores) if all_accuracy_scores else 0,
        "overall_format_accuracy": sum(all_format_scores) / len(all_format_scores) if all_format_scores else 0,
        "per_dataset_metrics": metrics_by_dataset,
        "total_examples": sum(len(results) for results in results_by_dataset.values())
    }
    
    # Log overall metrics to wandb
    if config.use_wandb:
        wandb.log({
            "overall/accuracy": overall_metrics["overall_accuracy"],
            "overall/format_accuracy": overall_metrics["overall_format_accuracy"],
            "overall/total_examples": overall_metrics["total_examples"]
        })
    
    # Save results
    save_results(results_by_dataset, overall_metrics, config)
    
    # Print final summary with rich formatting
    rich.print(f"\n[bold]=" * 60)
    rich.print("[bold green]EVALUATION COMPLETE[/bold green]")
    rich.print(f"[bold]=" * 60)
    rich.print(f"[green]Overall Accuracy:[/green] {overall_metrics['overall_accuracy']:.3f}")
    rich.print(f"[green]Overall Format Accuracy:[/green] {overall_metrics['overall_format_accuracy']:.3f}")
    rich.print(f"[green]Total Examples:[/green] {overall_metrics['total_examples']}")
    
    if config.use_wandb:
        run.finish()
        rich.print("[green]✓[/green] Wandb run finished")

if __name__ == "__main__":
    evaluate() 