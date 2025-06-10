import os
import rich
import pyrallis
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTConfig, SFTTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import time
import numpy as np
from utils import formatting_prompts_func, formatting_prompts_func_cot, compute_ttft, compute_ttft_from_tokens, formatting_func_general, formatting_func_kk, formatting_func_musique
from utils import parse_cot_eval, save_evaluation_results
from data_utils import DATASET_CONFIGS, load_single_dataset, THINK_ANSWER_TEMPLATE, make_conversation
from functools import partial

@dataclass
class ExperimentConfig:
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ["kk"])
    train_subset_size: Optional[int] = 100  
    test_subset_size: Optional[int] = None
    kk_subset_size: Optional[int] = 100 
    musique_subset_size: Optional[int] = None 
    
    # MuSiQue specific settings
    musique_max_context_length: int = 300
    musique_max_paragraphs: int = 3
    
    # Evaluation configuration
    run_post_training_eval: bool = True  # Whether to run detailed evaluation after training
    formatting_mode: str = "func"  # "func" or "cot" for different reasoning styles
    
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ])
    
    output_dir: str = None  
    learning_rate: float = 5e-5
    train_batch_size: int = 4
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 100
    bf16: bool = True
    fp16: bool = False
    
    max_seq_length: int = 256
    remove_unused_columns: bool = False
    
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    eval_strategy: str = "steps"
    eval_steps: int = 5
    push_to_hub: bool = False
    
    seed: int = 42
    hf_cache_dir: str = "/scr/aliang80/hf_cache"
    
    def __post_init__(self):
        # Generate output directory name based on key parameters
        model_name = self.model_id.split("/")[-1]  
        datasets_str = "_".join(sorted(self.datasets))  
        lr_str = f"{self.learning_rate:.0e}".replace("-", "")  
        lora_str = f"r{self.lora_r}_a{self.lora_alpha}" if self.use_lora else "no_lora"
        
        # Add subset information if using subsets
        subset_info = ""
        if self.train_subset_size is not None:
            subset_info = f"_sub{self.train_subset_size}"
        
        self.output_dir = (
            f"{model_name}_SFT_"
            f"{datasets_str}_"
            f"{self.formatting_mode}_"
            f"lr{lr_str}_"
            f"ep{self.num_train_epochs}_"
            f"bs{self.train_batch_size}_"
            f"{lora_str}"
            f"{subset_info}"
        )
        
        print(f"Generated output directory: {self.output_dir}")

def create_compute_metrics(config: ExperimentConfig, test_dataset: Dataset, tokenizer):
    """Create a compute_metrics function for SFTTrainer evaluation."""
    
    def compute_metrics(eval_preds):
        prediction_logits, labels = eval_preds

        import ipdb; ipdb.set_trace()
        predicted_token_ids = np.argmax(prediction_logits, axis=-1)
        decoded_preds = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
        
        # Clear references to original tensors
        del prediction_logits, labels
        
        correct_count = 0
        total_count = 0
        correct_ratios = []
        ttft_scores = []
        detailed_results = []
        
        max_detailed_samples = min(10, len(decoded_preds), len(test_dataset))
        for i, pred in enumerate(decoded_preds):
            if i >= len(test_dataset):
                break
                
            sample = test_dataset[i]
            dataset_source = sample["dataset_source"]
            dataset_config = DATASET_CONFIGS[dataset_source]
            ground_truth = sample[dataset_config["answer_field"]]
    
            # import ipdb; ipdb.set_trace()
            # Use existing evaluation logic
            is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions = parse_cot_eval(pred, ground_truth)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            correct_ratios.append(correct_ratio)
            
            # Calculate TTFT only for detailed samples to save compute
            if i < max_detailed_samples:
                ttft = compute_ttft(pred, tokenizer)
                ttft_scores.append(ttft)
                
                question = sample[dataset_config["question_field"]]
                detailed_results.append({
                    'sample_id': i,
                    'dataset_source': dataset_source,
                    'question': question,  # Truncate long questions
                    'ground_truth': ground_truth,
                    'prediction': pred,  # Truncate long predictions
                    'is_correct': is_correct,
                    'wrong_reason': wrong_reason,
                    'correct_ratio': correct_ratio,
                    'ttft': ttft,
                })
        
        # Calculate metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_correct_ratio = sum(correct_ratios) / len(correct_ratios) if correct_ratios else 0.0
        avg_ttft = sum(ttft_scores) / len(ttft_scores) if ttft_scores else 0.0
        
        # Save detailed results only occasionally to reduce I/O
        if detailed_results and total_count % 20 == 0:  # Save every 20 evaluations
            save_evaluation_results(config, detailed_results, accuracy, avg_correct_ratio, avg_ttft, total_count)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "eval_accuracy": accuracy,
            "eval_avg_correct_ratio": avg_correct_ratio,
            "eval_avg_ttft": avg_ttft,
            "eval_correct_count": correct_count,
            "eval_total_count": total_count
        }
    
    return compute_metrics

@pyrallis.wrap()
def train(config: ExperimentConfig):
    rich.print(config)

    torch.manual_seed(config.seed)
    
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets_to_combine = []
    dataset_info = []

    print(f"Loading datasets: {config.datasets}")

    # Load all specified datasets using general approach
    train_datasets_dict = {}
    test_datasets_dict = {}
    
    for dataset_key in config.datasets:
        train_data, test_data, info = load_single_dataset(dataset_key, config)
        if train_data is not None and test_data is not None:
            train_datasets_dict[dataset_key] = train_data
            test_datasets_dict[dataset_key] = test_data
            dataset_info.append(info)

    if not train_datasets_dict and not test_datasets_dict:
        raise ValueError(f"No datasets were successfully loaded from: {config.datasets}")

    print("="*80)
    print("DATASET METADATA AND STRUCTURE")
    print("="*80)
    
    for info in dataset_info:
        print(info)

    # Create combined training dataset
    if train_datasets_dict:
        train_dataset = concatenate_datasets(list(train_datasets_dict.values()))
        print(f"\nCombined train dataset size: {len(train_dataset)}")
        print(f"Training datasets: {list(train_datasets_dict.keys())}")
    else:
        raise ValueError("No training datasets found!")
    
    # Keep test datasets separate but also create a combined version for the trainer
    test_dataset = concatenate_datasets(list(test_datasets_dict.values()))
    print(f"Combined test dataset size: {len(test_dataset)}")
    print(f"Test datasets: {list(test_datasets_dict.keys())}")
    
    # Print individual dataset sizes
    print(f"\nIndividual dataset sizes:")
    for dataset_key, dataset in train_datasets_dict.items():
        print(f"  {dataset_key} train: {len(dataset)} samples")
    for dataset_key, dataset in test_datasets_dict.items():
        print(f"  {dataset_key} test: {len(dataset)} samples")
    
    # Apply global subset if specified
    if config.train_subset_size is not None:
        train_dataset = train_dataset.select(range(min(config.train_subset_size, len(train_dataset))))
        print(f"Applied global train subset: {len(train_dataset)} samples")
    
    if config.test_subset_size is not None and test_dataset is not None:
        test_dataset = test_dataset.select(range(min(config.test_subset_size, len(test_dataset))))
        print(f"Applied global test subset: {len(test_dataset)} samples")

    print("\n" + "-"*40)
    print("COMBINED DATASET STRUCTURE")
    print("-"*40)
    print(f"Train columns: {train_dataset.column_names} \n\n")
    print(f"Train features: {train_dataset.features} \n\n")
    
    if len(train_dataset) > 0:
        print("\nSample from combined train dataset:")
        sample = train_dataset[0]
        for col_name, col_value in sample.items():
            print(f"\n[{col_name}]:")
            if isinstance(col_value, str):
                preview = col_value[:200] + "..." if len(col_value) > 200 else col_value
                print(f"  Type: {type(col_value).__name__}")
                print(f"  Length: {len(col_value)}")
                print(f"  Content: {preview}")
            else:
                print(f"  Type: {type(col_value).__name__}")
                print(f"  Content: {col_value}")

    # Show formatted training examples from each dataset type
    print("\n" + "="*80)
    print("FORMATTED TRAINING EXAMPLES BY DATASET")
    print("="*80)
    
    for dataset_source, train_data in train_datasets_dict.items():
        if len(train_data) > 0:
            sample = train_data[0]
            print(f"\n{dataset_source.upper()} Dataset - Formatted Example:")
            print("-" * 60)
            formatted_example = formatting_func_general(sample, mode=config.formatting_mode)
            print(formatted_example)
            print("-" * 60)

    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        load_in_4bit=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Use left padding for decoder-only models

    if config.use_lora:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        bf16=config.bf16,
        fp16=config.fp16,
        max_seq_length=config.max_seq_length,
        remove_unused_columns=config.remove_unused_columns,
        report_to=config.report_to,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        push_to_hub=config.push_to_hub,
        # predict_with_generate=True,
        # generation_max_length=500,
        # generation_num_beams=1,
        # generation_do_sample=True,
        # generation_temperature=0.7,
    )

    # Create compute_metrics function for evaluation
    compute_metrics_fn = create_compute_metrics(config, test_dataset, tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        formatting_func=lambda example: formatting_func_general(example, mode=config.formatting_mode),
        peft_config=peft_config if config.use_lora else None,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Run detailed post-training evaluation if enabled
    if config.run_post_training_eval:
        print("\n" + "="*60)
        print("RUNNING DETAILED POST-TRAINING EVALUATION")
        print("="*60)
        eval(config, test_dataset)

if __name__ == "__main__":
    train() 