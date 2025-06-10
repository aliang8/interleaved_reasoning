import os
import rich
import pyrallis
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import GRPOConfig, GRPOTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import time
from utils import formatting_prompts_func, formatting_prompts_func_cot, compute_ttft, compute_ttft_from_tokens, formatting_func_general, formatting_func_kk, formatting_func_musique, formatting_func_math
from data_utils import DATASET_CONFIGS, load_single_dataset, THINK_ANSWER_TEMPLATE, make_conversation
from rewards import format_reward, accuracy_reward, conditional_reward_function, reward_tracker
from evaluation_callback import CustomEvaluationCallback
import wandb

@dataclass
class ExperimentConfig:
    datasets: List[str] = field(default_factory=lambda: ["kk"])  # List of datasets to use: ["kk", "musique"]
    eval_only_datasets: List[str] = field(default_factory=lambda: ["math500", "gpqa"])  # Datasets only for evaluation
    train_subset_size: Optional[int] = 10  
    test_subset_size: Optional[int] = 10
    kk_subset_size: Optional[int] = 100 
    musique_subset_size: Optional[int] = 100 
    math500_subset_size: Optional[int] = 50  # Subset size for MATH-500 evaluation
    gpqa_subset_size: Optional[int] = 50  # Subset size for GPQA evaluation
    
    # MuSiQue specific settings
    musique_max_context_length: int = 300
    musique_max_paragraphs: int = 3
    
    # Evaluation configuration
    formatting_mode: str = "func"  # "func" or "cot" for different reasoning styles
    
    # Model configuration
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj", "lm_head"
    ])
    
    # GRPO Training configuration
    output_dir: str = None
    learning_rate: float = 1e-6
    train_batch_size: int = 4
    validation_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 100
    bf16: bool = True
    fp16: bool = False
    
    # GRPO specific parameters
    kl_coefficient: float = 0.001
    kl_loss_type: str = "low_variance_kl"
    max_prompt_length: int = 256
    max_response_length: int = 256
    max_completion_length: int = 256
    sampling_temperature: float = 0.8
    num_generations: int = 4
    num_samples_per_prompt: int = 4
    stable_training_threshold: float = 0.05
    critic_warmup_steps: int = 0
    evaluation_frequency: int = 2
    tensor_model_parallel_size: int = 2
    remove_unused_columns: bool = False
    
    # Reward functions
    reward_functions: List[str] = field(default_factory=lambda: [format_reward, accuracy_reward])
    intermediate_reward_type: str = "partial_credit"
    stable_training_epsilon: float = 0.05
    base_reward: float = 1.0
    lambda_a: float = 1.0
    
    # Logging and saving
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    logging_steps: int = 1
    save_strategy: str = "steps"
    save_steps: int = 200
    push_to_hub: bool = False
    
    # Environment
    seed: int = 42
    hf_cache_dir: str = "/scr/aliang80/hf_cache"
    
    # Wandb 
    wandb_tags: List[str] = field(default_factory=lambda: ["rl"])
    wandb_notes: str = "RL training"
    
    def __post_init__(self):
        # Generate output directory name based on key parameters
        if self.output_dir is None:
            model_name = self.model_id.split("/")[-1]  
            datasets_str = "_".join(sorted(self.datasets))  
            lr_str = f"{self.learning_rate:.0e}".replace("-", "")  
            lora_str = f"r{self.lora_r}_a{self.lora_alpha}" if self.use_lora else "no_lora"
            
            # Add subset information if using subsets
            subset_info = ""
            if self.train_subset_size is not None:
                subset_info = f"_sub{self.train_subset_size}"
            
            self.output_dir = (
                f"{model_name}_GRPO_"
                f"{datasets_str}_"
                f"{self.formatting_mode}_"
                f"lr{lr_str}_"
                f"ep{self.num_train_epochs}_"
                f"bs{self.train_batch_size}_"
                f"{lora_str}"
                f"{subset_info}"
            )
            
            print(f"Generated output directory: {self.output_dir}")
        
        global reward_tracker
        reward_tracker.epsilon = self.stable_training_epsilon

@pyrallis.wrap()
def train(config: ExperimentConfig):
    rich.print(config)

    torch.manual_seed(config.seed)
    
    # Set up wandb logging
    if "wandb" in config.report_to:
        run = wandb.init(
            project="interleaved-reasoning",
            name=config.output_dir,
            config=config,
            tags=config.wandb_tags,
            notes=config.wandb_notes,
            entity="clvr"
        )

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )

    response_template = "\n### Answer:\n"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_info = []

    print(f"Loading training datasets: {config.datasets}")
    print(f"Loading eval-only datasets: {config.eval_only_datasets}")

    # Load training datasets (both train and test splits)
    train_datasets_dict = {}
    test_datasets_dict = {}
    
    for dataset_key in config.datasets:
        train_data, test_data, info = load_single_dataset(dataset_key, config)
        if train_data is not None and test_data is not None:
            train_datasets_dict[dataset_key] = train_data
            test_datasets_dict[dataset_key] = test_data
            dataset_info.append(info)
        elif test_data is not None:
            # Handle case where dataset only has test split
            test_datasets_dict[dataset_key] = test_data
            dataset_info.append(info)
            print(f"Warning: {dataset_key} has no training split, only added to evaluation")

    # Load eval-only datasets (only test splits)
    for dataset_key in config.eval_only_datasets:
        train_data, test_data, info = load_single_dataset(dataset_key, config)
        if test_data is not None:
            test_datasets_dict[dataset_key] = test_data
            dataset_info.append(info)
            print(f"Added {dataset_key} to evaluation set only")

    # Check that we have training datasets
    if not train_datasets_dict:
        raise ValueError("No training datasets found! Need at least one dataset with training split.")
    
    print("="*80)
    print("DATASET METADATA AND STRUCTURE")
    print("="*80)
    
    for info in dataset_info:
        print(info)
    
    # Print dataset information 
    print(f"\nTraining datasets: {list(train_datasets_dict.keys())}")
    print(f"Test datasets: {list(test_datasets_dict.keys())}")
    
    # Print individual dataset sizes
    print(f"\nIndividual dataset sizes:")
    for dataset_key, dataset in train_datasets_dict.items():
        print(f"  {dataset_key} train: {len(dataset)} samples")
    for dataset_key, dataset in test_datasets_dict.items():
        print(f"  {dataset_key} test: {len(dataset)} samples")
    
    # Apply global subset to individual datasets if specified
    if config.train_subset_size is not None:
        for dataset_key in train_datasets_dict:
            original_size = len(train_datasets_dict[dataset_key])
            subset_size = min(config.train_subset_size, original_size)
            train_datasets_dict[dataset_key] = train_datasets_dict[dataset_key].select(range(subset_size))
            print(f"Applied train subset to {dataset_key}: {subset_size}/{original_size} samples")
    
            if dataset_key == "kk":
                # replace solution_text field with solution field
                train_datasets_dict[dataset_key] = train_datasets_dict[dataset_key].rename_column("solution", "tmp_solution")
                train_datasets_dict[dataset_key] = train_datasets_dict[dataset_key].rename_column("solution_text", "solution")
                train_datasets_dict[dataset_key] = train_datasets_dict[dataset_key].remove_columns("tmp_solution")

    if config.test_subset_size is not None:
        for dataset_key in test_datasets_dict:
            original_size = len(test_datasets_dict[dataset_key])
            subset_size = min(config.test_subset_size, original_size)
            test_datasets_dict[dataset_key] = test_datasets_dict[dataset_key].select(range(subset_size))
            print(f"Applied test subset to {dataset_key}: {subset_size}/{original_size} samples")

            if dataset_key == "kk":
                # replace solution_text field with solution field
                test_datasets_dict[dataset_key] = test_datasets_dict[dataset_key].rename_column("solution", "tmp_solution")
                test_datasets_dict[dataset_key] = test_datasets_dict[dataset_key].rename_column("solution_text", "solution")
                test_datasets_dict[dataset_key] = test_datasets_dict[dataset_key].remove_columns("tmp_solution")


    print("\n" + "-"*40)
    print("DATASET STRUCTURE")
    print("-"*40)
    
    # Show structure from first training dataset
    first_train_key = list(train_datasets_dict.keys())[0]
    first_train_dataset = train_datasets_dict[first_train_key]
    
    print(f"Sample from {first_train_key} train dataset:")
    print(f"Columns: {first_train_dataset.column_names}")
    print(f"Features: {first_train_dataset.features}")
    
    if len(first_train_dataset) > 0:
        print(f"\nFirst sample content:")
        sample = first_train_dataset[0]
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

    # Format datasets for conversation
    print("\nFormatting datasets for conversation...")
    train_datasets_dict = {dataset_key: dataset.map(make_conversation, desc=f"Formatting {dataset_key} train dataset") 
                          for dataset_key, dataset in train_datasets_dict.items()}
    test_datasets_dict = {dataset_key: dataset.map(make_conversation, desc=f"Formatting {dataset_key} test dataset") 
                         for dataset_key, dataset in test_datasets_dict.items()}
    
    # Show sample conversation format
    first_train_key = list(train_datasets_dict.keys())[0]
    first_train_dataset = train_datasets_dict[first_train_key]
    
    print(f"\nFormatted dataset columns: {first_train_dataset.column_names}")
    
    if len(first_train_dataset) > 0:
        print(f"\nSample conversation format from {first_train_key}:")
        sample_prompt = first_train_dataset[0]['prompt']
        for message in sample_prompt:
            print(f"Role: {message['role']}")
            content_preview = message['content'][:150] + "..." if len(message['content']) > 150 else message['content']
            print(f"Content: {content_preview}")
            print("-" * 40)
    
    # Create combined datasets for trainer (which requires Dataset objects, not dicts)
    print("\nCreating combined datasets for trainer...")
    train_dataset_combined = concatenate_datasets(list(train_datasets_dict.values()))
    test_dataset_combined = concatenate_datasets(list(test_datasets_dict.values()))
    
    print(f"Combined train dataset: {len(train_dataset_combined)} samples")
    print(f"Combined test dataset: {len(test_dataset_combined)} samples")

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

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        remove_unused_columns=config.remove_unused_columns,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        bf16=config.bf16,
        fp16=config.fp16,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.validation_batch_size,

        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        temperature=config.sampling_temperature,
        beta=config.kl_coefficient,
        num_iterations=1,

        report_to=config.report_to,
        logging_steps=config.logging_steps,
        push_to_hub=config.push_to_hub,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.evaluation_frequency,
    )

    eval_callback = CustomEvaluationCallback(
        eval_dataset=None,
        datasets_dict=test_datasets_dict,   
        max_eval_samples=20,
        batch_size=4  
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=config.reward_functions,
        args=training_args,
        train_dataset=train_dataset_combined,
        eval_dataset=test_dataset_combined,
        callbacks=[eval_callback],
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if "wandb" in config.report_to:
        run.finish()

if __name__ == "__main__":
    train()