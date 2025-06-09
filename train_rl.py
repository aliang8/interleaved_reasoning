import os
import rich
import pyrallis
from dataclasses import dataclass
from typing import Optional, List
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import time
from utils import formatting_prompts_func, formatting_prompts_func_cot, compute_ttft, compute_ttft_from_tokens
from rewards import format_reward, accuracy_reward, conditional_reward_function, reward_tracker

THINK_ANSWER_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

@dataclass
class ExperimentConfig:
    # Data configuration
    dataset_name: str = "K-and-K/knights-and-knaves"
    train_data: str = "train/people3_num1000.jsonl"
    test_data: str = "test/people3_num1000.jsonl"
    train_subset_size: Optional[int] = 10  # None for full dataset
    test_subset_size: Optional[int] = None  # None for full dataset
    
    # Model configuration
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # GRPO Training configuration
    output_dir: str = "Qwen2-0.5B-GRPO-test"
    learning_rate: float = 1e-6
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 1e-6
    train_batch_size: int = 16
    validation_batch_size: int = 2048
    ppo_mini_batch_size: int = 32
    ppo_micro_batch_size: int = 16
    critic_micro_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 1
    bf16: bool = True
    fp16: bool = False
    
    # GRPO specific parameters
    kl_coefficient: float = 0.001
    kl_loss_type: str = "low_variance_kl"
    max_prompt_length: int = 3096
    max_response_length: int = 2548
    max_completion_length: int = 2548
    sampling_temperature: float = 0.8
    num_generations: int = 8
    num_samples_per_prompt: int = 8
    stable_training_threshold: float = 0.05
    critic_warmup_steps: int = 0
    evaluation_frequency: int = 200
    tensor_model_parallel_size: int = 2
    remove_unused_columns: bool = False
    
    # Reward functions
    reward_functions: List[str] = None
    intermediate_reward_type: str = "partial_credit"
    stable_training_epsilon: float = 0.05
    base_reward: float = 1.0
    lambda_a: float = 1.0
    
    # Logging and saving
    report_to: List[str] = None
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    push_to_hub: bool = False
    
    # Environment
    seed: int = 42
    hf_cache_dir: str = "/scr/aliang80/hf_cache"
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ]
        if self.reward_functions is None:
            self.reward_functions = [format_reward, accuracy_reward, conditional_reward_function]
        if self.report_to is None:
            self.report_to = ["tensorboard"]
        
        global reward_tracker
        reward_tracker.epsilon = self.stable_training_epsilon

def eval():
    config = ExperimentConfig()

    trained_model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
    )
    trained_tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def generate_with_reasoning(prompt):
        # Build the prompt from the dataset
        prompt_text = " ".join(entry['content'] for entry in prompt)

        # Tokenize and move to the same device as the model
        inputs = trained_tokenizer(prompt_text, return_tensors="pt").to(trained_model.device)

        # Generate text without gradients
        start_time = time.time()
        with torch.no_grad():
            output_ids = trained_model.generate(**inputs, max_length=500)
        end_time = time.time()

        # Decode and extract model response
        generated_text = trained_tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Get inference time
        inference_duration = end_time - start_time

        # Get number of generated tokens
        num_input_tokens = inputs['input_ids'].shape[1]
        num_generated_tokens = output_ids.shape[1] - num_input_tokens

        # Compute normalized TTFT
        ttft = compute_ttft_from_tokens(
            inputs['input_ids'][0], 
            output_ids[0], 
            trained_tokenizer
        )

        return generated_text, inference_duration, num_generated_tokens, ttft
    
    generated_text, inference_duration, num_generated_tokens, ttft = generate_with_reasoning(prompt)
    print(f"Inference time: {inference_duration:.2f} seconds")
    print(f"Generated tokens: {num_generated_tokens}")
    print(f"Normalized TTFT: {ttft:.4f}")
    print(generated_text)

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

    response_template = "\n### Answer:\n"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = load_dataset(config.dataset_name, "train", split="2ppl")
    test_dataset = load_dataset(config.dataset_name, "test", split="2ppl")
    
    print("="*80)
    print("DATASET METADATA AND STRUCTURE")
    print("="*80)
    
    print(f"Dataset name: {config.dataset_name}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\n" + "-"*40)
    print("TRAIN DATASET STRUCTURE")
    print("-"*40)
    print(f"Column names: {train_dataset.column_names}")
    print(f"Features: {train_dataset.features}")
    
    if len(train_dataset) > 0:
        print("\nSample from train dataset:")
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
    
    if 'quiz' in train_dataset.column_names:
        quiz_lengths = [len(item) for item in train_dataset['quiz']]
        print(f"Quiz length stats - Min: {min(quiz_lengths)}, Max: {max(quiz_lengths)}, Avg: {sum(quiz_lengths)/len(quiz_lengths):.1f}")
    
    if 'solution' in train_dataset.column_names:
        solution_lengths = [len(str(item)) for item in train_dataset['solution']]
        print(f"Solution length stats - Min: {min(solution_lengths)}, Max: {max(solution_lengths)}, Avg: {sum(solution_lengths)/len(solution_lengths):.1f}")
    
    if 'messages' in train_dataset.column_names:
        msg_counts = [len(item) if isinstance(item, list) else 1 for item in train_dataset['messages']]
        print(f"Messages count stats - Min: {min(msg_counts)}, Max: {max(msg_counts)}, Avg: {sum(msg_counts)/len(msg_counts):.1f}")
    
    print("="*80)
    
    if config.train_subset_size is not None:
        train_dataset = train_dataset.select(range(config.train_subset_size))
        print(f"Selected train subset: {len(train_dataset)} samples")
    if config.test_subset_size is not None:
        test_dataset = test_dataset.select(range(config.test_subset_size))
        print(f"Selected test subset: {len(test_dataset)} samples")

    print(f"\nFinal train dataset: {train_dataset}")
    print(f"Final test dataset: {test_dataset}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        load_in_4bit=True,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    if config.use_lora:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    training_args = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.actor_learning_rate,
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
        # kl_coef=config.kl_coefficient,

        report_to=config.report_to,
        logging_steps=config.logging_steps,
        push_to_hub=config.push_to_hub,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        eval_strategy="steps",
        eval_steps=config.evaluation_frequency,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=config.reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()