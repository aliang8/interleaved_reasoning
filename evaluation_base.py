#!/usr/bin/env python3
"""
Base evaluation framework for standardized evaluation across different tasks.

This module provides a base class that handles the common evaluation logic
including worker initialization, batch processing, response generation,
and common metrics computation.
"""

import os
import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from omegaconf import DictConfig

from verl.workers.fsdp_workers import ActorRolloutRefWorker
from helpers import compute_completion_rate, compute_ttft_ratio

@dataclass
class EvaluationConfig:
    """Configuration for all evaluation tasks."""
    # Common parameters
    model_path: str = "Qwen/Qwen3-8B"
    output_dir: str = "logs/evaluation"
    batch_size: int = 50
    max_problems: int = 100
    temperature: float = 0.2
    template_type: str = "default"
    rollout_name: str = "vllm"
    response_length: Optional[int] = None
    no_thinking: bool = False
    
    # Math evaluation specific parameters
    autorater_service_url: str = "http://10.128.0.30:81"
    dataset: str = "math500"  # Choices: math500, aime2024
    WORLD_SIZE: int = 1
    RANK: int = 0
    LOCAL_RANK: int = 0

class BaseEvaluator(ABC):
    """
    Base class for standardized evaluation across different tasks.
    
    This class handles the common evaluation logic including:
    - Worker initialization and configuration
    - Batch processing and response generation
    - Common metrics computation (TTFT, completion rate)
    - Result storage and HTML visualization
    
    Subclasses should implement:
    - Data loading
    - Prompt creation
    - Response evaluation
    - Task-specific metrics
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dataclass instance
        """
        self.cfg = config
        self.rollout_config = None
        self.worker = None
        self.tokenizer = None
        self.examples = None
        self.all_results = []        

        # Metrics tracking
        self.total_task_completions = 0
        self.all_responses = []
        self.total_problems = 0
        
    def setup_environment(self):
        """Check GPU availability and setup environment."""
        if not torch.cuda.is_available():
            print("❌ CUDA not available! This evaluation requires a GPU.")
            return False
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name()}")

        os.environ["WORLD_SIZE"] = str(self.cfg.WORLD_SIZE)
        os.environ["RANK"] = str(self.cfg.RANK)
        os.environ["LOCAL_RANK"] = str(self.cfg.LOCAL_RANK)
        os.environ["MASTER_ADDR"] = "localhost"
        return True
    
    def load_data(self):
        """Load dataset - to be implemented by subclasses."""
        self.examples = self._load_dataset()
        
        # Limit number of problems
        if len(self.examples) > self.cfg.max_problems:
            self.examples = self.examples[:self.cfg.max_problems]
            print(f"Limited to {len(self.examples)} problems for evaluation")
        
        print(f"Processing all {len(self.examples)} problems")
    
    @abstractmethod
    def _load_dataset(self) -> List[Dict]:
        """Load dataset - must be implemented by subclasses."""
        pass
    
    def setup_config(self):
        """Setup and update configuration."""
        self.rollout_config = json.load(open("rollout_config.json"))
        self.rollout_config["model"]["path"] = self.cfg.model_path
        self.rollout_config["model"]["tokenizer_path"] = self.cfg.model_path
        self.rollout_config["rollout"]["temperature"] = self.cfg.temperature
        self.rollout_config["rollout"]["template_type"] = self.cfg.template_type
        self.rollout_config["rollout"]["name"] = self.cfg.rollout_name
        self.rollout_config["rollout"]["enable_thinking"] = not self.cfg.no_thinking
        
        # Override response length if specified
        if self.cfg.response_length is not None:
            self.rollout_config["rollout"]["response_length"] = self.cfg.response_length
            print(f"Response length overridden to: {self.cfg.response_length} tokens")
        
        if self.cfg.template_type == "plan_first":
            self.rollout_config["rollout"]["name"] = "vllm_rewind_and_repeat"
        
        print(f"Config: prompt_length={self.rollout_config['rollout']['prompt_length']}, response_length={self.rollout_config['rollout']['response_length']}")
        print(f"Generation: temperature={self.rollout_config['rollout']['temperature']}, batch_size={self.cfg.batch_size}")
        print(f"Template: {self.rollout_config['rollout']['template_type']}")
    
    def initialize_worker(self):
        """Initialize ActorRolloutRefWorker and model."""
        print("\nInitializing ActorRolloutRefWorker...")
        self.worker = ActorRolloutRefWorker(config=DictConfig(self.rollout_config), role="rollout")
        print("✓ ActorRolloutRefWorker initialized successfully!")
        
        self.worker.init_model()
        print("✓ Model initialized successfully!")
        
        # Load tokenizer for prompt creation
        self.tokenizer = self.worker.tokenizer
    
    def create_output_directory(self):
        """Create output directory for results."""
        output_dir = Path(self.cfg.output_dir)
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @abstractmethod
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """
        Create prompts for the batch - must be implemented by subclasses.
        
        Args:
            examples: List of examples to process
            batch_size: Number of examples per batch
            
        Yields:
            Batches of (batch_examples, prompts_dataproto)
        """
        pass
    
    def process_batch(self, batch_idx: int, batch_examples: List[Dict], prompts_dataproto):
        """
        Process a single batch of examples.
        
        Args:
            batch_idx: Index of the current batch
            batch_examples: Examples in this batch
            prompts_dataproto: DataProto for this batch
        """
        print(f"\n--- Batch {batch_idx + 1} ---")
        print(f"Processing problems {batch_idx * self.cfg.batch_size + 1}-{min((batch_idx + 1) * self.cfg.batch_size, len(self.examples))}")
        
        # Generate responses
        torch.cuda.empty_cache()
        output = self.worker.generate_sequences(prompts_dataproto)
        
        # Process results for this batch
        for i, example in enumerate(batch_examples):
            result = self._process_single_example(example, i, output, batch_idx)
            self.all_results.append(result)
    
    def _process_single_example(self, example: Dict, i: int, output, batch_idx: int) -> Dict:
        """
        Process a single example from the batch.
        
        Args:
            example: The example to process
            i: Index within the batch
            output: Output from worker.generate_sequences
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing the result for this example
        """
        # Get response tokens (excluding padding)
        response_tokens = output.batch['responses'][i]
        response_tokens = response_tokens[response_tokens != self.tokenizer.pad_token_id]
        
        # Decode response
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Collect response for TTFT computation
        self.all_responses.append(response_text)
        
        # Extract solution/code from response
        extracted_content = self._extract_content_from_response(response_text)
        
        # Evaluate the extracted content
        evaluation = self._evaluate_content(example, extracted_content)
        
        # Parse interleaved components if using plan_first template
        interleaved_components = []
        if self.rollout_config["rollout"]["template_type"] == "plan_first":
            from helpers import parse_interleaved_components
            interleaved_components = parse_interleaved_components(response_text)
            print(f"  Parsed {len(interleaved_components)} interleaved components")
            for comp in interleaved_components:
                print(f"    {comp['type'].upper()} {comp['index']}: {comp['content'][:100]}...")
        
        # Calculate task completion rate
        task_completed = compute_completion_rate(
            response_text,
            self.rollout_config["rollout"]["template_type"],
            self.rollout_config["rollout"]["enable_thinking"]
        )
        
        if task_completed:
            self.total_task_completions += 1
        
        # Calculate number of tokens for metrics
        num_tokens = int(response_tokens.shape[0]) if hasattr(response_tokens, 'shape') else 0
        
        # Calculate individual TTFT ratio for this problem
        ttft_ratio = compute_ttft_ratio(response_text, self.rollout_config["rollout"]["template_type"])
        
        # Track metrics
        self.total_problems += 1
        
        # Create base result structure
        result = self._create_base_result(
            example, response_text, extracted_content, evaluation,
            interleaved_components, task_completed, num_tokens, ttft_ratio
        )
        
        # Add task-specific fields
        result.update(self._add_task_specific_fields(example, evaluation))
        
        # Print progress
        self._print_progress(batch_idx, i, evaluation)
        
        return result
    
    @abstractmethod
    def _extract_content_from_response(self, response_text: str) -> str:
        """
        Extract the relevant content (solution/code) from response text.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _evaluate_content(self, example: Dict, extracted_content: str) -> Dict[str, Any]:
        """
        Evaluate the extracted content against ground truth.
        Must be implemented by subclasses.
        """
        pass
    
    def _create_base_result(self, example: Dict, response_text: str, extracted_content: str,
                           evaluation: Dict, interleaved_components: List[Dict],
                           task_completed: bool, num_tokens: int, ttft_ratio: float) -> Dict:
        """Create the base result structure common to all tasks."""
        return {
            'problem_id': example['id'],
            'prompt': example.get('prompt', example.get('problem', '')),
            'generated_code': extracted_content,  # Reuse field name for compatibility
            'full_response': response_text,
            'evaluation': evaluation,
            'interleaved_components': interleaved_components,
            'template_type': self.rollout_config["rollout"]["template_type"],
            # Metrics
            'task_completed': task_completed,
            'num_tokens': num_tokens,
            'ttft_ratio': ttft_ratio,
        }
    
    @abstractmethod
    def _add_task_specific_fields(self, example: Dict, evaluation: Dict) -> Dict[str, Any]:
        """
        Add task-specific fields to the result.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def _print_progress(self, batch_idx: int, i: int, evaluation: Dict):
        """
        Print progress for the current example.
        Must be implemented by subclasses.
        """
        pass
    
    def run_evaluation(self):
        """Run the complete evaluation process."""
        # Setup environment
        if not self.setup_environment():
            return False
        
        # Load data
        self.load_data()
        
        # Setup configuration
        self.setup_config()
        
        # Initialize worker
        self.initialize_worker()
        
        # Create output directory
        output_dir = self.create_output_directory()
        
        # Process problems in batches
        print(f"\n🚀 Processing {len(self.examples)} problems in batches of {self.cfg.batch_size}...")
        
        for batch_idx, (batch_examples, prompts_dataproto) in enumerate(
            self.create_prompts(self.examples, self.cfg.batch_size)
        ):
            self.process_batch(batch_idx, batch_examples, prompts_dataproto)
        
        # Save results
        self._save_results(output_dir)
        
        # Generate HTML visualization
        self._generate_html_visualization(output_dir)
        
        # Print final summary
        self._print_final_summary()
        
        return True
    
    def _save_results(self, output_dir: Path):
        """Save results to files."""
        # Generate filename
        thinking_tag = "thinking" if self.rollout_config["rollout"]["enable_thinking"] else "no_thinking"
        filename = f"{self._get_dataset_name()}_{self.cfg.template_type}_{self.rollout_config['rollout']['name']}_{thinking_tag}"
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{self.rollout_config['rollout']['n_candidates']}"
        
        # Save as JSON
        json_file = output_dir / f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {json_file}")
    
    @abstractmethod
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for filename generation."""
        pass
    
    @abstractmethod
    def _generate_html_visualization(self, output_dir: Path):
        """Generate HTML visualization - must be implemented by subclasses."""
        pass
    
    def _print_final_summary(self):
        """Print the final evaluation summary."""
        task_completion_rate = (self.total_task_completions / len(self.examples) * 100) if self.examples else 0
        
        print(f"\n=== Final Results ===")
        print(f"Total Problems: {len(self.all_results)}")
        print(f"Task Completion Rate: {task_completion_rate:.1f}% ({self.total_task_completions}/{len(self.examples)})")
        
        # Add task-specific metrics
        self._print_task_specific_metrics()
        
        print(f"===================\n")
        print(f"✨ {self._get_dataset_name().upper()} evaluation completed successfully!")
    
    @abstractmethod
    def _print_task_specific_metrics(self):
        """Print task-specific metrics - must be implemented by subclasses."""
        pass 