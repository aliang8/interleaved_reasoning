"""
Shared utilities for dataset loading and processing across SFT and RL training.
"""

from datasets import load_dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Dict, Any, List
    from datasets import Dataset

THINK_ANSWER_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Dataset configurations - easily extensible for new datasets
DATASET_CONFIGS = {
    "kk": {
        "name": "K-and-K/knights-and-knaves",
        "train_split": "train",
        "test_split": "test",
        "split_config": "2ppl",
        "question_field": "quiz",
        "answer_field": "solution",
        "format_type": "kk"
    },
    "musique": {
        "name": "dgslibisey/MuSiQue", 
        "train_split": "train",
        "test_split": "validation",
        "split_config": None,
        "question_field": "question",
        "answer_field": "answer",
        "format_type": "musique"
    },
    "math500": {
        "name": "HuggingFaceH4/MATH-500",
        "train_split": None,  
        "test_split": "test",
        "split_config": None,
        "question_field": "problem",
        "answer_field": "answer",
        "format_type": "math"
    },
    "gpqa": {
        "name": "Idavidrein/gpqa",
        "train_split": None,  
        "test_split": "gpqa_diamond", 
        "split_config": None,  
        "question_field": "Question",
        "answer_field": "Correct Answer",
        "format_type": "gpqa"
    }
}

def load_single_dataset(dataset_key: str, config) -> 'Tuple[Dataset, Dataset, str]':
    """
    Load a single dataset based on its configuration.
    
    Args:
        dataset_key: Key identifying the dataset (e.g., "kk", "musique", "math500")
        config: Configuration object with subset size attributes
        
    Returns:
        Tuple of (train_dataset, test_dataset, info_string)
    """
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASET_CONFIGS.keys())}")
    
    dataset_config = DATASET_CONFIGS[dataset_key]
    dataset_name = dataset_config["name"]
    
    print(f"Loading {dataset_key} dataset...")
    
    # Load train and test splits
    train_dataset = None
    if dataset_config["train_split"] is not None:
        if dataset_config["split_config"]:
            train_dataset = load_dataset(dataset_name, dataset_config["train_split"], split=dataset_config["split_config"])
        else:
            train_dataset = load_dataset(dataset_name, split=dataset_config["train_split"])
    
    # Load test split
    if dataset_config["split_config"]:
        test_dataset = load_dataset(dataset_name, dataset_config["test_split"], split=dataset_config["split_config"])
    else:
        if dataset_key == "gpqa":
            test_dataset = load_dataset(dataset_name, dataset_config["test_split"])["train"]
        else:
            test_dataset = load_dataset(dataset_name, split=dataset_config["test_split"])
    
    # Apply dataset-specific subset size if specified
    subset_attr = f"{dataset_key}_subset_size"
    if hasattr(config, subset_attr):
        subset_size = getattr(config, subset_attr)
        if subset_size is not None:
            if train_dataset is not None:
                train_dataset = train_dataset.select(range(min(subset_size, len(train_dataset))))
            test_dataset = test_dataset.select(range(min(subset_size, len(test_dataset))))
    
    # Add dataset identifier to track source
    if train_dataset is not None:
        train_dataset = train_dataset.add_column("dataset_source", [dataset_key] * len(train_dataset))
    test_dataset = test_dataset.add_column("dataset_source", [dataset_key] * len(test_dataset))
    
    train_size = len(train_dataset) if train_dataset is not None else 0
    test_size = len(test_dataset)
    info_string = f"{dataset_key.upper()} - Train: {train_size}, Test: {test_size}"
    
    return train_dataset, test_dataset, info_string

def get_dataset_config(dataset_key: str) -> 'Dict[str, Any]':
    """Get configuration for a specific dataset."""
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_key]

def get_available_datasets():
    """Get list of available dataset keys."""
    return list(DATASET_CONFIGS.keys())

def make_conversation(example):
    """
    Convert a dataset example into conversation format with system and user roles.
    
    Args:
        example: Dataset example with dataset_source field
        
    Returns:
        Dictionary with prompt containing system and user messages
    """
    dataset_source = example["dataset_source"]
    dataset_config = DATASET_CONFIGS[dataset_source]
    
    # Extract question/problem from the example
    question = example[dataset_config["question_field"]]
    
    return {
        "prompt": [
            {"role": "system", "content": THINK_ANSWER_TEMPLATE},
            {"role": "user", "content": question},
        ],
    }

