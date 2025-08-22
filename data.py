from datasets import load_dataset


def load_math500_dataset(dataset_name="math500"):
    """
    Load Math500 or AIME 2024 dataset and return the test split.

    Args:
        dataset_name: Name of dataset to load ("math500" or "aime2024")

    Returns:
        List of examples with problem, answer, etc.
    """
    if dataset_name == "math500":
        print("Loading Math500 dataset...")
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        has_level = True
        test_data = dataset["test"]
    elif dataset_name == "aime2024":
        print("Loading AIME 2024 dataset...")
        dataset = load_dataset("HuggingFaceH4/aime_2024")
        has_level = False
        test_data = dataset["train"]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"✓ Loaded {dataset_name} test split with {len(test_data)} examples")

    # Convert to list of dictionaries for easier processing
    examples = []
    for i, example in enumerate(test_data):
        example_data = {
            "id": i,
            "problem": example["problem"],
            "answer": example["answer"],
            "description": f"{dataset_name.title()} Problem {i + 1}",
        }

        # Add level information only if available
        if has_level:
            example_data["level"] = example.get("level", 0)

        examples.append(example_data)

        # Debug output for first few examples
        if i < 3:
            print(f"  Example {i + 1}:")
            print(f"    Problem: {example['problem'][:100]}...")
            print(f"    Answer: {example['answer']}")
            if has_level:
                print(f"    Level: {example.get('level', 0)}")

    return examples
