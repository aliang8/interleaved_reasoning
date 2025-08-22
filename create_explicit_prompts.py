#!/usr/bin/env python3
"""
Script to create explicit prompts from ambiguous ones and save to parquet format.

This script takes ambiguous prompts and creates more explicit, detailed versions
that specify exactly what needs to be done, then saves them to a parquet file
with 'question' and 'extra_info' fields.
"""

import pandas as pd
import json
from typing import List, Dict, Any

# Global definitions
# To add new prompts, simply add them to both lists in the same order
AMBIGUOUS_PROMPTS = [
    # "Write a Python function to process a list of numbers and output the result.",
    "Write a function to clean a list of strings.",
    # "Write a function to clean a list of strings.",
    # "Write a function to calculate volume of a triangular prism given its dimensions.",
    # "Write a function to calculate volume of a triangular prism given its dimensions."
    "Write a function to sort a list.",
    "Write a function to sort a list.",
    # "Write a function to sort a list.",
    # "Write a function to sort a list.",
    # "Write a Python function to organize a name.",
    # "Write a Python function to format a date a particular way."
    # "Write a function to determine the shortest path length between two points in a network.",
    # "Write a function to determine the shortest path length between two points in a network."
]

EXPLICIT_TASKS = [
    # "Write a Python function that computes the volume of a triangular prism given lengths of the three sides of the triangular base and the height of the prism.",
    # "Write a Python function that returns the volume of a triangular prism given only the area of the triangular base and the height of the prism. Make sure the arguments are correct"
    # "Write a Python function that takes a list of numbers as input and calculates the sum and average, and returns these statistics as separate values",
    "Write a Python function that takes a list of strings and remove any duplicate strings from the list.",
    "Write a Python function that takes a list of strings and remove empty strings from the list.",
    # "Write a Python function that takes a list of integers and sorts them in descending order.",
    # "Write a Python function sorts of a list of strings alphabetically.",
    "Write a Python function that takes a list of strings and sorts them in descending order.",
    "Write a Python function that takes a list of strings and sorts them based on the length of the strings.",
    # "Write a Python function that takes a first name and last name and returns a formatted, Last, First.",
    # "Write a Python function that takes a date and returns the date in the format of Month Day, Year."
    # "Find the length of the shortest path between two nodes in an undirected, unweighted graph.",
    # "Find the length of the shortest path between two nodes in an directed, weighted graph where edges can have both positive and negative weights. Implement with Bellman-Ford algorithm."
]

# AMBIGUOUS_PROMPTS = [
#     "I want to plan a weekend trip to New York.",
#     "Help me book a dinner for four.",
#     "I want to buy a gift for my friend."
# ]

# EXPLICIT_TASKS = [
#     "I want to plan a budget-friendly weekend trip to New York focusing on major tourist landmarks and using public transportation.",
#     "Help me book a fancy restaurant dinner for four to celebrate a birthday.",
#     "I want to buy a budget-friendly, practical gift for my friend."
# ]


def create_explicit_tasks(
    ambiguous_prompts: List[str], explicit_tasks: List[str]
) -> List[Dict[str, Any]]:
    """
    Convert ambiguous prompts to explicit, detailed tasks.

    Args:
        ambiguous_prompts: List of vague/ambiguous prompts
        explicit_tasks: List of detailed versions corresponding to each prompt

    Returns:
        List of dictionaries with original questions and explicit task info
    """
    # Zip the questions and explicit tasks together
    explicit_tasks_list = []
    for i, (question, explicit_task) in enumerate(
        zip(ambiguous_prompts, explicit_tasks)
    ):
        task_entry = {
            "question": question,  # Original prompt stays as the question
            "extra_info": {
                "explicit_task": explicit_task,  # More detailed version
            },
        }
        explicit_tasks_list.append(task_entry)

    return explicit_tasks_list


def save_to_parquet(
    tasks: List[Dict[str, Any]], output_file: str = "explicit_prompts.parquet"
):
    """
    Save the explicit tasks to a parquet file.

    Args:
        tasks: List of task dictionaries
        output_file: Output parquet file path
    """
    # Convert to DataFrame
    df = pd.DataFrame(tasks)

    # Save to parquet
    df.to_parquet(output_file, index=False)
    print(f"✅ Saved {len(tasks)} explicit tasks to {output_file}")

    # Save to JSONL
    with open(output_file.replace(".parquet", ".jsonl"), "w", encoding="utf-8") as f:
        for task in tasks:
            json.dump(task, f, ensure_ascii=False)
            f.write("\n")


def main():
    """Main function to create explicit prompts and save them."""

    print("🔄 Converting ambiguous prompts to explicit tasks...")

    # Create explicit versions
    explicit_tasks = create_explicit_tasks(AMBIGUOUS_PROMPTS, EXPLICIT_TASKS)

    print(f"✅ Created {len(explicit_tasks)} explicit tasks")

    # Display a preview
    print("\n📋 Preview of explicit tasks:")
    for i, task in enumerate(explicit_tasks):
        print(f"\n--- Task {i + 1} ---")
        print(f"Question: {task['question']}")
        print(f"Explicit Task: {task['extra_info']['explicit_task']}")

    # Save to parquet
    save_to_parquet(explicit_tasks, "explicit_prompts_general.parquet")

    print("\n🎉 All done! The explicit prompts are ready for use.")


if __name__ == "__main__":
    main()
