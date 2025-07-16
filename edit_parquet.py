#!/usr/bin/env python3
"""
Script to edit a parquet file's prompt field by appending a unit test instruction.
Usage:
    python edit_parquet.py --input input.parquet --output output.parquet --prompt_key question

python edit_parquet.py --input verl/data/bigcodebench/train.parquet --output verl/data/bigcodebench/train_with_unit_tests.parquet --prompt_key=prompt --instruction_type=replace;
python edit_parquet.py --input verl/data/bigcodebench/val.parquet --output verl/data/bigcodebench/val_with_unit_tests.parquet --prompt_key=prompt --instruction_type=replace;

python edit_parquet.py --input bigcodebench_data/bigcodebench_hard_interleaved_coding_dataset_train.parquet --output bigcodebench_data/bigcodebench_hard_interleaved_coding_dataset_train_with_unit_tests.parquet --prompt_key=question --instruction_type=add;
python edit_parquet.py --input bigcodebench_data/bigcodebench_hard_interleaved_coding_dataset_test.parquet --output bigcodebench_data/bigcodebench_hard_interleaved_coding_dataset_test_with_unit_tests.parquet --prompt_key=question --instruction_type=add;

"""
import json
import argparse
import pandas as pd

ADDITIONAL_INSTRUCTION = """
First, outline the solution in a markdown format.
Then, write the code to implement the solution.
Finally, generate unit tests to test the code. Format the unit tests as a python function with a docstring. Use this exact format:
```python
import unittest
from task_func import task_func

class Test(unittest.TestCase):
    def test_case_1(self):
        # Test case 1 description
        result = task_func(...)
        self.assertEqual(result, expected_value)
```
"""

UNIT_TEST_INSTRUCTION = """Also generate unit tests to test the code. Format the unit tests as a python function with a docstring.
Use this exact format:
```
import unittest
from task_func import task_func

class Test(unittest.TestCase):
    def test_case_1(self):
        # Test case 1 description
        result = task_func(...)
        self.assertEqual(result, expected_value)
```
"""


def main():
    parser = argparse.ArgumentParser(description="Edit prompt field in a parquet file.")
    parser.add_argument('--input', type=str, required=True, help='Input parquet file')
    parser.add_argument('--output', type=str, required=True, help='Output parquet file')
    parser.add_argument('--prompt_key', type=str, required=True, help='Prompt column name (e.g., question)')
    parser.add_argument('--instruction_type', type=str, required=True, help='Instruction type (add or replace)')
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)

    if args.prompt_key not in df.columns:
        raise ValueError(f"Prompt column '{args.prompt_key}' not found in columns: {df.columns}")

    print(f"Editing '{args.prompt_key}' column ...")
    # df[args.prompt_key] = df[args.prompt_key].astype(str) + ADDITIONAL_INSTRUCTION

    if args.instruction_type == "add":
        def apply_instruction(obj):
            content = str(obj)
            content = content.replace(UNIT_TEST_INSTRUCTION, "")
            content += ADDITIONAL_INSTRUCTION
            obj = content
            return str(obj)
    elif args.instruction_type == "replace":
        def apply_instruction(obj):
            content = obj[0]["content"]
            content = content.replace(UNIT_TEST_INSTRUCTION, "")
            content += ADDITIONAL_INSTRUCTION
            obj[0]["content"] = content
            return str(obj)
    
    df[args.prompt_key] = df[args.prompt_key].apply(apply_instruction)

    print(f"Saving to {args.output} ...")
    df.to_parquet(args.output)

    print("Done.")

    # also save to jsonl
    df.to_json(args.output.replace(".parquet", ".jsonl"), orient="records", lines=True)


if __name__ == "__main__":
    main() 