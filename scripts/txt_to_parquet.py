import argparse
import pandas as pd
import sys
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Create a Parquet file and JSONL file for training from a text file containing prompts.")
    parser.add_argument('--input', type=str, required=True, help='Input text file containing prompts (one per line)')
    parser.add_argument('--prompt_key', type=str, default='prompt', help='Column name for the prompt')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    # Read prompts from input file
    with open(args.input, "r") as f:
        prompts = f.readlines()

    # Remove empty lines and strip whitespace
    prompts = [prompt.strip() for prompt in prompts if prompt.strip()]

    if not prompts:
        print("Error: No valid prompts found in the input file.")
        sys.exit(1)

    # Create output filenames with same name but different extensions
    base_name = os.path.splitext(args.input)[0]
    parquet_file = base_name + '.parquet'
    jsonl_file = base_name + '.jsonl'

    # Create DataFrame with prompts
    df = pd.DataFrame([{args.prompt_key: prompt} for prompt in prompts])
    df.to_parquet(parquet_file, index=False)
    
    # Save JSONL file
    with open(jsonl_file, 'w') as f:
        for prompt in prompts:
            json.dump({args.prompt_key: prompt}, f)
            f.write('\n')
    
    print(f"Wrote {len(prompts)} prompts to:")
    print(f"  - {parquet_file} (parquet format)")
    print(f"  - {jsonl_file} (jsonl format)")
    print(f"Column name: '{args.prompt_key}'")


if __name__ == "__main__":
    main()