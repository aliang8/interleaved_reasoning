#!/usr/bin/env python3
"""
Helper script to merge all SFT parquet files into a single combined file.

This script finds all parquet files in the data/sft/ directory and merges them
into a single combined parquet file for training.

Usage: python data_gen/merge_sft_data.py --input_dir data/sft --output_file data/combined_sft_data.parquet
"""

import argparse
import os
import glob
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm


def find_parquet_files(input_dir: str) -> List[str]:
    """Find all parquet files in the input directory."""
    pattern = os.path.join(input_dir, "*.parquet")
    parquet_files = glob.glob(pattern)
    return sorted(parquet_files)


def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load a single parquet file and return as DataFrame."""
    print(f"  Loading: {os.path.basename(file_path)}")
    df = pd.read_parquet(file_path)
    print(f"    Rows: {len(df)}")
    return df


def merge_parquet_files(parquet_files: List[str]) -> pd.DataFrame:
    """Merge multiple parquet files into a single DataFrame."""
    if not parquet_files:
        raise ValueError("No parquet files found to merge")
    
    print(f"Found {len(parquet_files)} parquet files to merge:")
    
    # Load all DataFrames
    dataframes = []
    total_rows = 0
    
    for file_path in tqdm(parquet_files, desc="Loading parquet files"):
        df = load_parquet_file(file_path)
        dataframes.append(df)
        total_rows += len(df)
    
    print(f"\nMerging {len(dataframes)} DataFrames...")
    
    import ipdb; ipdb.set_trace()
    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Total rows: {len(combined_df)} (expected: {total_rows})")
    
    # Verify we didn't lose any data
    if len(combined_df) != total_rows:
        print(f"⚠️  Warning: Expected {total_rows} rows but got {len(combined_df)} rows")

    return combined_df


def main():
    parser = argparse.ArgumentParser(description="Merge all SFT parquet files into a single file")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/sft", 
        help="Input directory containing parquet files"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="data/combined_sft_data.parquet", 
        help="Output file path for combined parquet"
    )
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="Analyze the combined dataset after merging"
    )
    
    args = parser.parse_args()
    
    print(f"🚀 SFT DATA MERGER")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    print("=" * 60)
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Find all parquet files
    parquet_files = find_parquet_files(args.input_dir)
    
    if not parquet_files:
        print(f"❌ No parquet files found in '{args.input_dir}'")
        return
    
    # Merge the files
    try:
        combined_df = merge_parquet_files(parquet_files)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save combined file
        print(f"\nSaving combined data to: {args.output_file}")
        combined_df.to_parquet(args.output_file, index=False)
        print(f"✅ Successfully saved combined parquet file")
        
        print(f"\n🎉 Merge completed successfully!")
        print(f"Combined {len(parquet_files)} files into: {args.output_file}")
        print(f"Total samples: {len(combined_df)}")
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return


if __name__ == "__main__":
    main() 