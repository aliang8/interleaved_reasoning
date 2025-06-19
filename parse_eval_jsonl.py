#!/usr/bin/env python3
"""
Script to parse and analyze JSONL evaluation files.
Usage: python parse_eval_jsonl.py <jsonl_file> [--max_examples 5] [--max_text_length 200]
"""

import json
import argparse
import statistics
from typing import Dict, List, Any
from collections import defaultdict
import sys


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                        continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return data


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def print_sample_entries(data: List[Dict], max_examples: int = 5, max_text_length: int = 200):
    """Print sample entries in a readable format."""
    print(f"\n{'='*80}")
    print(f"SAMPLE ENTRIES (showing first {min(max_examples, len(data))} out of {len(data)})")
    print(f"{'='*80}")
    
    for i, entry in enumerate(data[:max_examples]):
        print(f"\n--- Entry {i+1} ---")
        
        # Print input (truncated)
        if 'input' in entry:
            print(f"Input: {truncate_text(entry['input'], max_text_length)}")
        
        # Print output (truncated)
        if 'output' in entry:
            print(f"Output: {truncate_text(entry['output'], max_text_length)}")
        
        # Print numerical metrics
        numerical_fields = ['score', 'step', 'reward', 'content_score', 'format_score', 'combined_score']
        metrics = []
        for field in numerical_fields:
            if field in entry:
                metrics.append(f"{field}: {entry[field]}")
        
        if metrics:
            print(f"Metrics: {', '.join(metrics)}")
        
        # Print any other fields
        other_fields = {k: v for k, v in entry.items() 
                       if k not in ['input', 'output'] + numerical_fields}
        if other_fields:
            print(f"Other fields: {other_fields}")


def compute_statistics(data: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for numerical fields."""
    # Identify numerical fields
    numerical_fields = set()
    for entry in data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                numerical_fields.add(key)
    
    stats = {}
    for field in numerical_fields:
        values = []
        for entry in data:
            if field in entry and isinstance(entry[field], (int, float)):
                values.append(entry[field])
        
        if values:
            stats[field] = {
                'count': len(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'sum': sum(values)
            }
    
    return stats


def print_statistics(stats: Dict[str, Dict[str, float]]):
    """Print statistics in a formatted table."""
    print(f"\n{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    
    if not stats:
        print("No numerical fields found.")
        return
    
    # Print header
    print(f"{'Field':<15} {'Count':<8} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Sum':<12}")
    print("-" * 80)
    
    # Print stats for each field
    for field, field_stats in sorted(stats.items()):
        print(f"{field:<15} "
              f"{field_stats['count']:<8} "
              f"{field_stats['mean']:<10.3f} "
              f"{field_stats['median']:<10.3f} "
              f"{field_stats['std']:<10.3f} "
              f"{field_stats['min']:<10.3f} "
              f"{field_stats['max']:<10.3f} "
              f"{field_stats['sum']:<12.3f}")


def analyze_data_structure(data: List[Dict]):
    """Analyze and print information about the data structure."""
    print(f"\n{'='*80}")
    print("DATA STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    if not data:
        print("No data found.")
        return
    
    print(f"Total entries: {len(data)}")
    
    # Analyze fields
    all_fields = set()
    field_counts = defaultdict(int)
    field_types = defaultdict(set)
    
    for entry in data:
        all_fields.update(entry.keys())
        for key, value in entry.items():
            field_counts[key] += 1
            field_types[key].add(type(value).__name__)
    
    print(f"Total unique fields: {len(all_fields)}")
    print(f"\nField analysis:")
    print(f"{'Field':<20} {'Count':<8} {'Coverage':<10} {'Types':<20}")
    print("-" * 60)
    
    for field in sorted(all_fields):
        coverage = (field_counts[field] / len(data)) * 100
        types_str = ", ".join(sorted(field_types[field]))
        print(f"{field:<20} {field_counts[field]:<8} {coverage:<10.1f}% {types_str:<20}")


def main():
    parser = argparse.ArgumentParser(description="Parse and analyze JSONL evaluation files")
    parser.add_argument("jsonl_file", help="Path to the JSONL file")
    parser.add_argument("--max_examples", type=int, default=5, 
                       help="Maximum number of examples to display (default: 5)")
    parser.add_argument("--max_text_length", type=int, default=200,
                       help="Maximum length for text fields display (default: 200)")
    parser.add_argument("--no_samples", action="store_true",
                       help="Skip displaying sample entries")
    parser.add_argument("--stats_only", action="store_true",
                       help="Only show statistics, skip samples and structure analysis")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.jsonl_file}")
    data = load_jsonl(args.jsonl_file)
    
    if not data:
        print("No valid data found in the file.")
        return
    
    # Perform analysis
    if not args.stats_only:
        analyze_data_structure(data)
        
        if not args.no_samples:
            print_sample_entries(data, args.max_examples, args.max_text_length)
    
    # Compute and print statistics
    stats = compute_statistics(data)
    print_statistics(stats)
    
    # Additional insights
    if not args.stats_only:
        print(f"\n{'='*80}")
        print("INSIGHTS")
        print(f"{'='*80}")
        
        # Check for format compliance
        if 'format_score' in stats:
            format_mean = stats['format_score']['mean']
            print(f"Format compliance rate: {format_mean*100:.1f}%")
        
        # Check content vs combined scores
        if 'content_score' in stats and 'combined_score' in stats:
            content_mean = stats['content_score']['mean']
            combined_mean = stats['combined_score']['mean']
            print(f"Content score avg: {content_mean:.3f}, Combined score avg: {combined_mean:.3f}")
        
        # Check reward distribution
        if 'reward' in stats:
            reward_stats = stats['reward']
            print(f"Reward distribution: min={reward_stats['min']:.3f}, "
                  f"max={reward_stats['max']:.3f}, mean={reward_stats['mean']:.3f}")


if __name__ == "__main__":
    main() 