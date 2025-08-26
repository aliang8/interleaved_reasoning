#!/usr/bin/env python3
"""
Plot metrics comparison between different methods.
Creates a scatter plot of tokens spent vs pass rate for different approaches.
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from pathlib import Path
from PIL import ImageFont

# Set matplotlib to use a nicer font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.size'] = 14

# Load a nice big readable font
font = font_manager.FontProperties(family="sans-serif", weight="bold")
file = font_manager.findfont(font)
font = ImageFont.truetype(file, 12)


def load_metrics(json_file: str) -> dict:
    """Load metrics from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_metrics_plot(metrics_data: dict, output_file: str = None):
    """
    Create a scatter plot comparing average tokens spent vs pass@1 for different methods.
    
    Args:
        metrics_data: Dictionary containing metrics for each method
        output_file: Optional output file path for saving the plot
    """
    # Define method configurations
    method_configs = {
        "think-answer": {"color": "#1f77b4", "marker": "o", "size": 800, "label": "Think-Answer"},
        "no-thinking": {"color": "#ff7f0e", "marker": "s", "size": 800, "label": "No-Thinking"},
        "best-of-n": {"color": "#2ca02c", "marker": "^", "size": 800, "label": "Best-of-N"},
        "rewind-and-repeat": {"color": "#d62728", "marker": "D", "size": 800, "label": "Rewind-and-Repeat"}
    }
    
    # Extract dataset name for title
    dataset_name = metrics_data.get("dataset", "Unknown Dataset")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot each method
    for method, config in method_configs.items():
        if method in metrics_data:
            method_data = metrics_data[method]
            
            # Extract data points
            avg_tokens = method_data.get("avg_token_spent")
            pass_at_1 = method_data.get("pass@1")
            
            if avg_tokens is not None and pass_at_1 is not None:
                # Create scatter plot
                ax.scatter(
                    avg_tokens, 
                    pass_at_1,
                    c=config["color"],
                    marker=config["marker"],
                    s=config["size"],
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=2,
                    label=config["label"]
                )
                
                # No method labels on shapes - cleaner look
    
    # Customize the plot with bigger fonts
    ax.set_xlabel("Average Tokens Spent", fontsize=28, fontweight='bold')
    ax.set_ylabel("Pass@1 (%)", fontsize=28, fontweight='bold')
    ax.set_title(f"Tokens Spent v.s. Pass@1", fontsize=32, fontweight='bold')
    
    # Set axis limits and grid (only bottom and left lines)
    ax.grid(True, alpha=0.3, axis='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Focus y-axis on the range where most Pass@1 values are (around 80)
    ax.set_ylim(70, 100)  # Pass@1 is percentage, focus on higher range
    
    # Scale x-axis with some padding
    if len([m for m in method_configs.keys() if m in metrics_data]) > 0:
        # Get all x values to determine range
        x_values = []
        for method, config in method_configs.items():
            if method in metrics_data:
                avg_tokens = metrics_data[method].get("avg_token_spent")
                if avg_tokens is not None:
                    x_values.append(avg_tokens)
        
        if x_values:
            x_min, x_max = min(x_values), max(x_values)
            x_range = x_max - x_min
            # Add 10% padding on each side
            x_padding = x_range * 0.1
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
    
    # Make tick labels bigger
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Add legend with bigger font and no border in bottom right
    ax.legend(fontsize=22, loc='lower right', frameon=False)
    
    # Tight layout and save
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot metrics comparison between methods")
    parser.add_argument("--input", "-i", type=str, help="Input JSON file with metrics data")
    parser.add_argument("--output", "-o", type=str, help="Output plot file (optional)")
    
    args = parser.parse_args()
    
    if not args.input:
        print("Error: Please provide an input JSON file with --input")
        print("Or use --create-sample to generate a sample file")
        return
    
    # Load and plot metrics
    try:
        metrics_data = load_metrics(args.input)
        create_metrics_plot(metrics_data, args.output)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the JSON file has the correct format.")
        print("Use --create-sample to see the expected structure.")


if __name__ == "__main__":
    main() 