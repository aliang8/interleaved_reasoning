#!/usr/bin/env python3
"""
Standalone script to create HTML visualizations from saved Long-Form QA evaluation results.
Usage: python create_longform_qa_html.py --results_path path/to/results.json --output_file output.html
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from html_base import (
    get_shared_css, get_shared_javascript, render_interleaved_components,
    render_thinking_section, render_problem_metrics
)


def create_longform_qa_html_visualization(results: List[Dict], output_file: str):
    """
    Create HTML visualization for Long-Form QA evaluation results.

    Args:
        results: List of evaluation results
        output_file: Path to save the HTML file
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Long-Form QA Evaluation Results</title>
    <style>
{get_shared_css()}
    </style>
</head>
<body>
    <h1>🧠 Long-Form QA Evaluation Results</h1>
    <p>This visualization shows the question answering results for each long-form question, including context, thinking, and answer correctness.</p>
    
"""

    # Calculate overall statistics
    total_problems = len(results)
    total_correct = sum(1 for r in results if r.get("answer_correct", False))
    overall_accuracy = (
        (total_correct / total_problems * 100) if total_problems > 0 else 0
    )

    # Calculate Pass@1 (questions that were answered correctly)
    pass_at_1 = (
        (total_correct / total_problems * 100) if total_problems > 0 else 0
    )

    # Calculate Task Completion Rate
    total_task_completions = sum(1 for r in results if r.get("task_completed", False))
    task_completion_rate = (
        (total_task_completions / total_problems * 100) if total_problems > 0 else 0
    )

    # Calculate TTFT statistics
    ttft_values = [
        r.get("ttft_ratio", 0)
        for r in results
        if r.get("ttft_ratio") is not None and r.get("ttft_ratio") != "N/A"
    ]
    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0
    min_ttft = min(ttft_values) if ttft_values else 0
    max_ttft = max(ttft_values) if ttft_values else 0

    # Categorize problems
    correct_answers = [r for r in results if r.get("answer_correct", False)]
    incorrect_answers = [r for r in results if not r.get("answer_correct", False)]

    html_content += f"""
    <div class="summary-stats">
        <h2>📊 Overall Results</h2>
        <div class="metrics-grid">
            <div class="metric-item">
                <strong>Total Questions:</strong> {total_problems}
            </div>
            <div class="metric-item">
                <strong>Correct Answers:</strong> {total_correct}
            </div>
            <div class="metric-item">
                <strong>Incorrect Answers:</strong> {total_problems - total_correct}
            </div>
            <div class="metric-item">
                <strong>Answer Accuracy:</strong> <span class="accuracy-rate">{overall_accuracy:.1f}%</span>
            </div>
            <div class="metric-item">
                <strong>Pass@1:</strong> <span class="pass-at-1">{pass_at_1:.1f}%</span>
            </div>
            <div class="metric-item">
                <strong>Task Completion Rate:</strong> <span class="completion-badge {("completion-high" if task_completion_rate >= 80 else "completion-medium" if task_completion_rate >= 50 else "completion-low")}">{task_completion_rate:.1f}%</span>
            </div>
            <div class="metric-item">
                <strong>Average TTFT:</strong> <span class="ttft-badge">{avg_ttft:.2f}</span>
            </div>
            <div class="metric-item">
                <strong>Min TTFT:</strong> <span class="ttft-badge">{min_ttft:.2f}</span>
            </div>
            <div class="metric-item">
                <strong>Max TTFT:</strong> <span class="ttft-badge">{max_ttft:.2f}</span>
            </div>
        </div>
        
        <div class="problem-categories">
            <h3>🎯 Question Categories</h3>
            <div class="category-grid">
                <div class="category-item correct">
                    <strong>✅ Correct Answers:</strong> {len(correct_answers)} questions
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in correct_answers]) if correct_answers else "None"}
                    </div>
                </div>
                <div class="category-item incorrect">
                    <strong>❌ Incorrect Answers:</strong> {len(incorrect_answers)} questions
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in incorrect_answers]) if incorrect_answers else "None"}
                    </div>
                </div>
            </div>
        </div>
    </div>
"""

    # Generate problem details
    for i, result in enumerate(results):
        problem_id = result["problem_id"]
        prompt = result["prompt"]
        context = result.get("context", "")
        question = result.get("question", "")
        options = result.get("options", [])
        gold_label = result.get("gold_label", 0)
        selected_option = result.get("selected_option", 0)
        answer_correct = result.get("answer_correct", False)
        evaluation = result["evaluation"]
        interleaved_components = result.get("interleaved_components", [])
        full_response = result.get("full_response", "")
        template_type = result.get("template_type", "default")

        html_content += f"""
    <div class="problem-container">
        <div class="problem-header" onclick="toggleProblem({i})">
            <h2>Question {problem_id + 1}</h2>
            <span class="problem-toggle-icon">▼</span>
        </div>
        
        <div class="problem-content" id="problem-content-{i}">
{render_problem_metrics(result)}
            
            <div class="prompt-text">
                <strong>Context:</strong>
{context[:500]}{"..." if len(context) > 500 else ""}

                <strong>Question:</strong>
{question}
            </div>
"""

        # Show interleaved components if available (plan_first template)
        if interleaved_components and len(interleaved_components) > 0 and template_type == "plan_first":
            html_content += render_interleaved_components(interleaved_components, i, template_type)
        else:
            # Default template: show thinking block if present
            html_content += render_thinking_section(full_response)

        # Show answer correctness
        correctness_class = "correct" if answer_correct else "incorrect"
        correctness_icon = "✅" if answer_correct else "❌"
        
        # Handle None selected_option
        selected_display = f"{selected_option} ({chr(64 + selected_option)})" if selected_option is not None else "Could not determine"
        
        html_content += f"""
            <div class="answer-correctness {correctness_class}">
                {correctness_icon} Answer: {correctness_class.upper()}
            </div>
            
            <div class="answer-details">
                <h4>📋 Answer Details</h4>
                <p><strong>Selected Option:</strong> {selected_display}</p>
                <p><strong>Correct Option:</strong> {gold_label} ({chr(64 + gold_label)})</p>
                <p><strong>Result:</strong> {"Correct" if answer_correct else "Incorrect"}</p>
            </div>
            
            <div class="answer-options">
                <h4>🔍 Multiple Choice Options</h4>
"""

        # Show all options with highlighting
        for idx, option in enumerate(options):
            option_letter = chr(65 + idx)  # A, B, C, D
            option_number = idx + 1
            
            css_class = "answer-option"
            if selected_option is not None and option_number == selected_option:
                css_class += " selected"
            if option_number == gold_label:
                css_class += " correct"
            elif selected_option is not None and option_number == selected_option and not answer_correct:
                css_class += " incorrect"
            
            html_content += f"""
                <div class="{css_class}">
                    <strong>{option_letter}. {option}</strong>
                    {f" (Selected)" if selected_option is not None and option_number == selected_option else ""}
                    {f" (Correct)" if option_number == gold_label else ""}
                </div>
"""

        html_content += """
            </div>
            
            <div class="generated-answer-section">
                <h4>💡 Generated Answer</h4>
                <div class="generated-answer-text">{result.get("generated_code", "No answer generated")}</div>
            </div>
        </div>
    </div>
"""

    html_content += f"""
    
    <script>
{get_shared_javascript()}
    </script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated Long-Form QA HTML visualization with {len(results)} questions")


def load_results_from_file(file_path: str) -> List[Dict]:
    """
    Load results from either JSON or JSONL file.

    Args:
        file_path: Path to the results file

    Returns:
        List of evaluation results
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".json":
        # Load from JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from JSON file: {file_path}")

    elif file_path.suffix.lower() == ".jsonl":
        # Load from JSONL file
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))
        print(f"Loaded {len(results)} results from JSONL file: {file_path}")

    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}. Use .json or .jsonl"
        )

    return results


def main():
    """Main function to create HTML visualization from saved results."""
    parser = argparse.ArgumentParser(
        description="Create HTML visualization from Long-Form QA evaluation results"
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to the results file (.json or .jsonl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="longform_qa_visualization.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    print("=== Long-Form QA HTML Visualization Generator ===\n")
    print(f"Results file: {args.results_path}")
    print(f"Output file: {args.output_file}")

    try:
        # Load results
        results = load_results_from_file(args.results_path)

        if not results:
            print("❌ No results found in the file!")
            return False

        # Generate HTML visualization
        create_longform_qa_html_visualization(results, args.output_file)

        print(f"\n✅ HTML visualization created successfully!")
        print(f"📁 Output file: {args.output_file}")

        return True

    except Exception as e:
        print(f"❌ Error creating HTML visualization: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main() 