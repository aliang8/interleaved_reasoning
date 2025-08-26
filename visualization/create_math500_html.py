#!/usr/bin/env python3
"""
Standalone script to create HTML visualizations from saved Math500 evaluation results.
Usage: python create_math500_html.py --results_path path/to/results.json --output_file output.html
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re


def create_math500_html_visualization(results: List[Dict], output_file: str):
    """
    Create HTML visualization for Math500 evaluation results.

    Args:
        results: List of evaluation results
        output_file: Path to save the HTML file
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Math500 Problem Solving Evaluation Results</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .problem-container {
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .problem-header {
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 0;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.2s;
        }
        .problem-header:hover {
            background: #34495e;
        }
        .problem-content {
            padding: 20px;
            display: block;
        }
        .problem-content.collapsed {
            display: none;
        }
        .problem-toggle-icon {
            font-size: 18px;
            transition: transform 0.3s ease;
        }
        .problem-container.collapsed .problem-toggle-icon {
            transform: rotate(-90deg);
        }
        .prompt-text {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            position: relative;
        }
        .prompt-text::before {
            content: "📝";
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 18px;
            opacity: 0.7;
        }
        .ground-truth-section {
            margin: 15px 0;
        }
        .ground-truth-text {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #4caf50;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            position: relative;
        }
        .ground-truth-text::before {
            content: "✅";
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 18px;
            opacity: 0.7;
        }
        .generated-answer-section {
            margin: 15px 0;
        }
        .generated-answer-text {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 15px;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
        }
        .thinking-section {
            margin: 15px 0;
        }
        .thinking-text {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 4px;
            padding: 15px;
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
        }
        .summary-stats {
            background: #e2e3e5;
            border: 1px solid #d6d8db;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 16px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-item {
            background: white;
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            text-align: center;
        }
        .accuracy-rate {
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        }
        .metrics-grid .metric-item .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 6px;
        }
        .badge.yes { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .badge.no { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .metrics-grid .metric-item small { color: #666; }
        .metrics-grid .metric-item small code { background: #f1f3f5; padding: 0 4px; border-radius: 3px; }
        .problem-categories {
            margin-top: 20px;
        }
        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .level-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .category-item {
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid;
        }
        .category-item.correct {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .category-item.incorrect {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .problem-indices {
            margin-top: 8px;
            font-size: 14px;
            font-family: monospace;
            background: rgba(255, 255, 255, 0.3);
            padding: 5px;
            border-radius: 4px;
            word-break: break-all;
        }
        .error-message {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 10px;
            color: #721c24;
            font-family: monospace;
            margin-bottom: 20px;
        }
        .collapsible {
            background: #e9ecef;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            margin: 10px 0;
            overflow: hidden;
        }
        .collapsible-header {
            background: #6c757d;
            color: white;
            padding: 10px;
            cursor: pointer;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .collapsible-header:hover {
            background: #5a6268;
        }
        .collapsible-content {
            padding: 15px;
            background: white;
            display: none;
        }
        .collapsible-content.show {
            display: block;
        }
        .toggle-icon {
            font-size: 18px;
            transition: transform 0.3s ease;
        }
        .collapsed .toggle-icon { transform: rotate(-90deg); }
        .interleaved-flow {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
        }
        .interleaved-section {
            margin: 15px 0;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid transparent;
        }
        .interleaved-section.think {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-left: 4px solid #ffc107;
        }
        .interleaved-section.answer {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-left: 4px solid #17a2b8;
        }
        .component-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #dee2e6;
        }
        .toggle-component {
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 8px;
            cursor: pointer;
            font-size: 12px;
            transition: background-color 0.2s;
        }
        .toggle-component:hover { background: #5a6268; }
        .component-content { display: block; padding: 10px; background: white; border-radius: 4px; margin-top: 10px; }
        .component-content.collapsed { display: none; }
        .think-content, .answer-content { font-family: monospace; white-space: pre-wrap; font-size: 13px; line-height: 1.4; }
        .problem-metrics {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
        }
        .metric-badges { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
        .metric-badge { padding: 6px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; text-align: center; white-space: nowrap; border: 1px solid; }
        .ttft-badge { background: #e3f2fd; border-color: #2196f3; color: #1565c0; }
        .token-badge { background: #f3e5f5; border-color: #9c27b0; color: #7b1fa2; }
        .total-tokens-badge { background: #fff3e0; border-color: #ff9800; color: #e65100; }
        .tokens-to-answer-badge { background: #e8f5e8; border-color: #4caf50; color: #2e7d32; }
        .completion-badge { background: #e8f5e8; border-color: #4caf50; color: #2e7d32; }
        .completion-badge:has(.task-incomplete) { background: #ffebee; border-color: #f44336; color: #c62828; }
        .solution-status { margin: 15px 0; padding: 15px; border-radius: 6px; font-weight: bold; text-align: center; font-size: 18px; }
        .solution-status.correct { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .solution-status.incorrect { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .autorater-explanation { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; padding: 15px; margin: 15px 0; font-style: italic; }
    </style>
</head>
<body>
    <h1>🧮 Math500 Problem Solving Evaluation Results</h1>
    <p>This visualization shows the problem solving results for each Math500 problem, including generated solutions and ground truth answers.</p>
    
"""

    # Calculate overall statistics
    total_problems = len(results)
    total_correct = sum(1 for r in results if r.get("solution_correct", False))
    overall_accuracy = (
        (total_correct / total_problems * 100) if total_problems > 0 else 0
    )

    # Calculate level-based statistics
    level_stats = {}
    for result in results:
        level = result.get("level", 0)
        if level not in level_stats:
            level_stats[level] = {"total": 0, "correct": 0}
        level_stats[level]["total"] += 1
        if result.get("solution_correct", False):
            level_stats[level]["correct"] += 1

    # Sort levels for display
    sorted_levels = sorted(level_stats.keys())

    # Calculate TTFT statistics
    ttft_values = [
        r.get("ttft_ratio", 0)
        for r in results
        if r.get("ttft_ratio") is not None and r.get("ttft_ratio") != "N/A"
    ]
    avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else 0
    min_ttft = min(ttft_values) if ttft_values else 0
    max_ttft = max(ttft_values) if ttft_values else 0

    # Calculate total tokens generated statistics
    total_tokens_values = [
        r.get("total_tokens_generated", 0)
        for r in results
        if r.get("total_tokens_generated") is not None and r.get("total_tokens_generated") != "N/A"
    ]
    total_tokens_sum = sum(total_tokens_values) if total_tokens_values else 0
    avg_total_tokens = total_tokens_sum / len(total_tokens_values) if total_tokens_values else 0
    
    # Calculate tokens to first answer statistics
    tokens_to_answer_values = [
        r.get("tokens_to_first_answer", 0)
        for r in results
        if r.get("tokens_to_first_answer") is not None and r.get("tokens_to_first_answer") != "N/A"
    ]
    avg_tokens_to_answer = sum(tokens_to_answer_values) / len(tokens_to_answer_values) if tokens_to_answer_values else 0
    min_tokens_to_answer = min(tokens_to_answer_values) if tokens_to_answer_values else 0
    max_tokens_to_answer = max(tokens_to_answer_values) if tokens_to_answer_values else 0

    # Categorize problems
    correct_solutions = [r for r in results if r.get("solution_correct", False)]
    incorrect_solutions = [r for r in results if not r.get("solution_correct", False)]

    html_content += f"""
    <div class="summary-stats">
        <h2>📊 Overall Results</h2>
        <div class="metrics-grid">
            <div class="metric-item">
                <strong>Total Problems:</strong> {total_problems}
            </div>
            <div class="metric-item">
                <strong>Correct Solutions:</strong> {total_correct}
            </div>
            <div class="metric-item">
                <strong>Incorrect Solutions:</strong> {total_problems - total_correct}
            </div>
            <div class="metric-item">
                <strong>Solution Accuracy:</strong> <span class="accuracy-rate">{overall_accuracy:.1f}%</span>
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
            <div class="metric-item">
                <strong>Total Tokens Generated:</strong> <span class="total-tokens-badge">{total_tokens_sum:,}</span>
            </div>
            <div class="metric-item">
                <strong>Average Total Tokens:</strong> <span class="total-tokens-badge">{avg_total_tokens:.0f}</span>
            </div>
            <div class="metric-item">
                <strong>Average Tokens to Answer:</strong> <span class="tokens-to-answer-badge">{avg_tokens_to_answer:.1f}</span>
            </div>
            <div class="metric-item">
                <strong>Min Tokens to Answer:</strong> <span class="tokens-to-answer-badge">{min_tokens_to_answer:.0f}</span>
            </div>
            <div class="metric-item">
                <strong>Max Tokens to Answer:</strong> <span class="tokens-to-answer-badge">{max_tokens_to_answer:.0f}</span>
            </div>
            <div class="metric-item">
                <strong>Thinking Enabled:</strong>
                <span class="badge yes">Yes</span> <small>(detected per-response via <code><think></code>)</small>
            </div>
        </div>
        
        <div class="problem-categories">
            <h3>🎯 Problem Categories</h3>
            <div class="category-grid">
                <div class="category-item correct">
                    <strong>✅ Correct Solutions:</strong> {len(correct_solutions)} problems
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in correct_solutions]) if correct_solutions else "None"}
                    </div>
                </div>
                <div class="category-item incorrect">
                    <strong>❌ Incorrect Solutions:</strong> {len(incorrect_solutions)} problems
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in incorrect_solutions]) if incorrect_solutions else "None"}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="level-analysis">
            <h3>📊 Performance by Difficulty Level</h3>
            <div class="level-grid">
"""

    # Add level-based analysis
    if sorted_levels:
        for level in sorted_levels:
            stats = level_stats[level]
            accuracy = (
                (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            )
            accuracy_class = (
                "correct"
                if accuracy >= 70
                else "incorrect"
                if accuracy < 50
                else "partial"
            )

            html_content += f"""
                    <div class="category-item {accuracy_class}">
                        <strong>Level {level}:</strong> {accuracy:.1f}% ({stats["correct"]}/{stats["total"]})
                        <div class="problem-indices">
                            Problems: {", ".join([str(r["problem_id"] + 1) for r in results if r.get("level") == level])}
                        </div>
                    </div>
    """

    html_content += """
            </div>
        </div>
"""

    # Helper to extract thinking content for default template
    def extract_thinking_from_full_response(full_response: str) -> str:
        if not isinstance(full_response, str) or not full_response:
            return ""
        # Extract the first <think>...</think> block
        match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    # Generate problem details
    for i, result in enumerate(results):
        problem_id = result["problem_id"]
        problem = result.get("problem", result.get("prompt", ""))
        generated_answer = result.get("generated_code", "")
        full_response = result.get("full_response", "")
        ground_truth_answer = result.get("ground_truth_answer", "")
        solution_correct = result.get("solution_correct", False)
        interleaved_components = result.get("interleaved_components", [])
        template_type = result.get("template_type", "default")

        html_content += f"""
    <div class="problem-container">
        <div class="problem-header" onclick="toggleProblem({i})">
            <h2>Problem {problem_id + 1} ({result.get("level", "N/A")})</h2>
            <span class="problem-toggle-icon">▼</span>
        </div>
        
        <div class="problem-content" id="problem-content-{i}">
            <div class="problem-metrics">
                <div class="metric-badges">
                    <span class="metric-badge ttft-badge" title="Time to First Token Ratio - Lower is better">
                        🚀 TTFT: {result.get("ttft_ratio", 0):.3f}
                    </span>
                    <span class="metric-badge token-badge" title="Number of tokens in final response">
                        📊 Response: {result.get("num_tokens", "N/A")}
                    </span>
                    <span class="metric-badge total-tokens-badge" title="Total tokens generated across all attempts (including rewind)">
                        🔄 Total: {result.get("total_tokens_generated", "N/A")}
                    </span>
                    <span class="metric-badge tokens-to-answer-badge" title="Number of tokens to first answer">
                        🎯 To Answer: {result.get("tokens_to_first_answer", "N/A")}
                    </span>
                    <span class="metric-badge completion-badge" title="Task completion status">
                        {"✅" if result.get("task_completed", False) else "❌"} Task Complete
                    </span>
                </div>
            </div>
            
            <div class="solution-status {"correct" if solution_correct else "incorrect"}">
                {"✅ SOLUTION CORRECT" if solution_correct else "❌ SOLUTION INCORRECT"}
            </div>
            
            <div class="prompt-text">{problem}</div>
            
            <div class="ground-truth-section">
                <h4>🎯 Ground Truth Answer</h4>
                <div class="ground-truth-text">{ground_truth_answer}</div>
            </div>
"""

        # Interleaved (plan_first): show think/answer flow
        if interleaved_components and len(interleaved_components) > 0 and template_type == "plan_first":
            html_content += f"""
            <div class="interleaved-flow">
                <h4>🔄 Generation Flow ({len(interleaved_components)} components)</h4>
"""
            for component_idx, component in enumerate(interleaved_components):
                if component["type"] == "think":
                    html_content += f"""
                <div class="interleaved-section think">
                    <div class="component-header">
                        <strong>💭 Think {component["index"]}</strong>
                        <button class="toggle-component" onclick="toggleComponent({i}, {component_idx})">▼</button>
                    </div>
                    <div class="component-content" id="component-{i}-{component_idx}">
                        <div class="think-content">{component["content"]}</div>
                    </div>
                </div>
"""
                else:  # answer
                    html_content += f"""
                <div class="interleaved-section answer">
                    <div class="component-header">
                        <strong>💡 Answer {component["index"]}</strong>
                        <button class="toggle-component" onclick="toggleComponent({i}, {component_idx})">▼</button>
                    </div>
                    <div class="component-content" id="component-{i}-{component_idx}">
                        <div class="answer-content">{component["content"]}</div>
                    </div>
                </div>
"""
            html_content += """
            </div>
"""
        else:
            # Default template: show thinking block if present
            thinking_text = extract_thinking_from_full_response(full_response)
            if thinking_text:
                html_content += f"""
            <div class="thinking-section">
                <h4>💭 Thinking</h4>
                <div class="thinking-text">{thinking_text}</div>
            </div>
"""

        # Dedicated Generated Answer section (extracted content)
        html_content += f"""
            <div class="generated-answer-section">
                <h4>✅ Generated Answer</h4>
                <div class="generated-answer-text">{generated_answer}</div>
            </div>
"""

        html_content += """
        </div>
    </div>
"""

    html_content += """
    
    <script>
        function toggleProblem(index) {
            const problemContainer = document.querySelector(`#problem-content-${index}`).closest('.problem-container');
            const content = document.getElementById(`problem-content-${index}`);
            const toggleIcon = problemContainer.querySelector('.problem-toggle-icon');
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                problemContainer.classList.remove('collapsed');
                toggleIcon.textContent = '▼';
            } else {
                content.classList.add('collapsed');
                problemContainer.classList.add('collapsed');
                toggleIcon.textContent = '▶';
            }
        }
        
        function toggleComponent(problemIndex, componentIndex) {
            const contentId = `component-${problemIndex}-${componentIndex}`;
            const content = document.getElementById(contentId);
            
            if (!content) {
                console.error(`Component content not found: ${contentId}`);
                return;
            }
            
            const componentSection = content.closest('.interleaved-section');
            if (!componentSection) {
                console.error(`Component section not found for: ${contentId}`);
                return;
            }
            
            const button = componentSection.querySelector('.toggle-component');
            if (!button) {
                console.error(`Toggle button not found for: ${contentId}`);
                return;
            }
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                button.textContent = '▼';
            } else {
                content.classList.add('collapsed');
                button.textContent = '▶';
            }
        }
        
        // Initialize all interleaved components as collapsed
        document.addEventListener('DOMContentLoaded', function() {
            const componentContents = document.querySelectorAll('.component-content');
            componentContents.forEach((content, index) => {
                content.classList.add('collapsed');
            });
        });
    </script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated Math500 HTML visualization with {len(results)} problems")


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
    """Main function to create HTML visualization from saved Math500 results."""
    parser = argparse.ArgumentParser(
        description="Create HTML visualization from Math500 evaluation results"
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
        default="math500_visualization.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    print("=== Math500 HTML Visualization Generator ===\n")
    print(f"Results file: {args.results_path}")
    print(f"Output file: {args.output_file}")

    try:
        # Load results
        results = load_results_from_file(args.results_path)

        if not results:
            print("❌ No results found in the file!")
            return False

        # Generate HTML visualization
        create_math500_html_visualization(results, args.output_file)

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
