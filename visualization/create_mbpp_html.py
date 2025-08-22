#!/usr/bin/env python3
"""
Standalone script to create HTML visualizations from saved MBPP evaluation results.
Usage: python create_mbpp_html.py --results_path path/to/results.json --output_file output.html
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def create_mbpp_html_visualization(results: List[Dict], output_file: str):
    """
    Create HTML visualization for MBPP evaluation results.

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
    <title>MBPP Code Generation Evaluation Results</title>
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
        .explicit-task-section {
            margin: 15px 0;
        }
        .explicit-task-text {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #17a2b8;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            position: relative;
        }
        .explicit-task-text::before {
            content: "🎯";
            position: absolute;
            top: 10px;
            right: 15px;
            font-size: 18px;
            opacity: 0.7;
        }
        .test-item {
            padding: 8px;
            margin: 5px 0;
            border-radius: 4px;
            font-family: monospace;
            font-size: 13px;
        }
        .test-passed {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .test-failed {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
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
        .pass-rate, .pass-at-1 {
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        }
        .problem-categories {
            margin-top: 20px;
        }
        .category-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
        .category-item.partial {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
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
        .collapsed .toggle-icon {
            transform: rotate(-90deg);
        }
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
        .toggle-component:hover {
            background: #5a6268;
        }
        .component-content {
            display: block;
            padding: 10px;
            background: white;
            border-radius: 4px;
            margin-top: 10px;
        }
        .component-content.collapsed {
            display: none;
        }
        .think-content, .answer-content {
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
        }
        .test-imports {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .import-list {
            margin-top: 10px;
        }
        .import-item {
            background: #e9ecef;
            border: 1px solid #ced4da;
            border-radius: 4px;
            padding: 8px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 13px;
            color: #495057;
        }
        .test-header {
            font-weight: bold;
            margin-bottom: 8px;
            padding-bottom: 5px;
            border-bottom: 1px solid #dee2e6;
        }
        .test-assertion {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 8px;
            margin: 8px 0;
            font-family: monospace;
            font-size: 13px;
        }
        .test-error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            padding: 8px;
            margin: 8px 0;
            color: #721c24;
            font-size: 13px;
        }
        .code-section {
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
        .problem-metrics {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
        }
        .metric-badges {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .metric-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            text-align: center;
            white-space: nowrap;
            border: 1px solid;
        }
        .ttft-badge {
            background: #e3f2fd;
            border-color: #2196f3;
            color: #1565c0;
        }
        .token-badge {
            background: #f3e5f5;
            border-color: #9c27b0;
            color: #7b1fa2;
        }
        .completion-badge {
            background: #e8f5e8;
            border-color: #4caf50;
            color: #2e7d32;
        }
        .completion-badge.completion-high {
            background: #e8f5e8;
            border-color: #4caf50;
            color: #2e7d32;
        }
        .completion-badge.completion-medium {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        .completion-badge.completion-low {
            background: #ffebee;
            border-color: #f44336;
            color: #c62828;
        }
        .completion-badge:has(.task-incomplete) {
            background: #ffebee;
            border-color: #f44336;
            color: #c62828;
        }
    </style>
</head>
<body>
    <h1>🧪 MBPP Code Generation Evaluation Results</h1>
    <p>This visualization shows the code generation results for each MBPP problem, including generated code and test case results.</p>
    
"""

    # Calculate overall statistics
    total_problems = len(results)
    total_tests = sum(r["evaluation"]["total_tests"] for r in results)
    total_passed = sum(r["evaluation"]["tests_passed"] for r in results)
    overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    # Calculate Pass@1 (problems that passed ALL tests)
    fully_correct_problems = [
        r
        for r in results
        if r["evaluation"]["tests_passed"] == r["evaluation"]["total_tests"]
    ]
    pass_at_1 = (
        (len(fully_correct_problems) / total_problems * 100)
        if total_problems > 0
        else 0
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
    fully_correct = [
        r
        for r in results
        if r["evaluation"]["tests_passed"] == r["evaluation"]["total_tests"]
    ]
    partially_correct = [
        r
        for r in results
        if 0 < r["evaluation"]["tests_passed"] < r["evaluation"]["total_tests"]
    ]
    fully_incorrect = [r for r in results if r["evaluation"]["tests_passed"] == 0]

    html_content += f"""
    <div class="summary-stats">
        <h2>📊 Overall Results</h2>
        <div class="metrics-grid">
            <div class="metric-item">
                <strong>Total Problems:</strong> {total_problems}
            </div>
            <div class="metric-item">
                <strong>Total Tests:</strong> {total_tests}
            </div>
            <div class="metric-item">
                <strong>Tests Passed:</strong> {total_passed}
            </div>
            <div class="metric-item">
                <strong>Unit Test Pass Rate:</strong> <span class="pass-rate">{overall_pass_rate:.1f}%</span>
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
            <h3>🎯 Problem Categories</h3>
            <div class="category-grid">
                <div class="category-item correct">
                    <strong>✅ Fully Correct:</strong> {len(fully_correct)} problems
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in fully_correct]) if fully_correct else "None"}
                    </div>
                </div>
                <div class="category-item partial">
                    <strong>⚠️ Partially Correct:</strong> {len(partially_correct)} problems
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in partially_correct]) if partially_correct else "None"}
                    </div>
                </div>
                <div class="category-item incorrect">
                    <strong>❌ Fully Incorrect:</strong> {len(fully_incorrect)} problems
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in fully_incorrect]) if fully_incorrect else "None"}
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
        generated_code = result["generated_code"]
        evaluation = result["evaluation"]
        interleaved_components = result.get("interleaved_components", [])

        html_content += f"""
    <div class="problem-container">
        <div class="problem-header" onclick="toggleProblem({i})">
            <h2>Problem {problem_id + 1}</h2>
            <span class="problem-toggle-icon">▼</span>
        </div>
        
        <div class="problem-content" id="problem-content-{i}">
            <div class="problem-metrics">
                <div class="metric-badges">
                    <span class="metric-badge ttft-badge" title="Time to First Token Ratio - Lower is better">
                        🚀 TTFT: {result.get("ttft_ratio", 0):.3f}
                    </span>
                    <span class="metric-badge token-badge" title="Number of tokens generated">
                        📊 Tokens: {result.get("num_tokens", "N/A")}
                    </span>
                    <span class="metric-badge completion-badge" title="Task completion status">
                        {"✅" if result.get("task_completed", False) else "❌"} Task Complete
                    </span>
                </div>
            </div>
            
            <div class="prompt-text">{prompt}</div>
            
            <div class="explicit-task-section">
                <h4>🎯 Original Intent / Explicit Task</h4>
                <div class="explicit-task-text">{result.get("original_intent", result.get("explicit_task", "No explicit task provided"))}</div>
            </div>
"""

        # Show interleaved components if available (plan_first template)
        if interleaved_components and len(interleaved_components) > 0:
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

        html_content += f"""
            <div class="collapsible">
                <div class="collapsible-header" onclick="toggleCollapsible({i})">
                    <span>Generated Code (Extracted from last Answer)</span>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="collapsible-content" id="code-content-{i}">
                    <div class="code-section">{generated_code}</div>
                </div>
            </div>
            
            <div class="test-results">
                <h3>Test Results</h3>
                <p><strong>Tests Passed:</strong> {evaluation["tests_passed"]}/{evaluation["total_tests"]} ({evaluation["tests_passed"] / evaluation["total_tests"] * 100:.1f}%)</p>
"""

        # Show test imports if available
        if evaluation.get("test_imports") and len(evaluation["test_imports"]) > 0:
            html_content += f"""
                <div class="test-imports">
                    <h4>📦 Test Imports</h4>
                    <div class="import-list">
"""
            for import_stmt in evaluation["test_imports"]:
                html_content += f"""
                    <div class="import-item">{import_stmt}</div>
"""
            html_content += """
                    </div>
                </div>
"""

        # Show execution error if any
        if evaluation["execution_error"]:
            html_content += f"""
                <div class="error-message">
                    <strong>Execution Error:</strong> {evaluation["execution_error"]}
                </div>
"""

        # Show individual test results with more detail
        html_content += f"""
                <h4>🧪 Individual Test Cases</h4>
"""

        if evaluation["test_results"]:
            for test_idx, test_result in enumerate(evaluation["test_results"]):
                css_class = "test-passed" if test_result["passed"] else "test-failed"
                status = "✓ PASS" if test_result["passed"] else "✗ FAIL"
                test_number = test_idx + 1

                html_content += f"""
                <div class="test-item {css_class}">
                    <div class="test-header">
                        <strong>Test {test_number}: {status}</strong>
                    </div>
                    <div class="test-assertion">
                        <code>{test_result["test"]}</code>
                    </div>
"""

                if not test_result["passed"] and test_result["error"]:
                    html_content += f"""
                    <div class="test-error">
                        <strong>Error:</strong> {test_result["error"]}
                    </div>
"""

                html_content += "</div>"
        else:
            html_content += """
                <div class="test-item test-failed">
                    <div class="test-header">
                        <strong>⚠️ No Test Results Found</strong>
                    </div>
                    <div class="test-assertion">
                        <code>No test results available in evaluation</code>
                    </div>
                </div>
"""

        html_content += """
            </div>
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
        
        function toggleCollapsible(index) {
            const content = document.getElementById(`code-content-${index}`);
            const header = content.previousElementSibling;
            
            if (content.classList.contains('show')) {
                content.classList.remove('show');
                header.classList.remove('collapsed');
            } else {
                content.classList.add('show');
                header.classList.remove('collapsed');
            }
        }
        
        function toggleComponent(problemIndex, componentIndex) {
            console.log(`Toggling component: problem ${problemIndex}, component ${componentIndex}`);
            
            const contentId = `component-${problemIndex}-${componentIndex}`;
            const content = document.getElementById(contentId);
            
            if (!content) {
                console.error(`Component content not found: ${contentId}`);
                console.log('Available component IDs:', Array.from(document.querySelectorAll('[id^="component-"]')).map(el => el.id));
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
            
            console.log(`Found content:`, content);
            console.log(`Found button:`, button);
            console.log(`Content collapsed:`, content.classList.contains('collapsed'));
            
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                button.textContent = '▼';
                console.log(`Expanded component ${contentId}`);
            } else {
                content.classList.add('collapsed');
                button.textContent = '▶';
                console.log(`Collapsed component ${contentId}`);
            }
        }
        
        // Initialize all code sections as collapsed
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing page...');
            
            const headers = document.querySelectorAll('.collapsible-header');
            console.log(`Found ${headers.length} collapsible headers`);
            headers.forEach(header => {
                header.classList.add('collapsed');
            });
            
            // Initialize all interleaved components as collapsed
            const componentContents = document.querySelectorAll('.component-content');
            console.log(`Found ${componentContents.length} interleaved component contents`);
            componentContents.forEach((content, index) => {
                content.classList.add('collapsed');
            });
            
            // Initialize all problems as expanded (you can change this to 'collapsed' if you want them collapsed by default)
            const problemContents = document.querySelectorAll('.problem-content');
            console.log(`Found ${problemContents.length} problem contents`);
            problemContents.forEach((content, index) => {
                // content.classList.add('collapsed'); // Uncomment this line if you want problems collapsed by default
            });
        });
    </script>
</body>
</html>
"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated MBPP HTML visualization with {len(results)} problems")


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
        description="Create HTML visualization from MBPP evaluation results"
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
        default="mbpp_visualization.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    print("=== MBPP HTML Visualization Generator ===\n")
    print(f"Results file: {args.results_path}")
    print(f"Output file: {args.output_file}")

    try:
        # Load results
        results = load_results_from_file(args.results_path)

        if not results:
            print("❌ No results found in the file!")
            return False

        # Generate HTML visualization
        create_mbpp_html_visualization(results, args.output_file)

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
