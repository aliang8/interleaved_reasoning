#!/usr/bin/env python3
"""
Standalone script to create HTML visualizations from saved BigCodeBench evaluation results.
Usage: python create_bigcodebench_html.py --results_path path/to/results.json --output_file output.html
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import re

def extract_thinking_from_full_response(full_response: str) -> str:
    """Extract thinking content from full response if it contains <think> tags."""
    if not full_response or "<think>" not in full_response:
        return ""
    
    # Extract content between <think> and </think> tags
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, full_response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return the first thinking block
        return matches[0].strip()
    
    return full_response


def create_bigcodebench_html_visualization(results: List[Dict], output_file: str):
    """
    Create HTML visualization for BigCodeBench evaluation results.

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
    <title>BigCodeBench Code Generation Evaluation Results</title>
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
        .total-tokens-badge {
            background: #fff3e0;
            border-color: #ff9800;
            color: #e65100;
        }
        .tokens-to-answer-badge {
            background: #e8f5e8;
            border-color: #4caf50;
            color: #2e7d32;
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
        .thinking-section {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-left: 4px solid #ffc107;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .thinking-text {
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            background: white;
            padding: 10px;
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
        }
        .generated-answer-section {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            border-left: 4px solid #17a2b8;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .generated-answer-text {
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            background: white;
            padding: 10px;
            border-radius: 4px;
            max-height: 400px;
            overflow-y: auto;
        }
        .sandbox-results {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .sandbox-result {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
        }
        .sandbox-output {
            font-family: monospace;
            white-space: pre-wrap;
            font-size: 12px;
            line-height: 1.3;
            max-height: 200px;
            overflow-y: auto;
            background: #f8f9fa;
            padding: 8px;
            border-radius: 3px;
        }
        .libraries-section {
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }
        .library-item {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 8px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <h1>🧪 BigCodeBench Code Generation Evaluation Results</h1>
    <p>This visualization shows the code generation results for each BigCodeBench problem, including generated code, test cases, and sandbox execution results.</p>
    
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
            <div class="metric-item">
                <strong>Total Tokens Generated:</strong> <span class="total-tokens-badge">{total_tokens_sum:,}</span>
            </div>
            <div class="metric-item">
                <strong>Avg Total Tokens:</strong> <span class="total-tokens-badge">{avg_total_tokens:.1f}</span>
            </div>
            <div class="metric-item">
                <strong>Avg Tokens to Answer:</strong> <span class="tokens-to-answer-badge">{avg_tokens_to_answer:.1f}</span>
            </div>
            <div class="metric-item">
                <strong>Min Tokens to Answer:</strong> <span class="tokens-to-answer-badge">{min_tokens_to_answer}</span>
            </div>
            <div class="metric-item">
                <strong>Max Tokens to Answer:</strong> <span class="tokens-to-answer-badge">{max_tokens_to_answer}</span>
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
        full_response = result.get("full_response", "")  # Full response including thinking
        template_type = result.get("extra_info", {}).get("template_type", "default")
        test_cases = result.get("test_cases", "") # Changed from list to string
        sandbox_results = result.get("sandbox_results")
        libs = result.get("libs", "") # Changed from list to string

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
            
            <div class="prompt-text">{prompt}</div>
"""

        # Show libraries if available
        if libs and libs.strip():
            html_content += f"""
            <div class="libraries-section">
                <h4>📦 Required Libraries</h4>
"""
            # Split by lines and filter out empty lines
            lib_lines = [lib.strip() for lib in libs.splitlines() if lib.strip()]
            if lib_lines:
                for lib in lib_lines:
                    html_content += f"""
                <div class="library-item">{lib}</div>
"""
            else:
                # If no valid lines, show the original string
                html_content += f"""
                <div class="library-item">{libs}</div>
"""
            html_content += """
            </div>
"""

        # Show interleaved components if available
        if interleaved_components and len(interleaved_components) > 0:
            print(f"  Found {len(interleaved_components)} interleaved components for problem {i + 1}")
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

        html_content += f"""
            <div class="generated-answer-section">
                <h4>✅ Generated Answer</h4>
                <div class="generated-answer-text">{generated_code}</div>
            </div>
            
            <div class="test-results">
                <h3>Test Results</h3>
                <p><strong>Tests Passed:</strong> {evaluation["tests_passed"]}/{evaluation["total_tests"]} ({evaluation["tests_passed"] / evaluation["total_tests"] * 100:.1f}%)</p>
"""

        # Show test cases
        if test_cases:
            html_content += f"""
                <h4>🧪 Test Cases</h4>
                <div class="test-item test-passed">
                    <div class="test-header">
                        <strong>Test Case</strong>
                    </div>
                    <div class="test-assertion">
                        <code>{test_cases}</code>
                    </div>
                </div>
"""

        # Show sandbox results if available
        if sandbox_results:
            html_content += f"""
                <h4>🔬 Sandbox Execution Results</h4>
                <div class="sandbox-results">
"""
            for sandbox_idx, sandbox_result in enumerate(sandbox_results):
                if sandbox_result:
                    stdout = sandbox_result.get("stdout", "")
                    stderr = sandbox_result.get("stderr", "")
                    html_content += f"""
                    <div class="sandbox-result">
                        <h5>Test {sandbox_idx + 1} Execution</h5>
"""
                    if stdout:
                        html_content += f"""
                        <div class="sandbox-output">
                            <strong>stdout:</strong>
{stdout}
                        </div>
"""
                    if stderr:
                        html_content += f"""
                        <div class="sandbox-output">
                            <strong>stderr:</strong>
{stderr}
                        </div>
"""
                    html_content += """
                    </div>
"""
                else:
                    html_content += f"""
                    <div class="sandbox-result">
                        <h5>Test {sandbox_idx + 1} Execution</h5>
                        <div class="sandbox-output">
                            <strong>No execution result available</strong>
                        </div>
                    </div>
"""
            html_content += """
                </div>
"""

        # Show execution error if any
        if evaluation.get("execution_error"):
            html_content += f"""
                <div class="error-message">
                    <strong>Execution Error:</strong> {evaluation["execution_error"]}
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
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing page...');
            
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

    print(f"Generated BigCodeBench HTML visualization with {len(results)} problems")


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
        description="Create HTML visualization from BigCodeBench evaluation results"
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
        default="bigcodebench_visualization.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    print("=== BigCodeBench HTML Visualization Generator ===\n")
    print(f"Results file: {args.results_path}")
    print(f"Output file: {args.output_file}")

    try:
        # Load results
        results = load_results_from_file(args.results_path)

        if not results:
            print("❌ No results found in the file!")
            return False

        # Generate HTML visualization
        create_bigcodebench_html_visualization(results, args.output_file)

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