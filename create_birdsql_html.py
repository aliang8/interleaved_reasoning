#!/usr/bin/env python3
"""
Standalone script to create HTML visualizations from saved BirdSQL evaluation results.
Usage: python create_birdsql_html.py --results_path path/to/results.json --output_file output.html
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from html_base import (
    get_shared_css, get_shared_javascript, render_interleaved_components,
    render_thinking_section, render_problem_metrics
)


def create_birdsql_html_visualization(results: List[Dict], output_file: str):
    """
    Create HTML visualization for BirdSQL evaluation results.

    Args:
        results: List of evaluation results
        output_file: Path to save the HTML file
    """
    # Start with the HTML header and CSS
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BirdSQL Text-to-SQL Evaluation Results</title>
    <style>
""" + get_shared_css() + """
        .sql-section {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .sql-header {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #dee2e6;
        }
        .sql-content {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            background: white;
            padding: 10px;
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e9ecef;
        }
        .difficulty-badge {
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            text-align: center;
            white-space: nowrap;
            border: 1px solid;
        }
        .difficulty-simple {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .difficulty-moderate {
            background: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        .difficulty-challenging {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .database-info {
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 6px;
            padding: 10px;
            margin: 10px 0;
            font-size: 14px;
        }
        .evidence-section {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .evidence-text {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            background: white;
            padding: 10px;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }
        .schema-section {
            background: #e8f5e8;
            border: 1px solid #c8e6c9;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .schema-text {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 12px;
            line-height: 1.3;
            background: white;
            padding: 10px;
            border-radius: 4px;
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }
    </style>
</head>
<body>
    <h1>🐦 BirdSQL Text-to-SQL Evaluation Results</h1>
    <p>This visualization shows the SQL generation results for each BirdSQL question, including database context, thinking, and SQL correctness.</p>
    
"""

    # Calculate overall statistics
    total_problems = len(results)
    total_correct = sum(1 for r in results if r.get("sql_correct", False))
    overall_accuracy = (
        (total_correct / total_problems * 100) if total_problems > 0 else 0
    )

    # Calculate Pass@1 (questions that generated correct SQL)
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

    # Calculate syntax validity
    total_syntax_valid = sum(1 for r in results if r.get("syntax_valid", False))
    syntax_validity_rate = (
        (total_syntax_valid / total_problems * 100) if total_problems > 0 else 0
    )

    # Calculate difficulty-based statistics
    difficulty_stats = {}
    for result in results:
        difficulty = result.get("difficulty", "unknown")
        if difficulty not in difficulty_stats:
            difficulty_stats[difficulty] = {"total": 0, "correct": 0}
        difficulty_stats[difficulty]["total"] += 1
        if result.get("sql_correct", False):
            difficulty_stats[difficulty]["correct"] += 1

    # Categorize problems
    correct_sql = [r for r in results if r.get("sql_correct", False)]
    incorrect_sql = [r for r in results if not r.get("sql_correct", False)]
    syntax_invalid = [r for r in results if not r.get("syntax_valid", False)]

    html_content += f"""
    <div class="summary-stats">
        <h2>📊 Overall Results</h2>
        <div class="metrics-grid">
            <div class="metric-item">
                <strong>Total Questions:</strong> {total_problems}
            </div>
            <div class="metric-item">
                <strong>Correct SQL:</strong> {total_correct}
            </div>
            <div class="metric-item">
                <strong>Incorrect SQL:</strong> {total_problems - total_correct}
            </div>
            <div class="metric-item">
                <strong>SQL Accuracy:</strong> <span class="accuracy-rate">{overall_accuracy:.1f}%</span>
            </div>
            <div class="metric-item">
                <strong>Pass@1:</strong> <span class="pass-at-1">{pass_at_1:.1f}%</span>
            </div>
            <div class="metric-item">
                <strong>Task Completion Rate:</strong> <span class="completion-badge {("completion-high" if task_completion_rate >= 80 else "completion-medium" if task_completion_rate >= 50 else "completion-low")}">{task_completion_rate:.1f}%</span>
            </div>
            <div class="metric-item">
                <strong>Syntax Validity:</strong> <span class="completion-badge {("completion-high" if syntax_validity_rate >= 80 else "completion-medium" if syntax_validity_rate >= 50 else "completion-low")}">{syntax_validity_rate:.1f}%</span>
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
                    <strong>✅ Correct SQL:</strong> {len(correct_sql)} questions
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in correct_sql]) if correct_sql else "None"}
                    </div>
                </div>
                <div class="category-item incorrect">
                    <strong>❌ Incorrect SQL:</strong> {len(incorrect_sql)} questions
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in incorrect_sql]) if incorrect_sql else "None"}
                    </div>
                </div>
                <div class="category-item partial">
                    <strong>⚠️ Invalid Syntax:</strong> {len(syntax_invalid)} questions
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in syntax_invalid]) if syntax_invalid else "None"}
                    </div>
                </div>
            </div>
        </div>
"""

    # Add difficulty-based statistics if available
    if difficulty_stats:
        html_content += """
        <div class="problem-categories">
            <h3>📈 Difficulty-based Statistics</h3>
            <div class="category-grid">
"""
        
        for difficulty, stats in difficulty_stats.items():
            if stats["total"] > 0:
                accuracy = (stats["correct"] / stats["total"] * 100)
                difficulty_class = f"difficulty-{difficulty}" if difficulty in ["simple", "moderate", "challenging"] else ""
                
                html_content += f"""
                <div class="category-item {difficulty_class}">
                    <strong>{difficulty.capitalize()}:</strong> {accuracy:.1f}% ({stats['correct']}/{stats['total']})
                    <div class="problem-indices">
                        {", ".join([str(r["problem_id"] + 1) for r in results if r.get("difficulty") == difficulty])}
                    </div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""

    html_content += """
    </div>
"""

    # Generate problem details
    for i, result in enumerate(results):
        problem_id = result["problem_id"]
        prompt = result["prompt"]
        question = result.get("question", "")
        db_id = result.get("db_id", "")
        difficulty = result.get("difficulty", "unknown")
        evidence = result.get("evidence", "")
        ground_truth_sql = result.get("ground_truth_sql", "")
        generated_sql = result.get("generated_sql", "")
        sql_correct = result.get("sql_correct", False)
        syntax_valid = result.get("syntax_valid", False)
        evaluation = result["evaluation"]
        interleaved_components = result.get("interleaved_components", [])
        full_response = result.get("full_response", "")
        template_type = result.get("template_type", "default")

        # Get difficulty badge class
        difficulty_class = f"difficulty-{difficulty}" if difficulty in ["simple", "moderate", "challenging"] else ""

        html_content += f"""
    <div class="problem-container">
        <div class="problem-header" onclick="toggleProblem({i})">
            <h2>Question {problem_id + 1}</h2>
            <span class="problem-toggle-icon">▼</span>
        </div>
        
        <div class="problem-content" id="problem-content-{i}">
{render_problem_metrics(result)}
            
            <div class="database-info">
                <strong>🗄️ Database:</strong> {db_id}
                <span class="difficulty-badge {difficulty_class}">{difficulty.capitalize()}</span>
            </div>
            
            <div class="prompt-text">
                <strong>Question:</strong>
{question}
            </div>
"""

        # Show database schema
        database_schema = result.get("database_schema", "")
        if database_schema and database_schema.strip():
            html_content += f"""
            <div class="schema-section">
                <h4>🗄️ Database Schema</h4>
                <div class="schema-text">{database_schema}</div>
            </div>
"""
        
        # Show evidence if available
        if evidence and evidence.strip():
            html_content += f"""
            <div class="evidence-section">
                <h4>📚 External Knowledge Evidence</h4>
                <div class="evidence-text">{evidence}</div>
            </div>
"""

        # Show interleaved components if available (plan_first template)
        if interleaved_components and len(interleaved_components) > 0:
            html_content += render_interleaved_components(interleaved_components, i, template_type)
        else:
            # Default template: show thinking block if present
            html_content += render_thinking_section(full_response)

        # Show SQL comparison
        correctness_class = "correct" if sql_correct else "incorrect"
        correctness_icon = "✅" if sql_correct else "❌"
        syntax_class = "correct" if syntax_valid else "incorrect"
        syntax_icon = "✅" if syntax_valid else "❌"
        
        # Get execution information
        execution_result = result.get("execution_result", {})
        execution_error = result.get("execution_error")
        has_execution = execution_result and not execution_error
        
        html_content += f"""
            <div class="answer-correctness {correctness_class}">
                {correctness_icon} SQL Generation: {correctness_class.upper()}
            </div>
            
            <div class="answer-correctness {syntax_class}">
                {syntax_icon} SQL Syntax: {syntax_class.upper()}
            </div>
"""
        
        # Show execution status if available
        if has_execution:
            exec_class = "correct" if execution_result.get('res') == 1 else "incorrect"
            exec_icon = "✅" if execution_result.get('res') == 1 else "❌"
            html_content += f"""
            <div class="answer-correctness {exec_class}">
                {exec_icon} SQL Execution: {exec_class.upper()}
            </div>
"""
        elif execution_error:
            html_content += f"""
            <div class="answer-correctness incorrect">
                ❌ SQL Execution: FAILED ({execution_error})
            </div>
"""
        
        # Add SQL sections and response
        html_content += f"""
            <div class="sql-section">
                <h4>🎯 Ground Truth SQL</h4>
                <div class="sql-content">{ground_truth_sql}</div>
            </div>
            
            <div class="sql-section">
                <h4>💡 Generated SQL</h4>
                <div class="sql-content">{generated_sql if generated_sql else "No SQL generated"}</div>
            </div>
            
            <div class="generated-answer-section">
                <h4>📝 Full Response</h4>
                <div class="generated-answer-text">{result.get("generated_code", "No response generated")}</div>
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

    print(f"Generated BirdSQL HTML visualization with {len(results)} questions")


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
        description="Create HTML visualization from BirdSQL evaluation results"
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
        default="birdsql_visualization.html",
        help="Output HTML file path",
    )

    args = parser.parse_args()

    print("=== BirdSQL HTML Visualization Generator ===\n")
    print(f"Results file: {args.results_path}")
    print(f"Output file: {args.output_file}")

    try:
        # Load results
        results = load_results_from_file(args.results_path)

        if not results:
            print("❌ No results found in the file!")
            return False

        # Generate HTML visualization
        create_birdsql_html_visualization(results, args.output_file)

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