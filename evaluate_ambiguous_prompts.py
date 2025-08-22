#!/usr/bin/env python3
"""
Ambiguous Prompts Evaluation Script

This script:
1. Loads generated ambiguous prompts from JSONL files
2. Extracts canonical solutions and unit tests
3. Runs the solutions against the tests
4. Evaluates correctness and creates HTML visualization

Run with: python evaluate_ambiguous_prompts.py --input_file path/to/ambiguous_data.jsonl
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset

# Import helper functions from mbpp_evaluation
from mbpp_evaluation import (
    evaluate_code_against_tests,
    extract_function_name_from_tests,
)

# Import HTML generation function from standalone module
from create_mbpp_html import create_mbpp_html_visualization


def load_ambiguous_data(input_file: str) -> List[Dict]:
    """
    Load ambiguous prompts data from JSONL file.

    Args:
        input_file: Path to the JSONL file

    Returns:
        List of ambiguous prompt entries
    """
    print(f"Loading ambiguous data from {input_file}...")

    entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                entry = json.loads(line.strip())
                entries.append(entry)

    print(f"✅ Loaded {len(entries)} entries from {input_file}")
    return entries


def extract_canonical_solution_and_tests(entry: Dict) -> tuple[str, List[str]]:
    """
    Extract canonical solution and unit tests from an entry.

    Args:
        entry: Dictionary containing the ambiguous prompt entry

    Returns:
        Tuple of (canonical_solution, unit_tests)
    """
    # Try to get generated solution and tests first
    canonical_solution = entry.get("extra_info", {}).get("generated_solution", "")
    unit_tests = entry.get("extra_info", {}).get("generated_tests", [])

    # If no generated solution, try to get from reward_model
    if not canonical_solution:
        reward_model = entry.get("reward_model", {})
        if reward_model and "ground_truth" in reward_model:
            canonical_solution = (
                reward_model["ground_truth"][0] if reward_model["ground_truth"] else ""
            )

    # If no generated tests, try to get from reward_model
    if not unit_tests:
        reward_model = entry.get("reward_model", {})
        if reward_model and "unit_tests" in reward_model:
            unit_tests_raw = (
                reward_model["unit_tests"][0] if reward_model["unit_tests"] else ""
            )
            if isinstance(unit_tests_raw, str):
                # Convert string tests to list of assert statements
                unit_tests = [
                    line.strip()
                    for line in unit_tests_raw.split("\n")
                    if line.strip() and "assert" in line.strip()
                ]
            else:
                unit_tests = unit_tests_raw

    # Ensure unit_tests is a list
    if not isinstance(unit_tests, list):
        unit_tests = []

    return canonical_solution, unit_tests


def validate_solution_and_tests(
    canonical_solution: str, unit_tests: List[str]
) -> Dict[str, Any]:
    """
    Validate that the solution and tests are properly formatted.

    Args:
        canonical_solution: The canonical solution code
        unit_tests: List of unit test assertions

    Returns:
        Dictionary with validation results
    """
    validation = {
        "has_solution": bool(canonical_solution and canonical_solution.strip()),
        "has_tests": bool(unit_tests and len(unit_tests) > 0),
        "solution_has_function": "def task_func" in canonical_solution
        if canonical_solution
        else False,
        "tests_have_asserts": all("assert" in test for test in unit_tests)
        if unit_tests
        else False,
        "solution_length": len(canonical_solution) if canonical_solution else 0,
        "test_count": len(unit_tests) if unit_tests else 0,
    }

    return validation


def run_evaluation(entry: Dict) -> Dict[str, Any]:
    """
    Run evaluation for a single ambiguous prompt entry.

    Args:
        entry: Dictionary containing the ambiguous prompt entry

    Returns:
        Dictionary with evaluation results
    """
    # Extract solution and tests
    canonical_solution, unit_tests = extract_canonical_solution_and_tests(entry)

    # Validate
    validation = validate_solution_and_tests(canonical_solution, unit_tests)

    # Run evaluation if we have both solution and tests
    evaluation = None
    if validation["has_solution"] and validation["has_tests"]:
        evaluation = evaluate_code_against_tests(
            code=canonical_solution,
            test_list=unit_tests,
            entry_point="task_func",
            test_imports=[],
        )
    else:
        evaluation = {
            "code": canonical_solution,
            "tests_passed": 0,
            "tests_failed": len(unit_tests) if unit_tests else 0,
            "total_tests": len(unit_tests) if unit_tests else 0,
            "test_results": [],
            "execution_error": "Missing solution or tests",
            "test_imports": [],
        }

    # Create result entry
    result = {
        "problem_id": entry.get("extra_info", {}).get(
            "index", 0
        ),  # Add problem ID for HTML rendering
        "problem_name": f"MBPP-{entry.get('extra_info', {}).get('original_task_id', 'Unknown')}-Ambiguous-{entry.get('extra_info', {}).get('ambiguity_set_index', 0)}-{entry.get('extra_info', {}).get('intent_index', 0)}",  # Descriptive problem name
        "entry_id": entry.get("data_source", "unknown"),
        "prompt": entry.get("prompt", ""),
        "explicit_task": entry.get("extra_info", {}).get("explicit_task", ""),
        "original_intent": entry.get("extra_info", {}).get(
            "explicit_task", ""
        ),  # Add original intent for HTML rendering
        "canonical_solution": canonical_solution,
        "generated_code": canonical_solution,  # Add generated_code for HTML visualization compatibility
        "unit_tests": unit_tests,
        "validation": validation,
        "evaluation": evaluation,
        "ambiguity_type": entry.get("extra_info", {}).get("ambiguity_type", "unknown"),
        "original_task": entry.get("extra_info", {}).get("original_task", ""),
        "original_task_id": entry.get("extra_info", {}).get(
            "original_task_id", "unknown"
        ),
    }

    return result


def create_evaluation_summary(results: List[Dict]) -> Dict[str, Any]:
    """
    Create a summary of evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with summary statistics
    """
    total_entries = len(results)

    # Count validation results
    valid_solutions = sum(1 for r in results if r["validation"]["has_solution"])
    valid_tests = sum(1 for r in results if r["validation"]["has_tests"])
    valid_function_defs = sum(
        1 for r in results if r["validation"]["solution_has_function"]
    )
    valid_asserts = sum(1 for r in results if r["validation"]["tests_have_asserts"])

    # Count evaluation results
    total_tests = sum(r["evaluation"]["total_tests"] for r in results)
    total_passed = sum(r["evaluation"]["tests_passed"] for r in results)
    total_failed = sum(r["evaluation"]["tests_failed"] for r in results)

    # Calculate pass rates
    overall_pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    # Count entries by ambiguity type
    ambiguity_types = {}
    for r in results:
        amb_type = r["ambiguity_type"]
        ambiguity_types[amb_type] = ambiguity_types.get(amb_type, 0) + 1

    summary = {
        "total_entries": total_entries,
        "validation": {
            "valid_solutions": valid_solutions,
            "valid_tests": valid_tests,
            "valid_function_defs": valid_function_defs,
            "valid_asserts": valid_asserts,
            "solution_rate": (valid_solutions / total_entries * 100)
            if total_entries > 0
            else 0,
            "test_rate": (valid_tests / total_entries * 100)
            if total_entries > 0
            else 0,
            "function_def_rate": (valid_function_defs / total_entries * 100)
            if total_entries > 0
            else 0,
            "assert_rate": (valid_asserts / total_entries * 100)
            if total_entries > 0
            else 0,
        },
        "evaluation": {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_pass_rate": overall_pass_rate,
        },
        "ambiguity_types": ambiguity_types,
    }

    return summary


def print_evaluation_summary(summary: Dict[str, Any]):
    """
    Print a formatted summary of evaluation results.

    Args:
        summary: Dictionary with summary statistics
    """
    print("\n" + "=" * 60)
    print("📊 AMBIGUOUS PROMPTS EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\n📋 Total Entries: {summary['total_entries']}")

    print(f"\n🔍 Validation Results:")
    val = summary["validation"]
    print(
        f"  Solutions Generated: {val['valid_solutions']}/{summary['total_entries']} ({val['solution_rate']:.1f}%)"
    )
    print(
        f"  Tests Generated: {val['valid_tests']}/{summary['total_entries']} ({val['test_rate']:.1f}%)"
    )
    print(
        f"  Function Definitions: {val['valid_function_defs']}/{summary['total_entries']} ({val['function_def_rate']:.1f}%)"
    )
    print(
        f"  Assert Statements: {val['valid_asserts']}/{summary['total_entries']} ({val['assert_rate']:.1f}%)"
    )

    print(f"\n🧪 Test Execution Results:")
    eval_results = summary["evaluation"]
    print(f"  Total Tests: {eval_results['total_tests']}")
    print(f"  Tests Passed: {eval_results['total_passed']}")
    print(f"  Tests Failed: {eval_results['total_failed']}")
    print(f"  Overall Pass Rate: {eval_results['overall_pass_rate']:.1f}%")

    print(f"\n🎯 Ambiguity Types:")
    for amb_type, count in summary["ambiguity_types"].items():
        percentage = (
            (count / summary["total_entries"] * 100)
            if summary["total_entries"] > 0
            else 0
        )
        print(f"  {amb_type}: {count} ({percentage:.1f}%)")

    print("=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate generated ambiguous prompts")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the JSONL file with ambiguous prompts",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ambiguous_evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=None,
        help="Maximum number of entries to evaluate (for testing)",
    )

    args = parser.parse_args()

    print("=== Ambiguous Prompts Evaluation ===\n")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return False

    # Load ambiguous data
    entries = load_ambiguous_data(args.input_file)

    if not entries:
        print("❌ No entries found in the input file!")
        return False

    # Limit entries if specified
    if args.max_entries and args.max_entries < len(entries):
        entries = entries[: args.max_entries]
        print(f"Limited evaluation to {len(entries)} entries for testing")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run evaluation
    print(f"\n🚀 Running evaluation on {len(entries)} entries...")
    results = []

    for i, entry in enumerate(entries):
        print(
            f"  Evaluating entry {i + 1}/{len(entries)}: {entry.get('data_source', 'unknown')}"
        )

        result = run_evaluation(entry)
        results.append(result)

        # Print progress
        if result["evaluation"]["total_tests"] > 0:
            pass_rate = (
                result["evaluation"]["tests_passed"]
                / result["evaluation"]["total_tests"]
                * 100
            )
            print(
                f"    Tests: {result['evaluation']['tests_passed']}/{result['evaluation']['total_tests']} passed ({pass_rate:.1f}%)"
            )
        else:
            print(f"    ⚠️  No tests to run")

    if not results:
        print("❌ No results generated from evaluation!")
        return False

    # Create summary
    summary = create_evaluation_summary(results)
    print_evaluation_summary(summary)

    # Save results
    input_filename = Path(args.input_file).stem
    output_file = output_dir / f"{input_filename}_evaluation.jsonl"
    json_file = output_dir / f"{input_filename}_evaluation.json"
    html_file = output_dir / f"{input_filename}_evaluation.html"

    # Save as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    # Save as JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(f"\n✅ Results saved to:")
    print(f"  JSONL: {output_file}")
    print(f"  JSON: {json_file}")

    # Generate HTML visualization
    create_mbpp_html_visualization(results, html_file)
    print(f"🎨 HTML visualization saved to {html_file}")

    print(f"\n✨ Ambiguous prompts evaluation completed successfully!")
    return True


if __name__ == "__main__":
    # Example usage:
    # python evaluate_ambiguous_prompts.py --input_file data/mbpp_subtly_ambiguous.jsonl --max_entries 10
    main()
