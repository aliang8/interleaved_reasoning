#!/usr/bin/env python3
"""
Simplified HTML visualization for vLLM rollout results.
Shows only prompts and generated answers/interleaved content.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import os


def load_qwen3_tokenizer():
    """
    Load Qwen3 tokenizer for token counting.

    Returns:
        Tokenizer object or None if loading fails
    """
    try:
        from transformers import AutoTokenizer

        # Load Qwen3-8B tokenizer
        model_name = "Qwen/Qwen3-8B"
        try:
            print(f"Loading tokenizer from {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            print(f"✅ Successfully loaded tokenizer from {model_name}")
            return tokenizer
        except Exception as e:
            print(f"❌ Failed to load from {model_name}: {e}")

            # Fallback: try to load from local path if specified in environment
            local_model_path = os.environ.get("QWEN3_MODEL_PATH")
            if local_model_path:
                try:
                    print(
                        f"Attempting to load tokenizer from local path: {local_model_path}"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        local_model_path, trust_remote_code=True
                    )
                    print(
                        f"✅ Successfully loaded tokenizer from local path: {local_model_path}"
                    )
                    return tokenizer
                except Exception as e2:
                    print(f"❌ Failed to load from local path: {e2}")

            print("⚠️  Could not load Qwen3 tokenizer. Token counting will be disabled.")
            return None

    except ImportError:
        print(
            "⚠️  transformers library not available. Install with: pip install transformers"
        )
        return None
    except Exception as e:
        print(f"⚠️  Error loading tokenizer: {e}")
        return None

    except ImportError:
        print(
            "⚠️  transformers library not available. Install with: pip install transformers"
        )
        return None
    except Exception as e:
        print(f"⚠️  Error loading tokenizer: {e}")
        return None


def count_tokens_for_component(text: str, tokenizer) -> int:
    """
    Count tokens for a given text using the tokenizer.

    Args:
        text: Text to count tokens for
        tokenizer: Tokenizer object

    Returns:
        Number of tokens
    """
    if not tokenizer or not text:
        return 0

    try:
        # Encode the text and count tokens
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception as e:
        print(f"⚠️  Error counting tokens: {e}")
        return 0


def create_simple_html_visualization(
    results: List[Dict], output_file: str, tokenizer=None
):
    """
    Create simplified HTML visualization showing only prompts and answers.

    Args:
        results: List of generation results with 'question' and 'response' keys
        output_file: Path to save the HTML file
        tokenizer: Optional tokenizer for counting tokens in components
    """
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vLLM Rollout Results - Simplified View</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .result-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .result-header {
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        .prompt-section {
            background: #ecf0f1;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .prompt-label {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .prompt-text {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            white-space: pre-wrap;
            line-height: 1.5;
            color: #2c3e50;
        }
        .response-section {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
        }
        .response-label {
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .response-text {
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            line-height: 1.4;
            color: #495057;
            background: white;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #e9ecef;
            max-height: 600px;
            overflow-y: auto;
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
        .component-title {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .token-badge {
            background: #17a2b8;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
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
            font-family: 'Courier New', monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.4;
            max-height: 400px;
            overflow-y: auto;
            margin: 0;
        }
        .component-content.collapsed {
            display: none;
        }
        .component-content * {
            margin: 0;
            padding: 0;
        }
        .component-content p:first-child {
            margin-top: 0;
        }
        .component-content p:last-child {
            margin-bottom: 0;
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
        .token-count {
            font-size: 18px;
            font-weight: bold;
            color: #17a2b8;
        }
        .component-token-summary {
            background: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>🚀 vLLM Rollout Results - Simplified View</h1>
    <p>This visualization shows the prompts and generated responses from the vLLM rollout.</p>
    
"""

    # Calculate summary statistics
    total_results = len(results)
    total_tokens = sum(result.get("num_tokens", 0) for result in results)
    avg_tokens = total_tokens / total_results if total_results > 0 else 0

    html_content += f"""
    <div class="summary-stats">
        <h2>📊 Summary</h2>
        <div class="metrics-grid">
            <div class="metric-item">
                <strong>Total Results:</strong> {total_results}
            </div>
            <div class="metric-item">
                <strong>Total Tokens:</strong> {total_tokens:,}
            </div>
            <div class="metric-item">
                <strong>Average Tokens:</strong> <span class="token-count">{avg_tokens:.1f}</span>
            </div>
        </div>
    </div>
"""

    # Generate result details
    for i, result in enumerate(results):
        question = result.get("question", "No question provided")
        response = result.get("response", "No response generated")
        num_tokens = result.get("num_tokens", 0)

        html_content += f"""    <div class="result-container">
        <div class="result-header">
            <h2>Result {i + 1}</h2>
        </div>
        
        <div class="prompt-section">
            <div class="prompt-label">📝 Prompt</div>
            <div class="prompt-text">{question}</div>
        </div>
        
        <div class="response-section">
            <div class="response-label">💡 Response ({num_tokens:,} tokens)</div>
"""

        # Check if response has interleaved components (think/answer)
        from helpers import parse_interleaved_components

        interleaved_components = parse_interleaved_components(response)

        if interleaved_components:
            # Calculate token counts for each component if tokenizer is available
            component_token_counts = []
            total_component_tokens = 0

            for component in interleaved_components:
                if tokenizer:
                    token_count = count_tokens_for_component(
                        component["content"], tokenizer
                    )
                    component_token_counts.append(token_count)
                    total_component_tokens += token_count
                else:
                    component_token_counts.append(0)

            html_content += f"""            <div class="interleaved-flow">
                <h4>🔄 Generation Flow ({len(interleaved_components)} components)</h4>
"""

            # Add component token summary if tokenizer is available
            if tokenizer and total_component_tokens > 0:
                html_content += f"""                <div class="component-token-summary">
                    <strong>📊 Component Token Summary:</strong> Total: {total_component_tokens:,} tokens
                </div>
"""

            for component_idx, component in enumerate(interleaved_components):
                token_count = (
                    component_token_counts[component_idx]
                    if component_token_counts
                    else 0
                )
                token_display = f" ({token_count:,} tokens)" if token_count > 0 else ""

                if component["type"] == "think":
                    html_content += f"""                <div class="interleaved-section think">
                    <div class="component-header">
                        <div class="component-title">
                            <strong>💭 Think {component["index"]}</strong>
                            {f'<span class="token-badge">{token_count:,} tokens</span>' if token_count > 0 else ""}
                        </div>
                        <button class="toggle-component" onclick="toggleComponent({i}, {component_idx})">▼</button>
                    </div>
                    <div class="component-content" id="component-{i}-{component_idx}">{component["content"]}</div>
                </div>
"""
                else:  # answer
                    html_content += f"""                <div class="interleaved-section answer">
                    <div class="component-header">
                        <div class="component-title">
                            <strong>💡 Answer {component["index"]}</strong>
                            {f'<span class="token-badge">{token_count:,} tokens</span>' if token_count > 0 else ""}
                        </div>
                        <button class="toggle-component" onclick="toggleComponent({i}, {component_idx})">▼</button>
                    </div>
                    <div class="component-content" id="component-{i}-{component_idx}">{component["content"]}</div>
                </div>
"""
            html_content += """            </div>
"""
        else:
            # No interleaved components, show raw response
            html_content += f"""            <div class="response-text">{response}</div>
"""

        html_content += """        </div>
    </div>
"""

    html_content += """
    
    <script>
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

    print(f"Generated simplified HTML visualization with {len(results)} results")


def main():
    """Main function to create simplified HTML visualization."""
    parser = argparse.ArgumentParser(
        description="Create simplified HTML visualization for vLLM rollout results"
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
        default="simple_rollout_visualization.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--disable_token_counting",
        action="store_true",
        help="Disable token counting for interleaved components",
    )

    args = parser.parse_args()

    print("=== Simple vLLM Rollout HTML Visualization Generator ===\n")
    print(f"Results file: {args.results_path}")
    print(f"Output file: {args.output_file}")
    print(
        f"Token counting: {'Disabled' if args.disable_token_counting else 'Enabled (default)'}"
    )

    try:
        # Load Qwen3 tokenizer for token counting (enabled by default)
        tokenizer = None
        if not args.disable_token_counting:
            print("\n🔧 Loading Qwen3 tokenizer for component token counting...")
            tokenizer = load_qwen3_tokenizer()
            if tokenizer:
                print("✅ Tokenizer loaded successfully!")
            else:
                print("⚠️  Tokenizer loading failed. Continuing without token counting.")
        else:
            print("\n⚠️  Token counting disabled by user request.")

        # Load results
        results = load_results_from_file(args.results_path)

        if not results:
            print("❌ No results found in the file!")
            return False

        # Generate HTML visualization
        create_simple_html_visualization(results, args.output_file, tokenizer)

        print(f"\n✅ Simplified HTML visualization created successfully!")
        print(f"📁 Output file: {args.output_file}")

        return True

    except Exception as e:
        print(f"❌ Error creating HTML visualization: {e}")
        import traceback

        traceback.print_exc()
        return False


def load_results_from_file(file_path: str) -> List[Dict]:
    """
    Load results from either JSON or JSONL file.

    Args:
        file_path: Path to the results file

    Returns:
        List of generation results
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


if __name__ == "__main__":
    main()
