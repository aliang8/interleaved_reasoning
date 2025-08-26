#!/usr/bin/env python3
"""
Base HTML visualization module with shared components for all evaluators.
"""

import re
from typing import List, Dict, Any


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
    
    return ""


def get_shared_css() -> str:
    """Get shared CSS styles for all HTML visualizations."""
    return """
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
        .pass-rate, .pass-at-1, .accuracy-rate {
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
        .answer-correctness {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .answer-correctness.correct {
            background: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        .answer-correctness.incorrect {
            background: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        .answer-details {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }
        .answer-option {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 10px;
            margin: 5px 0;
            font-family: monospace;
            font-size: 13px;
        }
        .answer-option.selected {
            background: #e3f2fd;
            border-color: #2196f3;
            color: #1565c0;
        }
        .answer-option.correct {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .answer-option.incorrect {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
    """


def get_shared_javascript() -> str:
    """Get shared JavaScript functions for all HTML visualizations."""
    return """
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
        
        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize all interleaved components as collapsed
            const componentContents = document.querySelectorAll('.component-content');
            componentContents.forEach((content, index) => {
                content.classList.add('collapsed');
            });
        });
    """


def render_interleaved_components(interleaved_components: List[Dict], problem_index: int, template_type: str) -> str:
    """Render interleaved thinking/answering components."""
    if not interleaved_components or len(interleaved_components) == 0:
        return ""
    
    html = f"""
            <div class="interleaved-flow">
                <h4>🔄 Generation Flow ({len(interleaved_components)} components)</h4>
"""
    
    for component_idx, component in enumerate(interleaved_components):
        if component["type"] == "think":
            html += f"""
                <div class="interleaved-section think">
                    <div class="component-header">
                        <strong>💭 Think {component["index"]}</strong>
                        <button class="toggle-component" onclick="toggleComponent({problem_index}, {component_idx})">▼</button>
                    </div>
                    <div class="component-content" id="component-{problem_index}-{component_idx}">
                        <div class="think-content">{component["content"]}</div>
                    </div>
                </div>
"""
        else:  # answer
            html += f"""
                <div class="interleaved-section answer">
                    <div class="component-header">
                        <strong>💡 Answer {component["index"]}</strong>
                        <button class="toggle-component" onclick="toggleComponent({problem_index}, {component_idx})">▼</button>
                    </div>
                    <div class="component-content" id="component-{problem_index}-{component_idx}">
                        <div class="answer-content">{component["content"]}</div>
                    </div>
                </div>
"""
    
    html += """
            </div>
"""
    return html


def render_thinking_section(full_response: str) -> str:
    """Render thinking section for default template responses."""
    thinking_text = extract_thinking_from_full_response(full_response)
    if not thinking_text:
        return ""
    
    return f"""
            <div class="thinking-section">
                <h4>💭 Thinking</h4>
                <div class="thinking-text">{thinking_text}</div>
            </div>
"""


def render_problem_metrics(result: Dict) -> str:
    """Render problem metrics badges."""
    return f"""
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
""" 