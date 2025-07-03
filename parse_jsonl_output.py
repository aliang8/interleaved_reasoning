#!/usr/bin/env python3
"""
Parse JSONL files and display formatted prompts and responses.
Useful for reviewing validation rollout generation outputs.

Usage: python parse_jsonl_output.py <jsonl_file> [--max_entries N] [--show_metadata]
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_colored(text: str, color: str = Colors.END):
    """Print text with color."""
    print(f"{color}{text}{Colors.END}")


def print_separator(char: str = "=", length: int = 80, color: str = Colors.BLUE):
    """Print a colored separator line."""
    print_colored(char * length, color)


def extract_prompt(entry: Dict[str, Any]) -> Optional[str]:
    """Extract prompt from various possible formats in the JSONL entry."""
    
    # Try different common field names for prompts
    if "prompt" in entry:
        return entry["prompt"]
    
    if "input" in entry:
        return entry["input"]
    
    if "question" in entry:
        return entry["question"]
    
    # Try to extract from messages format
    if "messages" in entry:
        messages = entry["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            # Look for user message
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            # If no user message, return the last message content
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                return last_msg.get("content", "")
    
    # Try to extract from conversation format
    if "conversation" in entry:
        conv = entry["conversation"]
        if isinstance(conv, list) and len(conv) > 0:
            return conv[0].get("content", "")
    
    return None


def extract_response(entry: Dict[str, Any]) -> Optional[str]:
    """Extract response from various possible formats in the JSONL entry."""
    
    # Try different common field names for responses
    response_fields = ["generated_response", "answer", "response"]
    
    for field in response_fields:
        if field in entry:
            response = entry[field]
            if isinstance(response, str):
                return response
            elif isinstance(response, list) and len(response) > 0:
                # Handle list of responses (take first one)
                return str(response[0])
    
    # Try to extract from responses list
    if "responses" in entry:
        responses = entry["responses"]
        if isinstance(responses, list) and len(responses) > 0:
            first_response = responses[0]
            if isinstance(first_response, str):
                return first_response
            elif isinstance(first_response, dict):
                # Try common response fields in the response object
                for field in ["text", "content", "response", "output"]:
                    if field in first_response:
                        return str(first_response[field])
    
    # Try to extract from outputs
    if "outputs" in entry:
        outputs = entry["outputs"]
        if isinstance(outputs, list) and len(outputs) > 0:
            return str(outputs[0])

    if "raw_response" in entry:
        return entry["raw_response"]

    
    return None


def extract_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Extract interesting metadata from the entry."""
    metadata = {}
    
    # Common metadata fields
    metadata_fields = [
        "model", "temperature", "max_tokens", "timestamp", 
        "step", "num_trajectories", "system_template_type",
        "generation_time", "tokens_generated", "category",
        "generation_method", "raw_response", "rubric_response"
    ]
    
    for field in metadata_fields:
        if field in entry:
            metadata[field] = entry[field]
    
    # Extract from nested config or params
    if "config" in entry:
        config = entry["config"]
        if isinstance(config, dict):
            for field in metadata_fields:
                if field in config:
                    metadata[field] = config[field]
    
    if "params" in entry:
        params = entry["params"]
        if isinstance(params, dict):
            for field in metadata_fields:
                if field in params:
                    metadata[field] = params[field]
    
    return metadata


def format_text(text: str, max_width: int = 100) -> str:
    """Format text with word wrapping and preserve structure."""
    if not text:
        return ""
    
    # Preserve existing line breaks
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        if len(line) <= max_width:
            formatted_lines.append(line)
        else:
            # Simple word wrapping
            words = line.split(' ')
            current_line = ""
            for word in words:
                if len(current_line + " " + word) <= max_width:
                    current_line += (" " + word) if current_line else word
                else:
                    if current_line:
                        formatted_lines.append(current_line)
                    current_line = word
            if current_line:
                formatted_lines.append(current_line)
    
    return '\n'.join(formatted_lines)


def display_entry(entry: Dict[str, Any], index: int, show_metadata: bool = False):
    """Display a single JSONL entry with formatting."""
    
    print_separator("=", 80, Colors.BLUE)
    print_colored(f"Entry #{index + 1}", Colors.BOLD + Colors.CYAN)
    print_separator("-", 80, Colors.CYAN)
    
    # Extract and display prompt
    prompt = extract_prompt(entry)
    if prompt:
        print_colored("PROMPT:", Colors.BOLD + Colors.GREEN)
        print(format_text(prompt))
        print()
    else:
        print_colored("⚠️  No prompt found in entry", Colors.YELLOW)
        print()
    
    # Extract and display response
    response = extract_response(entry)
    if response:
        print_colored("RESPONSE:", Colors.BOLD + Colors.GREEN)
        print(format_text(response))
        print()
    else:
        print_colored("⚠️  No response found in entry", Colors.YELLOW)
        print()
    
    # Display metadata if requested
    if show_metadata:
        metadata = extract_metadata(entry)
        if metadata:
            print_colored("METADATA:", Colors.BOLD + Colors.YELLOW)
            for key, value in metadata.items():
                print(f"  {key}: {value}")
            print()
    
    # Show available fields for debugging
    available_fields = list(entry.keys())
    print_colored(f"Available fields: {', '.join(available_fields)}", Colors.CYAN)
    print()


def parse_jsonl_file(file_path: str, max_entries: Optional[int] = None, show_metadata: bool = False):
    """Parse and display contents of a JSONL file."""
    
    if not Path(file_path).exists():
        print_colored(f"❌ File not found: {file_path}", Colors.RED)
        return
    
    print_colored(f"📁 Parsing JSONL file: {file_path}", Colors.BOLD + Colors.BLUE)
    print()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            entries = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print_colored(f"⚠️  Skipping invalid JSON on line {line_num}: {e}", Colors.YELLOW)
                    continue
                
                # Stop if we've reached the maximum
                if max_entries and len(entries) >= max_entries:
                    break
        
        print_colored(f"✅ Loaded {len(entries)} entries from {file_path}", Colors.GREEN)
        
        if not entries:
            print_colored("No valid entries found in file.", Colors.YELLOW)
            return
        
        print()
        
        # Display entries
        for i, entry in enumerate(entries):
            display_entry(entry, i, show_metadata)
            
            # Add pause for large outputs
            if i > 0 and (i + 1) % 5 == 0 and i + 1 < len(entries):
                print_colored(f"--- Displayed {i + 1}/{len(entries)} entries ---", Colors.CYAN)
                if max_entries is None or len(entries) > 10:
                    response = input("Press Enter to continue, 'q' to quit: ")
                    if response.lower() == 'q':
                        break
                print()
        
        print_separator("=", 80, Colors.BLUE)
        print_colored(f"✅ Finished displaying {min(i + 1, len(entries))} entries", Colors.BOLD + Colors.GREEN)
        
    except Exception as e:
        print_colored(f"❌ Error reading file: {e}", Colors.RED)


def main():
    parser = argparse.ArgumentParser(
        description="Parse JSONL files and display formatted prompts and responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_jsonl_output.py output.jsonl
  python parse_jsonl_output.py results.jsonl --max_entries 5
  python parse_jsonl_output.py data.jsonl --show_metadata
        """
    )
    
    parser.add_argument("jsonl_file", help="Path to the JSONL file to parse")
    parser.add_argument("--max_entries", "-n", type=int, default=None,
                       help="Maximum number of entries to display (default: all)")
    parser.add_argument("--show_metadata", "-m", action="store_true",
                       help="Show metadata/configuration information for each entry")
    parser.add_argument("--no_color", action="store_true",
                       help="Disable colored output")
    
    args = parser.parse_args()
    
    # Disable colors if requested or if output is redirected
    if args.no_color or not sys.stdout.isatty():
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    parse_jsonl_file(args.jsonl_file, args.max_entries, args.show_metadata)


if __name__ == "__main__":
    main() 