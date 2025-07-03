#!/usr/bin/env python3
"""
Script to generate rubrics from BigCodeBench parquet files.

This script:
1. Loads a parquet file (e.g., from BigCodeBench)
2. Extracts <answer></answer> tags from the interleaved responses
3. Concatenates them to create reference answers
4. Uses the question as instruction to generate rubrics via Gemini API
5. Saves rubrics in the same folder as the parquet file
"""

import os
import json
import argparse
import pandas as pd
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from gemini_batch_query import GeminiAPIClient

try:
    from per_instance_rubric_template import generate_rubric_for_prompt
    RUBRIC_TEMPLATE_AVAILABLE = True
except ImportError:
    RUBRIC_TEMPLATE_AVAILABLE = False
    print("Warning: per_instance_rubric_template.py not available. Using basic rubric template.")


def extract_answer_tags(text: str) -> List[str]:
    """Extract all <answer></answer> tag contents from text."""
    if not text:
        return []
    
    # Find all answer tags (case insensitive, multiline)
    pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    # Clean up the matches (strip whitespace)
    return [match.strip() for match in matches if match.strip()]



def create_reference_answer(answer_texts: List[str]) -> str:
    """Create a reference answer by concatenating answer tag contents."""
    if not answer_texts:
        return ""
    
    # Concatenate all answers with their tags
    reference_parts = []
    for i, answer_text in enumerate(answer_texts, 1):
        if len(answer_texts) > 1:
            reference_parts.append(f"<answer>{answer_text}</answer>")
        else:
            reference_parts.append(f"<answer>{answer_text}</answer>")
    
    return "\n\n".join(reference_parts)


def generate_basic_rubric_prompt(instruction: str, reference_answer: str) -> str:
    """Generate a basic rubric creation prompt if template is not available."""
    return f"""Create a detailed evaluation rubric in Python dictionary format to assess AI responses for the following coding task that uses an interleaved reasoning format.

**Instruction:** {instruction}

**Reference Answer:** {reference_answer}

The AI response should demonstrate structured interleaved reasoning with multiple <answer></answer> tags, each containing meaningful and useful content. The focus is on STRUCTURE and CONTENT USEFULNESS rather than code correctness.

Expected structure with multiple <answer></answer> tags containing:
1. **Problem Analysis**: Clear understanding, breakdown, and approach planning
2. **Implementation**: Code or solution steps (doesn't need to be perfect)
3. **Validation/Testing**: Test cases, examples, or verification approach

Please create a rubric that evaluates how well an AI model follows this structured reasoning format. The rubric should be returned as a Python dictionary with the following exact format:

```python
rubric_data = {{
  "criteria": "Clear description of what skill/ability is being evaluated",
  "score1_description": "Description of poor performance (score 1)",
  "score2_description": "Description of below average performance (score 2)", 
  "score3_description": "Description of average performance (score 3)",
  "score4_description": "Description of good performance (score 4)",
  "score5_description": "Description of excellent performance (score 5)"
}}
```

The rubric should primarily evaluate:
- **Structure Adherence**: Presence and proper use of multiple <answer></answer> tags
- **Content Usefulness**: Each answer tag contains meaningful, relevant content that serves its purpose
- **Reasoning Progression**: Logical flow from analysis to implementation to validation
- **Tag Distinctiveness**: Each answer tag has a clear, distinct purpose and adds value
- **Completeness**: Coverage of the key reasoning components (analysis, solution, validation)

IMPORTANT: Focus on structure and reasoning process, NOT on whether the code actually works or is bug-free. A response with multiple useful answer tags and good reasoning structure should score well even if the code has minor issues.

Return only the Python dictionary, nothing else."""


def load_parquet_file(file_path: str) -> pd.DataFrame:
    """Load and validate parquet file."""
    try:
        df = pd.read_parquet(file_path)
        print(f"✓ Loaded {len(df)} rows from {file_path}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"✗ Error loading parquet file: {e}")
        raise


def process_parquet_data(df: pd.DataFrame, instruction_col: str = "question", answer_col: str = "answer") -> List[Dict[str, Any]]:
    """Process parquet data to extract instructions and reference answers."""
    results = []
    
    print(f"Processing {len(df)} rows...")
    print(f"Using instruction column: '{instruction_col}'")
    print(f"Using answer column: '{answer_col}'")
    
    # Validate columns exist
    if instruction_col not in df.columns:
        raise ValueError(f"Instruction column '{instruction_col}' not found. Available: {list(df.columns)}")
    if answer_col not in df.columns:
        raise ValueError(f"Answer column '{answer_col}' not found. Available: {list(df.columns)}")
    
    for idx, row in df.iterrows():
        instruction = str(row[instruction_col]).strip()
        answer_text = str(row[answer_col]).strip()
        
        # Extract answer tags
        answer_tags = extract_answer_tags(answer_text)
        
        if not instruction:
            print(f"Warning: Empty instruction at row {idx}, skipping")
            continue
            
        if not answer_tags:
            print(f"Warning: No <answer> tags found at row {idx}, using full answer text")
            reference_answer = answer_text
        else:
            reference_answer = create_reference_answer(answer_tags)
        
        results.append({
            "row_index": idx,
            "instruction": instruction,
            "full_answer_text": answer_text,
            "extracted_answers": answer_tags,
            "reference_answer": reference_answer,
            "answer_tag_count": len(answer_tags)
        })
    
    print(f"✓ Processed {len(results)} valid rows")
    
    # Show statistics
    total_with_tags = sum(1 for r in results if r["answer_tag_count"] > 0)
    interleaved_responses = sum(1 for r in results if r["answer_tag_count"] > 1)
    
    print(f"Rows with <answer> tags: {total_with_tags}/{len(results)}")
    print(f"Interleaved responses (>1 tag): {interleaved_responses}/{len(results)}")
    
    if total_with_tags > 0:
        avg_tags = sum(r["answer_tag_count"] for r in results) / len(results)
        print(f"Average <answer> tags per row: {avg_tags:.1f}")
    
    return results


def generate_rubrics(processed_data: List[Dict[str, Any]], 
                    api_key: str, 
                    model: str = "gemini-2.0-flash",
                    max_samples: Optional[int] = None,
                    delay: float = 1.0) -> List[Dict[str, Any]]:
    """Generate rubrics using Gemini API."""
    
    # Initialize REST API client (don't need thinking mode for rubric generation)
    client = GeminiAPIClient(api_key, model, enable_thinking=False, use_genai_client=False)
    
    # Limit samples if specified
    if max_samples and len(processed_data) > max_samples:
        print(f"Limiting to {max_samples} samples (out of {len(processed_data)})")
        processed_data = processed_data[:max_samples]
    
    rubric_results = []
    
    for i, data in enumerate(processed_data, 1):
        print(f"\n📝 Generating rubric {i}/{len(processed_data)}")
        print(f"Instruction: {data['instruction'][:100]}{'...' if len(data['instruction']) > 100 else ''}")
        print(f"Reference answer length: {len(data['reference_answer'])} chars")
        print(f"Answer tags found: {data['answer_tag_count']}")
        
        try:
            # Generate rubric prompt
            if RUBRIC_TEMPLATE_AVAILABLE:
                rubric_prompt = generate_rubric_for_prompt(data['instruction'], data['reference_answer'])
            else:
                rubric_prompt = generate_basic_rubric_prompt(data['instruction'], data['reference_answer'])
            
            # Make API call
            response = client.generate_content(
                rubric_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Extract response
            rubric_text = client.extract_text_response(response)
            
            # Store result
            result = {
                "row_index": data["row_index"],
                "instruction": data["instruction"],
                "reference_answer": data["reference_answer"],
                "answer_tag_count": data["answer_tag_count"],
                "rubric_prompt": rubric_prompt,
                "rubric_response": rubric_text,
                "model": model,
                "raw_api_response": response,
                "generation_success": "error" not in response
            }            
            rubric_results.append(result)
            
            print(f"✓ Generated rubric: {rubric_text[:150]}{'...' if len(rubric_text) > 150 else ''}")
            
            # Rate limiting
            if i < len(processed_data):
                print(f"⏱️  Waiting {delay}s...")
                time.sleep(delay)
                
        except Exception as e:
            print(f"✗ Error generating rubric for row {data['row_index']}: {e}")
            error_result = {
                "row_index": data["row_index"],
                "instruction": data["instruction"],
                "reference_answer": data["reference_answer"],
                "answer_tag_count": data["answer_tag_count"],
                "error": str(e),
                "generation_success": False
            }
            rubric_results.append(error_result)
    
    
    return rubric_results


def save_rubrics(rubric_results: List[Dict[str, Any]], output_file: str):
    """Save rubric results to JSONL file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in rubric_results:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + '\n')
        
        print(f"✓ Saved {len(rubric_results)} rubrics to {output_file}")
        
        # Print summary
        successful = sum(1 for r in rubric_results if r.get("generation_success", False))
        failed = len(rubric_results) - successful
        print(f"Successful: {successful}, Failed: {failed}")
        
    except Exception as e:
        print(f"✗ Error saving rubrics: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate rubrics from BigCodeBench parquet files")
    parser.add_argument("parquet_file", type=str, help="Path to parquet file")
    parser.add_argument("--api_key", type=str, help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--instruction_col", type=str, default="question", 
                       help="Column name for instructions (default: question)")
    parser.add_argument("--answer_col", type=str, default="answer",
                       help="Column name for answers (default: answer)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                       choices=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash"],
                       help="Gemini model to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (default: all)")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between API calls in seconds")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file name (default: auto-generated based on input file)")
    
    args = parser.parse_args()
    
    # Validate input file
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        print(f"✗ Parquet file not found: {args.parquet_file}")
        return
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("✗ Error: API key required. Use --api_key or set GEMINI_API_KEY environment variable")
        return
    
    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = parquet_path.parent / f"{parquet_path.stem}_rubrics.jsonl"
    
    print("🔬 BigCodeBench Rubric Generator")
    print("=" * 50)
    print(f"Input file: {args.parquet_file}")
    print(f"Output file: {output_file}")
    print(f"Instruction column: {args.instruction_col}")
    print(f"Answer column: {args.answer_col}")
    print(f"Model: {args.model}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Delay: {args.delay}s")
    print(f"Rubric template available: {RUBRIC_TEMPLATE_AVAILABLE}")
    print("=" * 50)
    
    try:
        # Load parquet file
        df = load_parquet_file(args.parquet_file)
        
        # Process data to extract instructions and reference answers
        processed_data = process_parquet_data(df, args.instruction_col, args.answer_col)
        
        if not processed_data:
            print("✗ No valid data to process")
            return
        
        # Show sample data
        print(f"\n📋 Sample processed data:")
        sample = processed_data[0]
        print(f"Row {sample['row_index']}:")
        print(f"  Instruction: {sample['instruction'][:100]}{'...' if len(sample['instruction']) > 100 else ''}")
        print(f"  Reference answer: {sample['reference_answer'][:200]}{'...' if len(sample['reference_answer']) > 200 else ''}")
        print(f"  Answer tags: {sample['answer_tag_count']}")
        
        # Generate rubrics
        rubric_results = generate_rubrics(
            processed_data, 
            api_key, 
            args.model, 
            args.max_samples, 
            args.delay
        )
        
        # Save results
        save_rubrics(rubric_results, output_file)
        
        print(f"\n✅ Rubric generation complete!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main() 