#!/usr/bin/env python3
"""
Batch query script for Google Gemini API.
Reads prompts from a text file and sends them to Gemini API.
Also supports generating rubric creation prompts from instruction/answer pairs.

Usage:
    # Regular prompts from file
    python gemini_batch_query.py --prompts_file prompts.txt --api_key YOUR_API_KEY
    python gemini_batch_query.py --prompts_file prompts.txt --output_file results.jsonl
    
    # Generate rubric prompts from template
    python gemini_batch_query.py --rubric_mode --api_key YOUR_API_KEY --output_file rubrics.jsonl
    
    # Generate scoring prompts to evaluate responses against rubrics
    python gemini_batch_query.py --scoring_mode --api_key YOUR_API_KEY --output_file scores.jsonl
    
    # Enable thinking mode (requires google-genai library)
    python gemini_batch_query.py --prompts_file prompts.txt --api_key YOUR_API_KEY --enable_thinking
    
    # Enable thinking mode with separate thinking/text output
    python gemini_batch_query.py --prompts_file prompts.txt --api_key YOUR_API_KEY --enable_thinking --thinking_separate
"""

import requests
import json
import argparse
import time
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try to import the Google genai library for thinking support
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Note: google-genai library not available. Thinking mode will use REST API fallback.")

# Import rubric template functionality
try:
    from per_instance_rubric_template import (
        generate_rubric_prompt, 
        SAMPLE_INSTRUCTION_ANSWER_PAIRS,
        generate_rubric_for_prompt,
        score_response,
        get_sample_responses,
        get_instruction_answer_pair
    )
    RUBRIC_TEMPLATE_AVAILABLE = True
except ImportError:
    print("Warning: per_instance_rubric_template.py not found. Rubric mode will not be available.")
    RUBRIC_TEMPLATE_AVAILABLE = False


class GeminiAPIClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", enable_thinking: bool = False, use_genai_client: bool = None):
        """Initialize the Gemini API client.
        
        Args:
            api_key: Google AI API key
            model: Model name to use (e.g., "gemini-2.0-flash", "gemini-2.5-pro")
            enable_thinking: Whether to enable thinking mode (requires google-genai library)
            use_genai_client: Whether to use google-genai client (True) or REST API (False).
                            If None, auto-decides based on thinking mode and library availability.
        """
        self.api_key = api_key
        self.model = model
        self.enable_thinking = enable_thinking
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # Determine which client to use
        if use_genai_client is None:
            # Auto-decide: use genai client if thinking is enabled and available
            use_genai_client = enable_thinking and GENAI_AVAILABLE
        
        self.use_genai = use_genai_client
        self.genai_client = None
        
        if self.use_genai:
            if not GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai library not available but genai client requested. "
                    "Install with: pip install google-genai"
                )
            
            try:
                self.genai_client = genai.Client(api_key=api_key)
                print("✓ Using google-genai library client")
                if enable_thinking:
                    print("✓ Thinking mode enabled with genai client")
                else:
                    print("✓ Standard generation with genai client")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize genai client: {e}")
        else:
            # Initialize REST API session
            self.session = requests.Session()
            self.session.headers.update({
                'Content-Type': 'application/json',
                'X-goog-api-key': api_key
            })
            print("✓ Using REST API client")
            if enable_thinking:
                print("⚠️  Warning: Thinking mode requested but using REST API (thinking not supported)")
                self.enable_thinking = False  # Disable thinking for REST API
    
    def generate_content(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Send a prompt to Gemini and get the response."""
        if self.use_genai:
            return self._generate_content_with_genai(prompt, **kwargs)
        else:
            return self._generate_content_with_rest(prompt, **kwargs)
    
    def _generate_content_with_genai(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate content using the google-genai library with thinking mode support."""
        try:
            # Build generation config
            config_params = {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "max_output_tokens": kwargs.get("max_tokens", 4096),
            }
            
            generation_config = types.GenerateContentConfig(**config_params)
            
            # Add thinking config if enabled (using correct format)
            if self.enable_thinking:
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=2048,
                    include_thoughts=True
                )
            
            # Make the API call
            response = self.genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=generation_config
            )
            
            # Convert to dictionary format similar to REST API
            result = {
                "candidates": [],
                "thinking_enabled": self.enable_thinking,
                "thinking_supported": False,  # Will be set to True if we find thinking content
                "api_method": "genai_library"
            }
            
            for candidate in response.candidates:
                candidate_dict = {
                    "content": {
                        "parts": []
                    }
                }
                
                for part in candidate.content.parts:
                    part_dict = {}
                    if hasattr(part, 'text') and part.text:
                        part_dict["text"] = part.text
                    if hasattr(part, 'thought') and part.thought:
                        part_dict["thought"] = part.thought
                        result["thinking_supported"] = True
                    
                    if part_dict:  # Only add non-empty parts
                        candidate_dict["content"]["parts"].append(part_dict)
                
                result["candidates"].append(candidate_dict)
            
            return result
            
        except Exception as e:
            return {
                "error": f"GenAI library request failed: {str(e)}",
                "api_method": "genai_library"
            }
    
    def _generate_content_with_rest(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate content using the REST API (fallback method)."""
        url = f"{self.base_url}/{self.model}:generateContent"
        
        # Default generation config
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "topP": kwargs.get("top_p", 0.9),
            "maxOutputTokens": kwargs.get("max_tokens", 2048),
        }
        
        # Safety settings (optional)
        safety_settings = kwargs.get("safety_settings", [])
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": generation_config
        }
        
        if safety_settings:
            payload["safetySettings"] = safety_settings
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            result["api_method"] = "rest_api"
            result["thinking_enabled"] = False
            return result
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"REST API request failed: {str(e)}",
                "status_code": getattr(e.response, 'status_code', None),
                "api_method": "rest_api"
            }
    
    def extract_text_response(self, response: Dict[str, Any]) -> str:
        """Extract the text content from Gemini response."""
        if "error" in response:
            return f"Error: {response['error']}"
        
        try:
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    # For thinking mode, we want to include both thoughts and text
                    if response.get("thinking_enabled", False) and response.get("thinking_supported", False):
                        return self._extract_thinking_response(parts)
                    else:
                        return parts[0].get("text", "No text in response")
            return "No valid response found"
        
        except (KeyError, IndexError) as e:
            return f"Error parsing response: {str(e)}"
    
    def _extract_thinking_response(self, parts: List[Dict[str, Any]]) -> str:
        """Extract and format thinking + text response."""
        thoughts = []
        text_responses = []
        
        for part in parts:
            # Check if this part is marked as thinking content
            if part.get("thought", False) and "text" in part:
                # This is thinking content
                thoughts.append(part["text"])
            elif "text" in part and not part.get("thought", False):
                # This is regular text content
                text_responses.append(part["text"])
        
        # Format the response with thinking content
        response_parts = []
        
        if thoughts:
            thinking_content = "\n\n".join(thoughts)
            response_parts.append(f"<thinking>\n{thinking_content}\n</thinking>")
        
        if text_responses:
            text_content = "\n\n".join(text_responses)
            response_parts.append(text_content)
        
        if not response_parts:
            return "No valid response content found"
        
        return "\n\n".join(response_parts)
    
    def extract_thinking_and_text_separately(self, response: Dict[str, Any]) -> Dict[str, str]:
        """Extract thinking and text content separately."""
        if "error" in response:
            return {"error": response["error"], "thinking": "", "text": ""}
        
        try:
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                
                thoughts = []
                text_responses = []
                
                for part in parts:
                    # Check if this part is marked as thinking content
                    if part.get("thought", False) and "text" in part:
                        # This is thinking content
                        thoughts.append(part["text"])
                    elif "text" in part and not part.get("thought", False):
                        # This is regular text content
                        text_responses.append(part["text"])
                
                return {
                    "thinking": "\n\n".join(thoughts),
                    "text": "\n\n".join(text_responses),
                    "has_thinking": len(thoughts) > 0
                }
            
            return {"thinking": "", "text": "", "has_thinking": False}
        
        except (KeyError, IndexError) as e:
            return {"error": f"Error parsing response: {str(e)}", "thinking": "", "text": ""}


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompts from a text file."""
    prompts = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            # Try different delimiters
            if '\n---\n' in content:
                # Prompts separated by ---
                prompts = [p.strip() for p in content.split('\n---\n') if p.strip()]
            elif content.count('\n\n') > content.count('\n') * 0.1:
                # Prompts separated by double newlines
                prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
            else:
                # One prompt per line
                prompts = [line.strip() for line in content.split('\n') if line.strip()]
        
        print(f"✓ Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"✗ Error loading prompts: {e}")
        return []


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to a JSONL file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"✓ Saved {len(results)} results to {output_file}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")


def generate_rubric_prompts() -> List[str]:
    """Generate rubric creation prompts from instruction/answer pairs."""
    if not RUBRIC_TEMPLATE_AVAILABLE:
        print("✗ Rubric template not available. Please ensure per_instance_rubric_template.py exists.")
        return []
    
    prompts = []
    for pair in SAMPLE_INSTRUCTION_ANSWER_PAIRS:
        prompt = generate_rubric_prompt(pair["instruction"], pair["reference_answer"])
        prompts.append(prompt)
    
    print(f"✓ Generated {len(prompts)} rubric creation prompts")
    return prompts


def generate_scoring_prompts() -> List[str]:
    """Generate scoring prompts for sample responses against sample rubrics."""
    if not RUBRIC_TEMPLATE_AVAILABLE:
        print("✗ Rubric template not available. Please ensure per_instance_rubric_template.py exists.")
        return []
    
    prompts = []
    
    # Sample rubric for technical explanation (this would normally come from a rubric generation step)
    sample_rubric = {
        "criteria": "Is the model proficient in explaining complex technical concepts in simple, accessible terms for non-technical audiences?",
        "score1_description": "The explanation is overly technical, uses jargon without explanation, or fails to make the concept understandable to a general audience.",
        "score2_description": "The explanation attempts to simplify but still contains unexplained technical terms or concepts that would confuse non-technical readers.",
        "score3_description": "The explanation is generally accessible but may lack clarity in some areas or miss opportunities for better analogies or examples.",
        "score4_description": "The explanation is clear and accessible, uses good analogies or examples, with only minor areas that could be improved for clarity.",
        "score5_description": "The explanation excellently translates complex technical concepts into simple, relatable terms with effective analogies, examples, and clear progression that any general audience can understand."
    }
    
    # Generate scoring prompts for technical explanation responses
    tech_responses = get_sample_responses("technical_explanation")
    tech_pair = get_instruction_answer_pair("technical_explanation")
    
    for response_type, response_text in tech_responses.items():
        scoring_prompt = score_response(
            sample_rubric, 
            response_text, 
            tech_pair["instruction"]
        )
        prompts.append(scoring_prompt)
    
    print(f"✓ Generated {len(prompts)} scoring prompts for sample responses")
    return prompts


def create_sample_prompts_file(file_path: str):
    """Create a sample prompts file for demonstration."""
    sample_prompts = [
        "Explain how artificial intelligence works in simple terms.",
        "What are the main differences between machine learning and deep learning?",
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing like I'm 10 years old.",
        "What are the ethical considerations of AI development?"
    ]
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for prompt in sample_prompts:
                f.write(prompt + '\n\n')
        print(f"✓ Created sample prompts file: {file_path}")
    except Exception as e:
        print(f"✗ Error creating sample file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch query Gemini API with prompts from file or generate rubrics")
    parser.add_argument("--prompts_file", type=str,
                       help="Path to text file containing prompts (required unless using --rubric_mode)")
    parser.add_argument("--rubric_mode", action="store_true",
                       help="Generate rubric creation prompts from instruction/answer pairs")
    parser.add_argument("--scoring_mode", action="store_true",
                       help="Generate scoring prompts to evaluate sample responses against rubrics")
    parser.add_argument("--api_key", type=str,
                       help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--output_file", type=str, default="gemini_results.jsonl",
                       help="Output file for results (default: gemini_results.jsonl)")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                       choices=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash"],
                       help="Gemini model to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature (0.0-1.0)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum output tokens")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between requests in seconds")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create a sample prompts file and exit")
    parser.add_argument("--enable_thinking", action="store_true",
                       help="Enable thinking mode (requires google-genai library)")
    parser.add_argument("--thinking_separate", action="store_true",
                       help="Save thinking and text content separately in results")
    parser.add_argument("--use_genai_client", action="store_true",
                       help="Use google-genai library client instead of REST API")
    parser.add_argument("--use_rest_api", action="store_true",
                       help="Force use of REST API (overrides --use_genai_client)")
    
    args = parser.parse_args()
    
    # Validate arguments
    mode_count = sum([bool(args.rubric_mode), bool(args.scoring_mode), bool(args.prompts_file)])
    if mode_count != 1:
        print("✗ Error: Specify exactly one of --prompts_file, --rubric_mode, or --scoring_mode")
        parser.print_help()
        return
    
    # Create sample file if requested
    if args.create_sample:
        if not args.prompts_file:
            print("✗ Error: --prompts_file required when using --create_sample")
            return
        create_sample_prompts_file(args.prompts_file)
        return
    
    # Get API key
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("✗ Error: API key required. Use --api_key or set GEMINI_API_KEY environment variable")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        return
    
    # Determine prompts source
    if args.rubric_mode:
        print(f"🎯 RUBRIC GENERATION MODE")
        print(f"Using built-in instruction/answer pairs from template")
        prompts = generate_rubric_prompts()
        if not prompts:
            return
        # Update output file default for rubric mode
        if args.output_file == "gemini_results.jsonl":
            args.output_file = "rubric_results.jsonl"
    elif args.scoring_mode:
        print(f"📊 SCORING MODE")
        print(f"Using sample responses and rubrics for evaluation")
        prompts = generate_scoring_prompts()
        if not prompts:
            return
        # Update output file default for scoring mode
        if args.output_file == "gemini_results.jsonl":
            args.output_file = "scoring_results.jsonl"
    else:
        # Check if prompts file exists
        if not Path(args.prompts_file).exists():
            print(f"✗ Prompts file not found: {args.prompts_file}")
            print(f"Create a sample file with: python {__file__} --create_sample --prompts_file {args.prompts_file}")
            return
    
    print(f"🤖 GEMINI API BATCH QUERY")
    print(f"Model: {args.model}")
    if args.rubric_mode:
        print(f"Mode: Rubric generation from template")
        print(f"Template pairs: {len(SAMPLE_INSTRUCTION_ANSWER_PAIRS) if RUBRIC_TEMPLATE_AVAILABLE else 0}")
    elif args.scoring_mode:
        print(f"Mode: Response scoring evaluation")
        tech_responses = get_sample_responses("technical_explanation") if RUBRIC_TEMPLATE_AVAILABLE else {}
        print(f"Sample responses to score: {len(tech_responses)}")
    else:
        print(f"Prompts file: {args.prompts_file}")
        # Load prompts from file
        prompts = load_prompts_from_file(args.prompts_file)
        if not prompts:
            return
    print(f"Output file: {args.output_file}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Delay: {args.delay}s")
    print(f"Thinking mode: {'Enabled' if args.enable_thinking else 'Disabled'}")
    if args.enable_thinking:
        print(f"Thinking output: {'Separate fields' if args.thinking_separate else 'Combined with response'}")
    
    # Show client configuration
    if args.use_rest_api:
        print(f"API Client: REST API (forced)")
    elif args.use_genai_client:
        print(f"API Client: google-genai library (forced)")
    else:
        print(f"API Client: Auto-decide (genai if thinking enabled and available, otherwise REST)")
    print("=" * 50)
    
    # Determine which client to use
    use_genai_client = None
    if args.use_rest_api:
        use_genai_client = False
    elif args.use_genai_client:
        use_genai_client = True
    # If neither specified, let the client auto-decide based on thinking mode
    
    # Initialize API client
    client = GeminiAPIClient(api_key, args.model, enable_thinking=args.enable_thinking, use_genai_client=use_genai_client)
    
    # Process prompts
    results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n📝 Processing prompt {i}/{len(prompts)}")
        if args.rubric_mode:
            # Show which instruction/answer pair this rubric is for
            pair = SAMPLE_INSTRUCTION_ANSWER_PAIRS[i-1] if RUBRIC_TEMPLATE_AVAILABLE and i <= len(SAMPLE_INSTRUCTION_ANSWER_PAIRS) else {}
            pair_id = pair.get("id", f"pair_{i}")
            print(f"Rubric for: {pair_id}")
            print(f"Instruction: {pair.get('instruction', 'Unknown')[:80]}{'...' if len(pair.get('instruction', '')) > 80 else ''}")
        elif args.scoring_mode:
            # Show which response is being scored
            tech_responses = get_sample_responses("technical_explanation") if RUBRIC_TEMPLATE_AVAILABLE else {}
            response_types = list(tech_responses.keys())
            if i <= len(response_types):
                response_type = response_types[i-1]
                print(f"Scoring: {response_type} response")
                print(f"Response preview: {tech_responses[response_type][:80]}{'...' if len(tech_responses[response_type]) > 80 else ''}")
            else:
                print(f"Scoring response #{i}")
        else:
            print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        try:
            # Make API call
            response = client.generate_content(
                prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            # Extract text response
            text_response = client.extract_text_response(response)
            
            # Store result with additional metadata
            result = {
                "prompt_id": i,
                "prompt": prompt,
                "response": text_response,
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "thinking_enabled": args.enable_thinking,
                "raw_response": response
            }
            
            # Add thinking-specific data if enabled and requested separately
            if args.enable_thinking and args.thinking_separate:
                thinking_data = client.extract_thinking_and_text_separately(response)
                result.update({
                    "thinking_content": thinking_data.get("thinking", ""),
                    "text_content": thinking_data.get("text", ""),
                    "has_thinking": thinking_data.get("has_thinking", False),
                    "thinking_separate": True
                })
                # Update the main response field to contain only text when separate
                if thinking_data.get("text"):
                    result["response"] = thinking_data["text"]
            
            # Add mode-specific metadata
            if args.rubric_mode and RUBRIC_TEMPLATE_AVAILABLE and i <= len(SAMPLE_INSTRUCTION_ANSWER_PAIRS):
                pair = SAMPLE_INSTRUCTION_ANSWER_PAIRS[i-1]
                result.update({
                    "rubric_mode": True,
                    "instruction_answer_pair": {
                        "id": pair["id"],
                        "instruction": pair["instruction"],
                        "reference_answer": pair["reference_answer"]
                    }
                })
            elif args.scoring_mode and RUBRIC_TEMPLATE_AVAILABLE:
                tech_responses = get_sample_responses("technical_explanation")
                tech_pair = get_instruction_answer_pair("technical_explanation")
                response_types = list(tech_responses.keys())
                if i <= len(response_types):
                    response_type = response_types[i-1]
                    result.update({
                        "scoring_mode": True,
                        "response_being_scored": {
                            "type": response_type,
                            "response_text": tech_responses[response_type],
                            "original_instruction": tech_pair.get("instruction", ""),
                            "task_id": "technical_explanation"
                        }
                    })
            
            results.append(result)
            
            print(f"✓ Response: {text_response[:100]}{'...' if len(text_response) > 100 else ''}")
            
            # Rate limiting delay
            if i < len(prompts):  # Don't delay after the last request
                print(f"⏱️  Waiting {args.delay}s...")
                time.sleep(args.delay)
        
        except Exception as e:
            print(f"✗ Error processing prompt {i}: {e}")
            error_result = {
                "prompt_id": i,
                "prompt": prompt,
                "response": f"Error: {str(e)}",
                "model": args.model,
                "thinking_enabled": args.enable_thinking,
                "error": True
            }
            
            # Add mode-specific metadata even for errors
            if args.rubric_mode and RUBRIC_TEMPLATE_AVAILABLE and i <= len(SAMPLE_INSTRUCTION_ANSWER_PAIRS):
                pair = SAMPLE_INSTRUCTION_ANSWER_PAIRS[i-1]
                error_result.update({
                    "rubric_mode": True,
                    "instruction_answer_pair": {
                        "id": pair["id"],
                        "instruction": pair["instruction"],
                        "reference_answer": pair["reference_answer"]
                    }
                })
            elif args.scoring_mode and RUBRIC_TEMPLATE_AVAILABLE:
                tech_responses = get_sample_responses("technical_explanation")
                tech_pair = get_instruction_answer_pair("technical_explanation")
                response_types = list(tech_responses.keys())
                if i <= len(response_types):
                    response_type = response_types[i-1]
                    error_result.update({
                        "scoring_mode": True,
                        "response_being_scored": {
                            "type": response_type,
                            "response_text": tech_responses[response_type],
                            "original_instruction": tech_pair.get("instruction", ""),
                            "task_id": "technical_explanation"
                        }
                    })
            
            results.append(error_result)
    
    # Save results
    if results:
        save_results(results, args.output_file)
        
        # Print summary
        successful = len([r for r in results if not r.get("error", False)])
        failed = len(results) - successful
        
        print(f"\n" + "=" * 50)
        print(f"BATCH PROCESSING COMPLETE")
        print(f"=" * 50)
        if args.rubric_mode:
            print(f"Mode: Rubric generation")
            print(f"Total rubric prompts: {len(prompts)}")
        elif args.scoring_mode:
            print(f"Mode: Response scoring")
            print(f"Total scoring prompts: {len(prompts)}")
        else:
            print(f"Mode: Regular prompts from file")
            print(f"Total prompts: {len(prompts)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Results saved to: {args.output_file}")
        
        if successful > 0:
            print(f"\n📋 Sample result:")
            sample = next(r for r in results if not r.get("error", False))
            if args.rubric_mode and "instruction_answer_pair" in sample:
                print(f"Rubric for: {sample['instruction_answer_pair']['id']}")
                print(f"Instruction: {sample['instruction_answer_pair']['instruction'][:100]}...")
                print(f"Generated rubric: {sample['response'][:200]}...")
            elif args.scoring_mode and "response_being_scored" in sample:
                print(f"Score for: {sample['response_being_scored']['type']} response")
                print(f"Task: {sample['response_being_scored']['task_id']}")
                print(f"Evaluation result: {sample['response'][:200]}...")
            else:
                print(f"Prompt: {sample['prompt'][:100]}...")
                print(f"Response: {sample['response'][:200]}...")
    else:
        print("✗ No results to save")


if __name__ == "__main__":
    main() 