#!/usr/bin/env python3
"""
Standardized BirdSQL Evaluation Script

This script uses the BaseEvaluator framework to provide a clean,
maintainable implementation of text-to-SQL evaluation.
"""

import json
import re
import sqlite3
import tempfile
import os
import multiprocessing as mp
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

import pyrallis
from datasets import load_dataset
from evaluation_base import BaseEvaluator, EvaluationConfig
from helpers import create_prompts_dataproto, extract_solution_from_response

# Try to import func_timeout, provide fallback if not available
try:
    from func_timeout import func_timeout, FunctionTimedOut
    FUNC_TIMEOUT_AVAILABLE = True
except ImportError:
    FUNC_TIMEOUT_AVAILABLE = False
    print("⚠️  func_timeout not available. Install with: pip install func_timeout")
    print("   SQL execution evaluation will be disabled.")


class BirdSQLEvaluator(BaseEvaluator):
    """BirdSQL text-to-SQL evaluation implementation using the base framework."""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.total_correct_sql = 0
        self.difficulty_stats = {"simple": 0, "moderate": 0, "challenging": 0}
        self.difficulty_correct = {"simple": 0, "moderate": 0, "challenging": 0}
        
        # SQL execution settings
        self.db_root_path = getattr(config, 'db_root_path', './data/minidev/MINIDEV/dev_databases/')
        self.sql_timeout = getattr(config, 'sql_timeout', 30.0)
        self.num_cpus = getattr(config, 'num_cpus', 1)
        self.enable_sql_execution = getattr(config, 'enable_sql_execution', True)
        
        # Cache for database schemas
        self.schema_cache = {}
    
    def _load_dataset(self) -> List[Dict]:
        """Load BirdSQL dataset from HuggingFace."""
        print("Loading BirdSQL dataset...")
        
        try:
            # Load the dataset from HuggingFace
            dataset = load_dataset("birdsql/bird_mini_dev", split="mini_dev_sqlite")
            train_data = dataset
            print(f"✓ Loaded {len(train_data)} examples from BirdSQL dataset")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return []
        
        # Convert to the format expected by the evaluator
        examples = []
        
        for idx, example in enumerate(train_data):
            # Create example entry
            example_entry = {
                "id": idx,
                "question": example["question"],
                "sql": example["SQL"],
                "db_id": example["db_id"],
                "difficulty": example.get("difficulty", "unknown"),
                "evidence": example.get("evidence", ""),
                "question_id": f"birdsql_{idx}"
            }
            
            examples.append(example_entry)
            
            # Debug output for first few examples
            if idx < 3:
                print(f"  Example {idx + 1}: {example['question'][:50]}...")
                print(f"    Database: {example['db_id']}")
                print(f"    Difficulty: {example.get('difficulty', 'unknown')}")
                print(f"    SQL: {example['SQL'][:100]}...")
        
        print(f"✓ Converted {len(examples)} examples from BirdSQL dataset")
        return examples
    
    def create_prompts(self, examples: List[Dict], batch_size: int):
        """Create BirdSQL prompts with database context and questions."""
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            # Create prompts
            prompts = []
            for example in batch_examples:
                # Load database schema for this example
                db_schema = self._load_database_schema(example['db_id'])
                
                # Format the prompt for text-to-SQL generation with schema and external knowledge
                prompt = f"""{db_schema}

Question: {example['question']}"""
                
                # Add external knowledge evidence if available
#                 if example.get('evidence') and example['evidence'].strip():
#                     prompt += f"""

# External Knowledge Evidence:
# {example['evidence']}"""
                
                prompt += """

Please generate a SQL query to answer the question above. Output only the SQL query without any explanation."""
                
                prompts.append(prompt)
            
            # Create DataProto for this batch
            prompts_dataproto = create_prompts_dataproto(
                tokenizer=self.tokenizer,
                questions=prompts,
                max_prompt_length=self.rollout_config["rollout"]["prompt_length"],
                template_type=self.rollout_config["rollout"]["template_type"],
                enable_thinking=self.rollout_config["rollout"]["enable_thinking"]
            )
            
            yield batch_examples, prompts_dataproto
    
    def _extract_content_from_response(self, response_text: str) -> str:
        """Extract SQL query from response text."""
        return extract_solution_from_response(response_text, self.rollout_config["rollout"]["template_type"], self.rollout_config["rollout"]["enable_thinking"])
    
    def _evaluate_batch(self, examples: List[Dict], extracted_contents: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of SQL queries against ground truth."""
        batch_evaluations = []
        
        # Prepare SQL execution if enabled
        if self.enable_sql_execution and FUNC_TIMEOUT_AVAILABLE:
            sql_pairs = []
            db_paths = []
            
            for example, extracted_content in zip(examples, extracted_contents):
                if extracted_content.strip():
                    sql_pairs.append((extracted_content, example['sql']))
                    db_path = self._get_database_path(example['db_id'])
                    db_paths.append(db_path)
            
            # Execute SQL queries if we have valid pairs
            if sql_pairs:
                print(f"  Executing {len(sql_pairs)} SQL queries with timeout {self.sql_timeout}s...")
                execution_results = self._execute_sqls_parallel(sql_pairs, db_paths)
                print(f"  SQL execution completed: {len(execution_results)} results")
            else:
                execution_results = []
        else:
            execution_results = []

        # Evaluate each example
        for i, (example, extracted_content) in enumerate(zip(examples, extracted_contents)):
            results = {
                'question': example['question'],
                'db_id': example['db_id'],
                'difficulty': example['difficulty'],
                'ground_truth_sql': example['sql'],
                'generated_sql': extracted_content,
                'is_correct': False,
                'execution_error': None,
                'syntax_valid': False,
                'execution_result': None,
                'explanation': ''
            }
            
            if not extracted_content.strip():
                results['evaluation_error'] = 'No SQL generated'
                batch_evaluations.append(results)
                continue
            
            # Check SQL syntax validity
            syntax_valid = self._validate_sql_syntax(extracted_content)
            results['syntax_valid'] = syntax_valid
            
            if not syntax_valid:
                results['evaluation_error'] = 'Invalid SQL syntax'
                batch_evaluations.append(results)
                continue
            
            # Check execution results if available
            if execution_results and i < len(execution_results):
                exec_result = execution_results[i]
                results['execution_result'] = exec_result
                
                if exec_result['error']:
                    results['execution_error'] = exec_result['error']
                    results['is_correct'] = False
                    results['explanation'] = f"SQL execution failed: {exec_result['error']}"
                else:
                    results['is_correct'] = exec_result['res'] == 1
                    if results['is_correct']:
                        results['explanation'] = "Generated SQL produces same results as ground truth."
                    else:
                        results['explanation'] = "Generated SQL produces different results than ground truth."
            else:
                # Fallback to string comparison if execution not available
                gt_sql_normalized = self._normalize_sql(example['sql'])
                gen_sql_normalized = self._normalize_sql(extracted_content)
                
                is_correct = gt_sql_normalized == gen_sql_normalized
                results['is_correct'] = is_correct
                
                if is_correct:
                    results['explanation'] = "Generated SQL matches ground truth exactly (string comparison)."
                else:
                    results['explanation'] = f"Generated SQL differs from ground truth (string comparison). Expected: {example['sql']}, Got: {extracted_content}"
            
            batch_evaluations.append(results)
        
        return batch_evaluations
    
    def _validate_sql_syntax(self, sql_query: str) -> bool:
        """Basic SQL syntax validation."""
        try:
            # Remove common SQL comments and clean up
            cleaned_sql = re.sub(r'--.*$', '', sql_query, flags=re.MULTILINE)
            cleaned_sql = re.sub(r'/\*.*?\*/', '', cleaned_sql, flags=re.DOTALL)
            cleaned_sql = cleaned_sql.strip()
            
            # Basic checks
            if not cleaned_sql:
                return False
            
            # Check for basic SQL keywords
            sql_lower = cleaned_sql.lower()
            if not any(keyword in sql_lower for keyword in ['select', 'from']):
                return False
            
            # Check for balanced parentheses
            if cleaned_sql.count('(') != cleaned_sql.count(')'):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _normalize_sql(self, sql_query: str) -> str:
        """Normalize SQL for comparison."""
        # Convert to lowercase
        normalized = sql_query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove common variations
        normalized = normalized.replace('select distinct', 'select')
        normalized = normalized.replace('select *', 'select *')
        
        # Remove trailing semicolons
        normalized = normalized.rstrip(';')
        
        return normalized.strip()
    
    def _execute_sql(self, predicted_sql: str, ground_truth_sql: str, db_path: str) -> int:
        """Execute SQL queries and compare results."""
        if not FUNC_TIMEOUT_AVAILABLE:
            return 0  # Can't execute without func_timeout
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Execute predicted SQL
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
            
            # Execute ground truth SQL
            cursor.execute(ground_truth_sql)
            ground_truth_res = cursor.fetchall()
            
            conn.close()
            
            # Compare results
            if set(predicted_res) == set(ground_truth_res):
                return 1
            else:
                return 0
                
        except Exception as e:
            return 0
    
    def _execute_sql_with_timeout(self, predicted_sql: str, ground_truth_sql: str, db_path: str, idx: int) -> Dict:
        """Execute SQL with timeout handling."""
        if not FUNC_TIMEOUT_AVAILABLE:
            return {'sql_idx': idx, 'res': 0, 'error': 'func_timeout not available'}
        
        try:
            res = func_timeout(
                self.sql_timeout, 
                self._execute_sql,
                args=(predicted_sql, ground_truth_sql, db_path)
            )
            return {'sql_idx': idx, 'res': res, 'error': None}
            
        except FunctionTimedOut:
            return {'sql_idx': idx, 'res': 0, 'error': 'timeout'}
        except Exception as e:
            return {'sql_idx': idx, 'res': 0, 'error': str(e)}
    
    def _execute_sqls_parallel(self, sql_pairs: List[tuple], db_paths: List[str]) -> List[Dict]:
        """Execute SQL queries in parallel."""
        if not self.enable_sql_execution or not FUNC_TIMEOUT_AVAILABLE:
            return []
        
        results = []
        
        if self.num_cpus == 1:
            # Sequential execution
            for i, (pred_sql, gt_sql) in enumerate(sql_pairs):
                result = self._execute_sql_with_timeout(pred_sql, gt_sql, db_paths[i], i)
                results.append(result)
        else:
            # Parallel execution
            with mp.Pool(processes=self.num_cpus) as pool:
                async_results = []
                for i, (pred_sql, gt_sql) in enumerate(sql_pairs):
                    async_result = pool.apply_async(
                        self._execute_sql_with_timeout,
                        args=(pred_sql, gt_sql, db_paths[i], i)
                    )
                    async_results.append(async_result)
                
                # Collect results
                for async_result in async_results:
                    results.append(async_result.get())
        
        # Sort results by index
        results.sort(key=lambda x: x['sql_idx'])
        return results
    
    def _get_database_path(self, db_id: str) -> str:
        """Get the database file path for a given database ID."""
        db_path = os.path.join(self.db_root_path, db_id, f"{db_id}.sqlite")
        return db_path
    
    def _load_database_schema(self, db_id: str) -> str:
        """Load and parse database schema from CSV description files."""
        # Check cache first
        if db_id in self.schema_cache:
            return self.schema_cache[db_id]
        
        try:
            # Path to database description directory
            schema_dir = os.path.join(self.db_root_path, db_id, "database_description")
            
            if not os.path.exists(schema_dir):
                print(f"⚠️  Schema directory not found: {schema_dir}")
                return f"Database: {db_id}\n(Schema information not available)"
            
            # Find all CSV files in the schema directory
            csv_files = [f for f in os.listdir(schema_dir) if f.endswith('.csv')]
            
            if not csv_files:
                print(f"⚠️  No CSV schema files found in: {schema_dir}")
                return f"Database: {db_id}\n(Schema information not available)"
            
            # Load and parse all CSV files
            schema_parts = [f"Database: {db_id}"]
            
            for csv_file in sorted(csv_files):
                csv_path = os.path.join(schema_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    
                    # Check if the expected columns exist
                    expected_columns = ['original_column_name', 'column_name', 'column_description', 'data_format', 'value_description']
                    if not all(col in df.columns for col in expected_columns):
                        print(f"⚠️  Unexpected columns in {csv_file}: {list(df.columns)}")
                        continue
                    
                    # Extract table name from filename (remove .csv extension)
                    table_name = csv_file.replace('.csv', '')
                    schema_parts.append(f"\nTable: {table_name}")
                    
                    # Add column information
                    for _, row in df.iterrows():
                        col_name = row['original_column_name']
                        col_desc = row['column_description'] if pd.notna(row['column_description']) else 'No description'
                        data_format = row['data_format'] if pd.notna(row['data_format']) else 'Unknown'
                        value_desc = row['value_description'] if pd.notna(row['value_description']) else 'No value description'
                        
                        schema_parts.append(f"  - {col_name}: {col_desc} ({data_format})")
                        if value_desc and value_desc != 'No value description':
                            schema_parts.append(f"    Values: {value_desc}")
                    
                except Exception as e:
                    print(f"⚠️  Error reading schema file {csv_file}: {e}")
                    continue
            
            # Join all schema parts
            schema_text = "\n".join(schema_parts)
            
            # Cache the result
            self.schema_cache[db_id] = schema_text
            
            return schema_text
            
        except Exception as e:
            print(f"⚠️  Error loading schema for database {db_id}: {e}")
            return f"Database: {db_id}\n(Schema information not available)"
    
    def _create_base_result(self, example: Dict, response_text: str, extracted_content: str,
                           evaluation: Dict, interleaved_components: List[Dict],
                           task_completed: bool, num_tokens: int, ttft_ratio: float, 
                           total_tokens_generated: int = None, tokens_to_first_answer: int = None) -> Dict:
        """Create the base result structure for BirdSQL problems."""
        base_result = super()._create_base_result(
            example, response_text, extracted_content, evaluation,
            interleaved_components, task_completed, num_tokens, ttft_ratio, 
            total_tokens_generated, tokens_to_first_answer
        )
        
        # Debug: Print interleaved components info
        print(f"BirdSQL: Interleaved components for example {example.get('id', 'unknown')}: {len(interleaved_components)} components")
        if interleaved_components:
            for i, comp in enumerate(interleaved_components):
                print(f"  Component {i}: type={comp.get('type', 'unknown')}, index={comp.get('index', 'unknown')}")
        
        # Only enrich the evaluation structure when batch evaluation has populated it
        if isinstance(evaluation, dict) and 'is_correct' in evaluation:
            base_result['evaluation'] = {
                'tests_passed': 1 if evaluation['is_correct'] else 0,
                'tests_failed': 0 if evaluation['is_correct'] else 1,
                'total_tests': 1,
                'test_results': [{
                    'test': f"SQL correctness: {evaluation['is_correct']}",
                    'passed': evaluation['is_correct'],
                    'error': None if evaluation['is_correct'] else evaluation.get('explanation')
                }],
                'execution_error': evaluation.get('evaluation_error'),
                'test_imports': []
            }
        
        return base_result
    
    def _add_task_specific_fields(self, example: Dict, evaluation: Dict) -> Dict[str, Any]:
        """Add BirdSQL-specific fields to the result."""
        # Track correct SQL queries
        if evaluation['is_correct']:
            self.total_correct_sql += 1
        
        # Track difficulty statistics
        difficulty = example.get('difficulty', 'unknown')
        if difficulty in self.difficulty_stats:
            self.difficulty_stats[difficulty] += 1
            if evaluation['is_correct']:
                self.difficulty_correct[difficulty] += 1
        
        return {
            'test_list': [f"SQL should match: {evaluation.get('ground_truth_sql', 'Unknown')}"],
            'entry_point': 'sql_generation',
            'db_id': example['db_id'],
            'difficulty': difficulty,
            'question': example['question'],  # Add the question field
            'ground_truth_sql': example['sql'],
            'generated_sql': evaluation.get('generated_sql', ''),
            'sql_correct': evaluation['is_correct'],
            'syntax_valid': evaluation.get('syntax_valid', False),
            'execution_result': evaluation.get('execution_result'),
            'execution_error': evaluation.get('execution_error'),
            'evidence': example.get('evidence', ''),
            'explanation': evaluation.get('explanation', ''),
            'database_schema': self._load_database_schema(example['db_id']),
            'template_type': self.rollout_config["rollout"]["template_type"]  # Add template type
            # Note: interleaved_components will be populated by the base evaluator
        }
    
    def _get_dataset_name(self) -> str:
        """Get the name of the dataset for filename generation."""
        return "birdsql"
    
    def _generate_html_visualization(self, output_dir: Path):
        """Generate HTML visualization for BirdSQL problems."""
        # Get the template directory from base class
        template_dir = super()._generate_html_visualization(output_dir)
        
        # Generate filename: rollout_name_response_length.html
        filename = self.rollout_config['rollout']['name']
        
        if self.cfg.template_type == "plan_first":
            filename += f"_{self.rollout_config['rollout']['n_candidates']}"
        
        filename += f"_{self.cfg.response_length}"
        
        html_file = template_dir / f"{filename}.html"
        
        # Import and use the HTML visualization function
        try:
            from create_birdsql_html import create_birdsql_html_visualization
            create_birdsql_html_visualization(self.all_results, html_file)
            print(f"🎨 HTML visualization saved to {html_file}")
        except ImportError:
            print(f"🎨 HTML visualization would be saved to {html_file}")
            print("   (HTML visualization module not found)")
    
    def _print_task_specific_metrics(self):
        """Print BirdSQL-specific metrics."""
        sql_accuracy = (self.total_correct_sql / self.total_problems * 100) if self.total_problems > 0 else 0
        print(f"SQL Generation Accuracy: {sql_accuracy:.1f}% ({self.total_correct_sql}/{self.total_problems})")
        
        # Print difficulty-based statistics
        print("\n📊 Difficulty-based Statistics:")
        for difficulty in ["simple", "moderate", "challenging"]:
            total = self.difficulty_stats[difficulty]
            correct = self.difficulty_correct[difficulty]
            if total > 0:
                accuracy = (correct / total * 100)
                print(f"  {difficulty.capitalize()}: {accuracy:.1f}% ({correct}/{total})")


def main():
    """Main BirdSQL evaluation function."""
    print("=== BirdSQL Text-to-SQL Evaluation ===\n")
    
    # Parse configuration using pyrallis
    cfg = pyrallis.parse(config_class=EvaluationConfig)
    
    # Create and run evaluator
    evaluator = BirdSQLEvaluator(cfg)
    success = evaluator.run_evaluation()
    
    return success


if __name__ == "__main__":
    main() 