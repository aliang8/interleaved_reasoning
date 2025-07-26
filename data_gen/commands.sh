python3 data_gen/generate_multiple_solutions_interleave.py --prompt_file data_gen/code_list_prompts.txt --output_dir data --num_code_solutions 2 --code_model Qwen/Qwen3-32B --device_map auto
python3 data_gen/generate_concat_interleaved_code.py --use_canonical --num_samples 50
python3 data_gen/generate_concat_interleaved_math.py --use_canonical --num_samples 50
python data_gen/generate_bcb_outline_code_test_interleave.py --num_samples 50