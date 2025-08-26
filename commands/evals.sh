# MBPP
# no thinking
MASTER_PORT=12345 CUDA_VISIBLE_DEVICES=0 python3 code_eval.py --dataset=mbpp --template_type=default --no_thinking=True
# force answer
MASTER_PORT=12346 CUDA_VISIBLE_DEVICES=1 python3 code_eval.py --dataset=mbpp --template_type=default --rollout_name=vllm_force_think_after_max
# best of n
MASTER_PORT=12347 CUDA_VISIBLE_DEVICES=2 python3 code_eval.py --dataset=mbpp --template_type=plan_first --rollout_name=vllm_best_of_n --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_600
# rewind and repeat
MASTER_PORT=12348 CUDA_VISIBLE_DEVICES=3 python3 code_eval.py --dataset=mbpp --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_600
# answer rewind and repeat
MASTER_PORT=12349 CUDA_VISIBLE_DEVICES=4 python3 code_eval.py --dataset=mbpp --template_type=default --rollout_name=vllm_answer_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_600
 
# MATH500
MASTER_PORT=12350 CUDA_VISIBLE_DEVICES=0 python3 math_eval.py --dataset=math500 --template_type=default --no_thinking=True
MASTER_PORT=12351 CUDA_VISIBLE_DEVICES=1 python3 math_eval.py --dataset=math500 --template_type=default --rollout_name=vllm_force_think_after_max
MASTER_PORT=12352 CUDA_VISIBLE_DEVICES=2 python3 math_eval.py --dataset=math500 --template_type=plan_first --rollout_name=vllm_best_of_n --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300
MASTER_PORT=12353 CUDA_VISIBLE_DEVICES=3 python3 math_eval.py --dataset=math500 --template_type=default --rollout_name=vllm_answer_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300
MASTER_PORT=12354 CUDA_VISIBLE_DEVICES=4 python3 math_eval.py --dataset=math500 --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300

MASTER_PORT=12355 CUDA_VISIBLE_DEVICES=5 python3 math_eval.py --dataset=math500 --template_type=plan_first --rollout_name=vllm_best_of_n --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300 --use_random_selection=True --max_problems=50
MASTER_PORT=12356 CUDA_VISIBLE_DEVICES=6 python3 math_eval.py --dataset=math500 --template_type=plan_first --rollout_name=vllm_best_of_n --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300 --use_random_selection=True --max_problems=50 --enable_iterative_reprompting=False --use_similarity_filtering=False
MASTER_PORT=12357 CUDA_VISIBLE_DEVICES=7 python3 math_eval.py --dataset=math500 --template_type=default --rollout_name=vllm_force_think_after_max --response_length=8192

 
MASTER_PORT=12350 CUDA_VISIBLE_DEVICES=0 python3 math_eval.py --dataset=math500 --template_type=default --no_thinking=True --model_path=Qwen/Qwen3-32B
MASTER_PORT=12351 CUDA_VISIBLE_DEVICES=1 python3 math_eval.py --dataset=math500 --template_type=default --rollout_name=vllm_force_think_after_max --model_path=Qwen/Qwen3-32B  
MASTER_PORT=12352 CUDA_VISIBLE_DEVICES=2 python3 math_eval.py --dataset=math500 --template_type=default --no_thinking=True --model_path=Qwen/Qwen3-4B
MASTER_PORT=12353 CUDA_VISIBLE_DEVICES=3 python3 math_eval.py --dataset=math500 --template_type=default --rollout_name=vllm_force_think_after_max --model_path=Qwen/Qwen3-4B   

# AIME2024
MASTER_PORT=12360 CUDA_VISIBLE_DEVICES=0 python3 math_eval.py --dataset=aime2024 --template_type=default --no_thinking 
MASTER_PORT=12361 CUDA_VISIBLE_DEVICES=1 python3 math_eval.py --dataset=aime2024 --template_type=default 
MASTER_PORT=12362 CUDA_VISIBLE_DEVICES=2 python3 math_eval.py --dataset=aime2024 --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300


# LongFormQA
MASTER_PORT=12370 CUDA_VISIBLE_DEVICES=0 python3 longform_qa_eval.py --dataset=longform_qa --template_type=default --no_thinking=True
MASTER_PORT=12371 CUDA_VISIBLE_DEVICES=1 python3 longform_qa_eval.py --dataset=longform_qa --template_type=default --rollout_name=vllm_force_answer
MASTER_PORT=12372 CUDA_VISIBLE_DEVICES=2 python3 longform_qa_eval.py --dataset=longform_qa --template_type=plan_first --rollout_name=vllm_best_of_n --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
MASTER_PORT=12373 CUDA_VISIBLE_DEVICES=3 python3 longform_qa_eval.py --dataset=longform_qa --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
MASTER_PORT=12374 CUDA_VISIBLE_DEVICES=4 python3 longform_qa_eval.py --dataset=longform_qa --template_type=plan_first --rollout_name=vllm_answer_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300

# BirdSQL
# no thinking
MASTER_PORT=12380 CUDA_VISIBLE_DEVICES=0 python3 birdsql_eval.py --dataset=birdsql --no_thinking=True --max_problems=100
# thinking
MASTER_PORT=12381 CUDA_VISIBLE_DEVICES=1 python3 birdsql_eval.py --dataset=birdsql --template_type=default --max_problems=100
# plan first
MASTER_PORT=12382 CUDA_VISIBLE_DEVICES=2 python3 birdsql_eval.py --dataset=birdsql --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_birdsql_plan_interleave/global_step_300


# bigcodebench
MASTER_PORT=12390 CUDA_VISIBLE_DEVICES=0 python3 bigcodebench_eval.py --dataset=bigcodebench --template_type=default --no_thinking=True
MASTER_PORT=12391 CUDA_VISIBLE_DEVICES=1 python3 bigcodebench_eval.py --dataset=bigcodebench --template_type=default --rollout_name=vllm_force_think_after_max
MASTER_PORT=12392 CUDA_VISIBLE_DEVICES=2 python3 bigcodebench_eval.py --dataset=bigcodebench --template_type=plan_first --rollout_name=vllm_best_of_n --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
MASTER_PORT=12393 CUDA_VISIBLE_DEVICES=3 python3 bigcodebench_eval.py --dataset=bigcodebench --template_type=default --rollout_name=vllm_answer_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
MASTER_PORT=12394 CUDA_VISIBLE_DEVICES=4 python3 bigcodebench_eval.py --dataset=bigcodebench --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model_path=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
