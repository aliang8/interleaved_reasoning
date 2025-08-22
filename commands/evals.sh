# answer rewind and repeat
MASTER_PORT=12355 CUDA_VISIBLE_DEVICES=0 python3 mbpp_evaluation.py --template_type=default --rollout_name=vllm_answer_repeat --model=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
# best of n
MASTER_PORT=12356 CUDA_VISIBLE_DEVICES=1 python3 mbpp_evaluation.py --template_type=plan_first --rollout_name=vllm_best_of_n --model=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
# rewind and repeat
MASTER_PORT=12356 CUDA_VISIBLE_DEVICES=2 python3 mbpp_evaluation.py --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_bcb_plan_code_interleave/global_step_300
# think answer
MASTER_PORT=12357 CUDA_VISIBLE_DEVICES=3 python3 mbpp_evaluation.py --template_type=default 
# no thinking
MASTER_PORT=12358 CUDA_VISIBLE_DEVICES=4 python3 mbpp_evaluation.py --template_type=default --no_thinking


# MATH500
MASTER_PORT=12359 CUDA_VISIBLE_DEVICES=5 python3 math_evaluation.py --dataset=math500 --template_type=default 
MASTER_PORT=12360 CUDA_VISIBLE_DEVICES=6 python3 math_evaluation.py --dataset=math500 --template_type=default --no_thinking
MASTER_PORT=12361 CUDA_VISIBLE_DEVICES=7 python3 math_evaluation.py --dataset=math500 --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300



# AIME2024
MASTER_PORT=12370 CUDA_VISIBLE_DEVICES=0 python3 math_evaluation.py --dataset=aime2024 --template_type=default --no_thinking 
MASTER_PORT=12371 CUDA_VISIBLE_DEVICES=1 python3 math_evaluation.py --dataset=aime2024 --template_type=default 
MASTER_PORT=12372 CUDA_VISIBLE_DEVICES=2 python3 math_evaluation.py --dataset=aime2024 --template_type=plan_first --rollout_name=vllm_rewind_and_repeat --model=checkpoints/interleaved-sft/interleaved-sft-Qwen3-8B_math500_plan_interleave/global_step_300
