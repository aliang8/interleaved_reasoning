docker build -t verl-custom .


docker run -it \
    -v $(pwd):/workspace \
    --name verl-custom \
    --gpus all \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e HF_TOKEN=$HF_TOKEN \
    verl-custom
    
./setup_docker.sh

# run verl

# base test command
CUDA_VISIBLE_DEVICES=1,2,3,4 PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=verl/data/gsm8k/train.parquet \
 data.val_files=verl/data/gsm8k/test.parquet \
 data.train_batch_size=32 \
 data.max_prompt_length=256 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=32 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 critic.ppo_micro_batch_size_per_gpu=1 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=verl \
 trainer.experiment_name=gsm8k \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log


 model.lora_rank=16 \

# run lm-eval 

lm_eval --model vllm \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --wandb_args project=lm-eval-harness-integration,entity=clvr \
    --output_path results \
    --limit 10


lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
    --tasks frames_benchmark \
    --device cuda:0 \
    --batch_size 8 \
    --log_samples \
    --wandb_args project=lm-eval-harness-integration,entity=clvr \
    --output_path results_frames_benchmark

gcloud compute firewall-rules create allow-http-from-vm-8000 \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:8000 \
    --source-ranges=10.150.0.2/32 \
    --target-tags=http-server-tag \
    --project=anthony-intern-2025 \
    --description="Allow HTTP traffic from verl-h100-2 (10.150.0.2) to instances with http-server-tag"

instance-20250616-144352                           10.128.0.2
workstations-153b2cc9-4513-415e-9284-189e3e6f9857  10.128.15.204
verl-h100                                          10.128.0.30
verl-2                                             10.128.0.22
verl                                               10.128.0.12
verl-h100-2                                        10.150.0.2
verl-h100-3                                        10.150.0.3


gcloud compute instances add-tags verl-h100 \
    --zone=us-central1-b \
    --tags=http-server-tag \
    --project=anthony-intern-2025


gcloud compute firewall-rules describe allow-http-from-vm-8000 --project=anthony-intern-2025