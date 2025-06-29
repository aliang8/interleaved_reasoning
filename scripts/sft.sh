torchrun --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=verl/data/gsm8k_sft/train.parquet \
    data.val_files=verl/data/gsm8k_sft/test.parquet \
    data.train_batch_size=256 \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \
    trainer.project_name=gsm8k-sft \
    trainer.default_hdfs_dir=hdfs://user/verl/experiments/gsm8k/Qwen2.5-7B-Instruct/ \
    trainer.experiment_name=gsm8k-sft-Qwen2.5-7B-Instruct \
    trainer.total_epochs=4 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb']


torchrun --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=reasoning_data/interleaved_listing_dataset_train.parquet \
    data.val_files=reasoning_data/interleaved_listing_dataset_test.parquet \
    data.train_batch_size=8 \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    trainer.project_name=interleaved-sft \
    trainer.experiment_name=interleaved-sft-Qwen3-8B \
    trainer.total_epochs=100 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    use_remove_padding=True \
    model.lora_rank=32 \
    ulysses_sequence_parallel_size=4 



torchrun --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=reasoning_data/interleaved_listing_dataset_train.parquet \
    data.val_files=reasoning_data/interleaved_listing_dataset_test.parquet \
    data.train_batch_size=8 \
    data.prompt_key=question \
    data.response_key=answer \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    trainer.project_name=interleaved-sft \
    trainer.experiment_name=interleaved-sft-Qwen3-8B \
    trainer.total_epochs=100 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    use_remove_padding=True \
    ulysses_sequence_parallel_size=4 



python3 -m verl.trainer.generate_validation_rollouts \
  --checkpoint_dir=checkpoints/interleaved-sft/test/global_step_1000 \
  --base_model_path=Qwen/Qwen3-8B \
  --val_data=[reasoning_data/interleaved_listing_dataset_train.parquet,reasoning_data/interleaved_listing_dataset_test.parquet] \