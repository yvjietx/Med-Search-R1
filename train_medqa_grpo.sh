#!/bin/bash

# # 设置HuggingFace镜像
# export HF_ENDPOINT='https://hf-mirror.com'
# export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='data/medqa_search'

# wandb配置
export WANDB_PROJECT='MedQA-GRPO'
export WANDB_NAME=$EXPERIMENT_NAME
export WANDB_WATCH=all  # 监控所有模型参数
export WANDB_LOG_MODEL=true  # 记录模型检查点
export WANDB_SILENT=false  # 显示详细日志
export WANDB_MODE=offline # 使用离线模式避免网络问题

# 设置模型和实验名称
export BASE_MODEL='/home/yvjie/models/Qwen/Qwen2.5-0.5B'
export EXPERIMENT_NAME=medqa_search-grpo-qwen2.5-0.5b

# 设置VLLM后端
export VLLM_ATTENTION_BACKEND=XFORMERS

# 设置torch显存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.max_start_length=2048 \
    data.max_obs_length=2048 \
    data.shuffle_train_dataloader=True \
    +data.truncation=longest_first \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    trainer.log_every_n_steps=10 \
    trainer.log_model_gradients=true \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=100 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=3 \
    retriever.url='http://127.0.0.1:8000/retrieve' \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log 