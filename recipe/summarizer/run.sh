MODEL_PATH=Qwen/Qwen3-0.6B
DATA_PATH=/workspace/datasets/summarizer
DATA_PATH=./datasets/summarizer

# Eval Data Process
python3 -m recipe.summarizer.data_process \
    --local_dir $DATA_PATH \
    --tasks all

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=12 \
    data.max_prompt_length=2560 \
    data.max_response_length=2560 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='compressor' \
    trainer.experiment_name='rl_reformulator' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=150 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    trainer.ray_wait_register_center_timeout=3000 \
    custom_reward_function.path=recipe/summarizer/reward_score.py \
    custom_reward_function.name=reward_func


