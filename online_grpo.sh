export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='/datadisk/data'
export TRAINER_DEFAULT_HDFS_DIR=null
export WAND_PROJECT='Veri-R1'
export WANDB_API_KEY=''
export DATA_DIR=/datadisk/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export BASE_MODEL='/datadisk/model/qwen2.5-3b-ins'
# export EXPERIMENT_NAME=qwen2.5-3b-ins-online-labelonly

# export BASE_MODEL='/datadisk/model/qwen2.5-3b-ins'
# export EXPERIMENT_NAME=qwen2.5-3b-ins-online-no_validity_weight

# export BASE_MODEL='/datadisk/model/llama3.2-3b-ins'
# export EXPERIMENT_NAME=llama3.2-3b-ins-online-no_validity_weight

# export BASE_MODEL='/datadisk/model/qwen2.5-3b-ins'
# export EXPERIMENT_NAME=qwen2.5-3b-ins-evi-true
# export EXPERIMENT_NAME=qwen2.5-3b-ins-label-only
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-7b-it-em


# export BASE_MODEL='/datadisk/checkpoints/Veri-R2/qwen2.5-3b-ins-offline/global_step_50_val_0.8203'
# export EXPERIMENT_NAME=scifact_off/qwen2.5-3b-ins-offline-online

# export BASE_MODEL='/datadisk/checkpoints/Veri-R2/qwen2.5-3b-ins-offline/actor/global_step_75_val_0.8379'
# export EXPERIMENT_NAME=qwen2.5-3b-ins-offline-online

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_online.parquet \
    data.val_files=$DATA_DIR/test_on.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=4864 \
    data.max_response_length=512 \
    data.max_start_length=512 \
    data.max_obs_length=768 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16\
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=10 \
    # trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/datadisk/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log