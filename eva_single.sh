export CUDA_VISIBLE_DEVICES=0
export DATA_DIR='/datadisk/data'
export TRAINER_DEFAULT_HDFS_DIR=null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_JSONL=true


# export BASE_MODEL='/datadisk/model/qwen2.5-3b-ins'
# export EXPERIMENT_NAME=scifact/qwen2.5-3b-ins

# export BASE_MODEL='/datadisk/checkpoints/qwen2.5-3b-ins-sft'
# export EXPERIMENT_NAME=qwen2.5-3b-ins-sft

declare -A num_pair
num_pair["fever"]=450
num_pair["hover"]=500
num_pair["feverous"]=441
num_pair["exfever"]=789
num_pair["scifact"]=711

#-----------------------------------------------------------------
# export BASE_MODEL='/datadisk/model/qwen2.5-7b-ins'
# export EXPERIMENT_NAME=qwen/fever/qwen2.5-7b-ins
# export DATA_NAME=qwen/fever

# export BASE_MODEL='/datadisk/model/qwen2.5-14b-ins'
# export EXPERIMENT_NAME=qwen/scifact/qwen2.5-14b-ins
# export DATA_NAME=qwen/scifact
## Change val_batch_size when changing dataset!!!

# export BASE_MODEL='/datadisk/model/llama3.1-8b-ins'
# export EXPERIMENT_NAME=llama/hover/llama3.1-8b-ins
# export DATA_NAME=llama/fever

# export BASE_MODEL='/datadisk/reasoning-cv/reasoner-guide-r2-final'
# export EXPERIMENT_NAME=qwen/fever/reasoning-guide-r2-final
# export DATA_NAME=fever

export BASE_MODEL='/datadisk/model/qwen2.5-3b-ins'
export EXPERIMENT_NAME=qwen/fever/qwen2.5-3b-ins
export DATA_NAME=fever
export RESULT_PATH=qwen/fever


# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

# claim-r2 Evaluate
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    eval.save_csv=true \
    data.val_files=$DATA_DIR/hold_out/fever.parquet \
    data.val_data_num=null \
    data.val_batch_size=${num_pair[fever]} \
    data.max_prompt_length=4864 \
    data.max_response_length=512 \
    data.max_start_length=512 \
    data.max_obs_length=768 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/datadisk/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log