export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='/datadisk/data'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_JSONL=true
export VLLM_ATTENTION_BACKEND=XFORMERS

# Define number of samples for each data source
declare -A num_pair
num_pair["fever"]=450
num_pair["hover"]=500
num_pair["feverous"]=441
num_pair["exfever"]=789
num_pair["scifact"]=711


export RESULT_PATH=qwen/exfever

export MODEL_NAME=qwen
MODELS=(
    "/datadisk/model/qwen2.5-3b-ins"
    "/datadisk/checkpoints/qwen2.5-3b-ins-sft"
    "/datadisk/checkpoints/Veri-R1/qwen2.5-3b-ins-offline/actor/global_step_80_val_0.7910"
    "/datadisk/checkpoints/Veri-R1/qwen2.5-3b-ins-online/actor/global_step_120_val_0.7539"
)

export DATA_NAME=exfever
EXPS=(
    "qwen2.5-3b-ins"
    "qwen2.5-3b-ins-sft"
    "qwen2.5-3b-ins-offline-80step"
    "qwen2.5-3b-ins-online-120step"
)

# export RESULT_PATH=llama/exfever

# export MODEL_NAME=llama
# MODELS=(
#     "/datadisk/model/llama3.2-3b-ins"
#     "/datadisk/checkpoints/llama3.2-3b-ins-sft"
#     "/datadisk/checkpoints/Veri-R1/llama3.2-3b-ins-offline/actor/global_step_65_val_0.7949"
#     "/datadisk/checkpoints/Veri-R1/llama3.2-3b-ins-online/actor/global_step_90_val_0.6895"
# )

# export DATA_NAME=exfever
# # 2) Corresponding experiment name array (order and number of elements must match MODELS)
# EXPS=(
#     "llama3.2-3b-ins"
#     "llama3.2-3b-ins-sft"
#     "llama3.2-3b-ins-offline-65step"
#     "llama3.2-3b-ins-online-90step"
# )


# 4) Iterate through indices, one-to-one correspondence
for i in "${!MODELS[@]}"; do
  BASE_MODEL="${MODELS[i]}"
  export BASE_MODEL
  EXP_NAME="${MODEL_NAME}/${DATA_NAME}/${EXPS[i]}"
  export EXPERIMENT_NAME="$EXP_NAME"
  
  echo "=== Experiment: $EXPERIMENT_NAME ==="
  PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    eval.save_csv=true \
    data.val_files=$DATA_DIR/hold_out/$DATA_NAME.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=${num_pair[${DATA_NAME}]} \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.max_start_length=512 \
    data.max_obs_length=768 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=0 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=true \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=3 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/datadisk/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
done
