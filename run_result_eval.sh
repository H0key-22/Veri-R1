#!/bin/bash

# Configuration parameters
DATA_PATH="/root/Veri-R1/case/qwen/fever/qwen2.5-3b-ins.jsonl"  # result file path
OUTPUT_DIR="./evaluation_results"                  # Output directory
EXPERIMENT_NAME="fever_evaluation"                 # Experiment name

# Create output directory
mkdir -p $OUTPUT_DIR

# Run evaluation
echo "Starting main_eval.py evaluation..."
echo "Data path: $DATA_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"

cd /root/Veri-R1

python -m verl.trainer.main_eval \
    data.path=$DATA_PATH \
    data.response_key=response \
    data.data_source_key=data_source \
    data.ground_truth_key=ground_truth \
    summary_file=$OUTPUT_DIR/${EXPERIMENT_NAME}_summary.csv \
    2>&1 | tee $OUTPUT_DIR/${EXPERIMENT_NAME}.log
