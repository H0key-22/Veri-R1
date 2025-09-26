#!/bin/bash

# Set your API key here
export API_KEY='your_api_key_here'
export API_BASE="your_api_base_here"
export MODEL_NAME="gpt-4o" 

# Default configuration
export DATA_DIR='/datadisk/data'
export DATASET_NAME='fever'
export DATASET_PATH="$DATA_DIR/hold_out/fever.parquet"
export EXPERIMENT_NAME="${MODEL_NAME}_${DATASET_NAME}_eval"

declare -A num_pair
num_pair["fever"]=450
num_pair["hover"]=500
num_pair["feverous"]=441
num_pair["exfever"]=789
num_pair["scifact"]=711
num_pair["fever_test"]=64

# Performance settings
export MAX_WORKERS=128
export BATCH_SIZE=${num_pair[$DATASET_NAME]}    

# Connection error handling settings
# All connection errors will be handled automatically with immediate failure and counting

# Check if API key is set
if [ "$API_KEY" = "your_api_key_here" ]; then
    echo "❌ Error: Please set your OpenAI API key in the script"
    echo "   Edit the API_KEY variable at the top of this file"
    exit 1
fi

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found: $DATASET_PATH"
    echo "   Please check your DATA_DIR setting"
    exit 1
fi

echo " Starting $MODEL_NAME evaluation..."
echo "   Dataset: $DATASET_NAME"
echo "   Path: $DATASET_PATH"
echo "   Model: $MODEL_NAME"
echo "   Workers: $MAX_WORKERS"
echo "   Batch size: $BATCH_SIZE"
echo "   Connection errors: Auto-handled with immediate failure and counting"
echo ""

# Run evaluation
PYTHONUNBUFFERED=1 python3 api_validator.py \
    --data_path "$DATASET_PATH" \
    --api_key "$API_KEY" \
    --experiment_name "$EXPERIMENT_NAME" \
    --max_workers "$MAX_WORKERS" \
    --batch_size "$BATCH_SIZE" \
    --api_base "$API_BASE" \
    --model "$MODEL_NAME" \
    2>&1 | tee "${EXPERIMENT_NAME}.log"

echo ""
echo "   Results: ./validation_results/$EXPERIMENT_NAME/"
echo "   Log: ${EXPERIMENT_NAME}.log" 