#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置模型和数据路径
MODEL_NAME='Qwen/Qwen2.5-3B'  # 使用Qwen2.5 3B模型
DATA_PATH="./data/medqa/test.parquet"
MODEL_SHORT_NAME=$(echo $MODEL_NAME | tr '/' '_')  # 将模型路径中的'/'替换为'_'
OUTPUT_DIR="./logs/medqa_inference_${MODEL_SHORT_NAME}"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行推理
python scripts/inference_medqa.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --max_new_tokens 512 