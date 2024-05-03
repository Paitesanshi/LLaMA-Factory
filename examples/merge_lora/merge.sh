#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES= python ../../src/export_model.py \
    --model_name_or_path THUDM/chatglm3-6b \
    --adapter_name_or_path /home/v-leiwang8/LLaMA-Factory/saves/ChatGLM/lora/point_10k/checkpoint-5000 \
    --template default \
    --finetuning_type lora \
    --export_dir ../../models/glm3_reward_10k \
    --export_size 2 \
    --export_legacy_format False
