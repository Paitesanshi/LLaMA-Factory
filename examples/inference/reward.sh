#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/cli_demo.py \
    --model_name_or_path  THUDM/chatglm3-6b\
    --template default \
    --adapter_name_or_path /home/v-leiwang8/LLaMA-Factory/saves/ChatGLM/lora/point/checkpoint-1000 \
    --finetuning_type lora
