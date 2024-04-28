#!/bin/bash

CUDA_VISIBLE_DEVICES=0 API_PORT=2025 python ../../src/api_demo.py \
    --model_name_or_path /home/v-leiwang8/llm_models/mistral-7B-v0.2 \
    --template mistral \
    --adapter_name_or_path ../../saves/Mistral-7B-v0.2/full/pt/ \
    --finetuning_type lora
