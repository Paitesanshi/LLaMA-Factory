#!/bin/bash
# DO NOT use quantized model or quantization_bit when merging lora weights

CUDA_VISIBLE_DEVICES= python ../../src/export_model.py \
    --model_name_or_path /home/v-leiwang8/llm_models/mistral-7B-v0.2 \
    --adapter_name_or_path /home/v-leiwang8/LLaMA-Factory/saves/Mistral-7B-v0.2/full/pt \
    --template mistral \
    --finetuning_type lora \
    --export_dir ../../models/mistral-pt \
    --export_size 2 \
    --export_legacy_format False
