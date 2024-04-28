#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/cli_demo.py \
    --model_name_or_path  /home/v-leiwang8/LLaMA-Factory/models/mistral-pt\
    --template default \
    --adapter_name_or_path /home/v-leiwang8/LLaMA-Factory/saves/Mistral-7B-v0.2/lora/sft \
    --finetuning_type lora
