#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/cli_demo.py \
    --model_name_or_path  meta-llama/Meta-Llama-3-8B-Instruct \
    --template default
