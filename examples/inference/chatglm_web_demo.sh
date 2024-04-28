#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../../src/web_demo.py \
    --model_name_or_path  THUDM/chatglm3-6b \
    --template default