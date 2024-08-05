#!/bin/bash
MODEL_NAME=distilbert/distilgpt2 #Qwen/Qwen1.5-0.5B-Chat
MODEL_SHORTNAME="${MODEL_NAME#*/}"

SAVE_MODELS_DIR=output poetry run torchrun tar.py \
    --base_model_name "${MODEL_NAME}" \
    --retain_model_name "${MODEL_NAME}" \
    --base "${MODEL_SHORTNAME}" \
    --max_data_size 100 \
    --adversary_dist_types foo:1.0 \
    --max_steps 10

