#!/bin/bash
MODEL_NAME=Qwen/Qwen1.5-0.5B-Chat   # distilbert/distilgpt2
MODEL_SHORTNAME=qwen-0.5b           # distilgpt2

SAVE_MODELS_DIR=output poetry run torchrun tar.py \
    --base_model_name "${MODEL_NAME}" \
    --retain_model_name "${MODEL_NAME}" \
    --base "${MODEL_SHORTNAME}" \
    --max_steps 10 \
    --max_data_size 100 \
    --tar_tamper_resistance_loss_type dpo \
    --adversary_dist_types adversary:1.0 \
    --subject beavertails,beavertails,beavertails
