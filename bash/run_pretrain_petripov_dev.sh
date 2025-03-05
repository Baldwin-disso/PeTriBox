#!/bin/bash 
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../" >/dev/null 2>&1 && pwd)" 

python ${ROOT_DIR}/PeTriBOX/scripts/train.py \
    --root-path ${ROOT_DIR}/my_weights/ \
    --model-name MyPeTriPOV_dev \
    --model-cls-name PeTriPOV \
    --task MLM \
    --data-path ${ROOT_DIR}/datasets/alphafoldDB_ftp_CANC_debug.pt \
    --log \
    --workers 0 \
    --device-id 0 \
    --seed  42 \
    --log true \
    --log-steps 100 \
    --epochs 3 \
    --batch-size 2 \
    --max-batches 2 \
    --split-rates 0.8 0.1 0.1 \
    --iter-max 350000 \
    --random-trunk false \
    --global-rotation-augmentation true \
    --global-translation-augmentation 20 \
    --translation-noise-augmentation 0.0 \
    --rotation-noise-augmentation 0.0 \
    --focused-rate 0.15 \
    --optimizer "adamw" \
    --criterion "CrossEntropyLoss" \
    --clip 1.0 \
    --init-lr 0.001 \
    --end-lr 1e-07 \
    --betas 0.9 0.999 \
    --weight-decay 0.01 \
    --epsilon 1e-08 \
    --warmup 30000 \
    --ending 250000 \
    --seq-len 1024 \
    --n-layers 5 \
    --n-heads 12 \
    --query-dimensions 16 \
    --value-dimensions 16 \
    --point-dimensions 1 \
    --feed-forward-dimensions 256 \
    --learnable-embedding-type learnable_weights_and_MLP \
    --vocab-size 25 \
    
    
    