#!/bin/bash 
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../" >/dev/null 2>&1 && pwd)" 

python ${ROOT_DIR}/PeTriBOX/scripts/test.py  \
    --data-path ${ROOT_DIR}/datasets/alphafoldDB_ftp_CANC.pt \
    --root-path ${ROOT_DIR}/weights/ \
    --model-name PeTriMPOV \
    --test-name lmreport \
    --test-collator Collatorforlm \
    --test-model LMreport \
    --batch-size 32 \
    --max-batches 100 \
    --workers 4 
    