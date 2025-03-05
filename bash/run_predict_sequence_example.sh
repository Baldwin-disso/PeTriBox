#!/bin/bash 
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../" >/dev/null 2>&1 && pwd)" 

python ${ROOT_DIR}/PeTriBOX/scripts/predict_sequence.py ${ROOT_DIR}/data/monomers ${ROOT_DIR}/seqs_out \
    --model-path ${ROOT_DIR}/weights/PeTriMPOV \
    --draws-per-pdb 2 \
    --temperature 1.0 \
    --out-file-suffix _design \
    --checkpoint_name inverse_folding.checkpoint
 