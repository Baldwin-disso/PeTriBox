#!/bin/bash 
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../" >/dev/null 2>&1 && pwd)" 

python ${ROOT_DIR}/PeTriBOX/scripts/predict_ppi.py ${ROOT_DIR}/data/pkls ${ROOT_DIR}/ppi_out/ppi_out.csv \
    --model-path ${ROOT_DIR}/weights/PeTriPPI \
    