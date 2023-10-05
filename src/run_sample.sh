#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

for fold in fold1 fold2; do
    echo "-----------------------------------"
    echo "Run on fold ${fold}"
    echo "-----------------------------------"

    echo "-----------------------------------"
    echo "FIT"
    echo "-----------------------------------"
    python $SCRIPT_DIR/fit.py \
        -d $SCRIPT_DIR/sample_data/${fold}/fit \
        -m $SCRIPT_DIR/model/${fold} \
        -o $SCRIPT_DIR/output/${fold}

    echo "-----------------------------------"
    echo "PREDICT"
    echo "-----------------------------------"
    python $SCRIPT_DIR/predict.py \
        -d $SCRIPT_DIR/sample_data/${fold}/predict \
        -m $SCRIPT_DIR/model/${fold} \
        -o $SCRIPT_DIR/output/${fold}
done
