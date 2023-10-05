#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

filename=my_submission_$(date +'%d%m%Y').zip

pushd $SCRIPT_DIR
zip ${filename} \
    fit.py \
    predict.py \
    my_model.py \
    requirements.txt
popd
