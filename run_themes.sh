#!/bin/bash
set -e



echo "run train_config_133_79...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_79



echo "run train_config_133_80...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_80



echo 'done'
