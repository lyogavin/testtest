#!/bin/bash
set -e



echo "run train_config_133_60...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_60



echo "run train_config_133_61...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_61



echo 'done'
