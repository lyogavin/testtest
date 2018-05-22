#!/bin/bash
set -e



echo "run train_config_133_67...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_67



echo "run train_config_133_68...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_68



echo 'done'
