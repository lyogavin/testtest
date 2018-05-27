#!/bin/bash
set -e



echo "run train_config_133_71...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_71



echo "run train_config_133_72...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_72



echo 'done'
