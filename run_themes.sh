#!/bin/bash
set -e



echo "run train_config_133_89...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_89



echo "run train_config_133_90...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_90


echo 'done'
