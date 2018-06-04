#!/bin/bash
set -e



echo "run train_config_133_94...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_94



echo "run train_config_133_95...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_95


echo 'done'
