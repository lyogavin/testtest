#!/bin/bash
set -e



echo "run train_config_133_82...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_82



echo "run train_config_133_83...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_83



echo 'done'
