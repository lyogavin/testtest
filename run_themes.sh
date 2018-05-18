#!/bin/bash
set -e



echo "run train_config_133_56...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_56



echo "run train_config_133_57...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_57



echo 'done'
