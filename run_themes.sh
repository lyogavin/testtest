#!/bin/bash
set -e



echo "run train_config_133_61...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_61



echo "run train_config_133_64...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_64



echo 'done'
