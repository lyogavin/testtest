#!/bin/bash
set -e



echo "run train_config_133_69...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_69



echo "run train_config_133_70...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_70



echo 'done'
