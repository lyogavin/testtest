#!/bin/bash
set -e



echo "run train_config_133_107...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_107



echo "run train_config_133_108...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_108

echo "run train_config_133_109...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_109

echo 'done'
