#!/bin/bash
set -e



echo "run train_config_133_73...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_73



echo "run train_config_133_74...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_74



echo 'done'
