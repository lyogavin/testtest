#!/bin/bash
set -e



echo "run train_config_133_92...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_92



echo "run train_config_133_93...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_93


echo 'done'
