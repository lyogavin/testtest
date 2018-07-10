#!/bin/bash
set -e



echo "run train_config_133_104...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_104



echo "run train_config_133_105...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_105

echo "run train_config_133_106...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_106

echo 'done'
