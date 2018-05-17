#!/bin/bash
set -e



echo "run train_config_133_47...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_47



echo "run train_config_133_48...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_48


echo "run train_config_133_49...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_49



echo "run train_config_133_50...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_50


echo 'done'
