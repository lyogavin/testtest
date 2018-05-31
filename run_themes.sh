#!/bin/bash
set -e



echo "run train_config_133_84...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_84



echo "run train_config_133_85...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_85


echo "run train_config_133_86...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_86


echo 'done'
