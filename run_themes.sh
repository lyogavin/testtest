#!/bin/bash
set -e



echo "run train_config_133_45...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_45



echo "run train_config_133_46...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_46


echo 'done'
