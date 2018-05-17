#!/bin/bash
set -e



echo "run train_config_133_43...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_43



echo "run train_config_133_44...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_44


echo 'done'
