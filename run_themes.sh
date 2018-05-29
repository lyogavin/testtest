#!/bin/bash
set -e



echo "run train_config_133_77...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_77



echo "run train_config_133_78...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_78



echo 'done'
