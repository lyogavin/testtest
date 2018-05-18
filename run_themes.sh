#!/bin/bash
set -e



echo "run train_config_133_62...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_62



echo "run train_config_133_63...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_63



echo 'done'
