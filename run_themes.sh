#!/bin/bash
set -e



echo "run train_config_133_75...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_75



echo "run train_config_133_76...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_76



echo 'done'
