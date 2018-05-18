#!/bin/bash
set -e



echo "run train_config_133_51...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_51



echo "run train_config_133_52...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_52


echo "run train_config_133_53...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_53



echo "run train_config_133_54...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_54


echo 'done'
