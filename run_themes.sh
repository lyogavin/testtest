#!/bin/bash
set -e


echo "run train_config_133_36 rerun2...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_36



echo "run train_config_133_39...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_39



echo "run train_config_133_40...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_40



echo "run train_config_133_41...."

python3 -u ./train_and_predict_simplified.py -c train_config_133_41


echo 'done'
