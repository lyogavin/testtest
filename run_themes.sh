#!/bin/bash
set -e


for i in `seq 120 124`;
do
        echo $i
        echo "run train_config_133_$i...."
        python3 -u ./train_and_predict_simplified.py -c train_config_133_$i


done






echo 'done'
