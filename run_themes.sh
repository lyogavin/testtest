#!/bin/bash
set -e


#echo 136
#echo "run train_config_133_136...."
#python3 -u ./train_and_predict_simplified.py -c train_config_133_136
#echo 141
#echo "run train_config_133_141...."
#python3 -u ./train_and_predict_simplified.py -c train_config_133_141
#echo 143
#echo "run train_config_133_143...."
#python3 -u ./train_and_predict_simplified.py -c train_config_133_143

for i in `seq 160 160`;
do
        echo $i
        echo "run train_config_133_$i...."
        python3 -u ./train_and_predict_simplified.py -c train_config_133_$i


done


if [ -f ./next.sh ]; then
    echo './next.sh exist run it.'
    ./next.sh
else
    echo './next.sh not found'
fi


echo 'done'
