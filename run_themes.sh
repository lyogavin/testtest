#!/bin/bash
set -e


#echo 120
#echo "run train_config_133_120...."
#python3 -u ./train_and_predict_simplified.py -c train_config_133_120

for i in `seq 146 146`;
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
