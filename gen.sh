#!/bin/bash
set -e


echo "feature extraction...."

#touch /tmp/$1_test


python3 -u ./train_and_predict_refined.py #./notebook.py

echo "FFM input data generation...."

util/parallelizer.py -s 12 ffm_converter.py train_fe.csv val_fe.csv new_train.sp  new_val.sp

echo "FFM training and val scoring"

./mark/mark1/mark1 -r 0.11 -s 12 -t 40 new_val.sp new_train.sp

python3 -u ./calculate_auc.py ./val_fe.csv ./new_val.sp.prd


if [ "$1" == "stack_val_only" ]; then
    echo "stack val only..."

else
    echo "FFM input data generating for test..."

    head -n 1000 train_fe.csv > train_fe.csv.sample

    util/parallelizer.py -s 12 ffm_converter.py  test_fe.csv train_fe.csv.sample new_test.sp  new_train.sp.sample

    echo "FFM test scoring...."

    ./mark/mark1/mark1 -r 0.11 -s 12 -t 40 new_test.sp new_train.sp

    #kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f new_test.sp.prd -m "ffm"
fi

echo 'done'
