#!/bin/bash
set -e

python3 -u ./notebook.py 

util/parallelizer.py -s 12 ffm_converter.py train_fe.csv val_fe.csv new_train.sp  new_val.sp

./mark/mark1/mark1 -r 0.11 -s 12 -t 40 new_val.sp new_train.sp 

python3 -u ./calculate_auc.py ./val_fe.csv ./new_val.sp.prd 
#./mark/mark1/mark1 -r 0.11 -s 12 -t 40 new_val.sp new_train.sp 

#python3 -u ./calculate_auc.py ./val_fe.csv ./new_val.sp.prd 

#head -n 1000 train_fe.csv > train_fe.csv.sample

#util/parallelizer.py -s 12 ffm_converter.py  test_fe.csv train_fe.csv.sample new_test.sp  new_train.sp.sample

#./mark/mark1/mark1 -r 0.078 -s 12 -t 40 new_test.sp new_train.sp


#kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f new_test.sp.prd -m "ffm"
