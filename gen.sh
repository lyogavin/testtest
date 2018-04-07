#!/bin/bash

python3 -u ./notebook.py 

#./test_fe.csv.05-04-2018_12-36-48  ./train_fe.csv.05-04-2018_12-17-46  ./val_fe.csv.05-04-2018_12-29-46

util/parallelizer.py -s 12 ffm_converter.py train_fe.csv val_fe.csv new_train.sp  new_val.sp

./mark/mark1/mark1 -r 0.01 -s 12 -t 13 new_val.sp new_train.sp 

head -n 1000 train_fe.csv > train_fe.csv.sample

util/parallelizer.py -s 12 ffm_converter.py  test_fe.csv train_fe.csv.sample new_test.sp  new_train.sp.sample

./mark/mark1/mark1 -r 0.01 -s 12 -t 13 new_test.sp new_train.sp




#cd ../libffm/libffm/


#./ffm-train -p ../../talkingdata/val.sp -l 0.03  ../../talkingdata/training.sp model.bin






