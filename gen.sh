#!/bin/bash

#python3 ./notebook.py 

util/parallelizer.py -s 12 ffm_converter.py new_train.csv  new_val.csv new_train.sp  new_val.sp

./mark/mark1/mark1 -r 0.03 -s 12 -t 13 new_val.sp new_train.sp 

util/parallelizer.py -s 12 ffm_converter.py new_test.csv  new_val.csv new_test.sp  new_val1.sp



#cd ../libffm/libffm/


#./ffm-train -p ../../talkingdata/val.sp -l 0.03  ../../talkingdata/training.sp model.bin






