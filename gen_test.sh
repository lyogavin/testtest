#!/bin/bash

python3 ./notebook.py 

head -n 100 ./to_submit.csv > ./to_submit_val.csv

util/parallelizer.py -s 12 ffm_converter.py to_submit.csv  to_submit_val.csv to_submit.sp  to_submit_val.sp



#cd ../libffm/libffm/


#./ffm-train -p ../../talkingdata/val.sp -l 0.03  ../../talkingdata/training.sp model.bin






