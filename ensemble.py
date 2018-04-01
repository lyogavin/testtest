# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
from sklearn import metrics
#from ggplot import *

import time

def get_dated_filename(filename):
    return '{}.{}_{}'.format(filename, time.strftime("%d-%m-%Y"), time.strftime("%X"))


print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

'''
Another CTR comp and so i suspect libffm will play its part, after all it is an atomic bomb for this kind of stuff.
A sci-kit learn inspired script to convert pandas dataframes into libFFM style data.

The script is fairly hacky (hey thats Kaggle) and takes a little while to run a huge dataset.
The key to using this class is setting up the features dtypes correctly for output (ammend transform to suit your needs)

Example below


'''

from sklearn.model_selection import train_test_split
import lightgbm as lgb
import sys
import gc

use_sample = True

gen_test_input = True

path = '../input/'
path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'
path_val = 'val.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

categorical = ['app', 'device', 'os', 'channel', 'hour']

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

skip = range(1, 140000000)
print("Loading Data")
# skiprows=skip,

import pickle

# gen lgbm val prediction

#lgb_model = lgb.Booster(model_file='model.txt')
#submit['is_attributed'] = lgb_model.predict(train[predictors1], num_iteration=lgb_model.best_iteration)



val = pd.read_csv('val.csv.01-04-2018_10:48:51', header=0, usecols=['is_attributed'])  # .sample(1000)

lgbm_prediction = pd.read_csv('val_prediction.csv.01-04-2018_10:59:18', header=0)  # .sample(1000)

ffm_prediction = pd.read_csv('new_val.sp.prd', header=0, usecols=['click'])  # .sample(1000)

print('val {}'.format(val.info()))
print('lgbm_prediction {}'.format( lgbm_prediction.info()))
print('ffm_prediction {}'.format(ffm_prediction.info()))


lgbm_prediction_values = lgbm_prediction['is_attributed'].values

fpr, tpr, thresholds = metrics.roc_curve(val['is_attributed'].values, lgbm_prediction_values)
lgbm_auc = metrics.auc(fpr, tpr)

ffm_prediction_values = ffm_prediction['click'].values
fpr, tpr, thresholds = metrics.roc_curve(val['is_attributed'].values,ffm_prediction_values)
ffm_auc = metrics.auc(fpr, tpr)

print('lgbm auc: {}, ffm auc: {}'.format(lgbm_auc, ffm_auc))

max_auc = 0
for ratio in range(0, 100, 1):
    i = ratio / 100.0
    fpr, tpr, thresholds = metrics.roc_curve(val['is_attributed'].values,
                                             (i * lgbm_prediction_values + (1-i) * ffm_prediction_values))
    auc = metrics.auc(fpr, tpr)
    if auc > max_auc:
        i_at_max_auc = i
        max_auc = auc
    print('ensembled auc of {}:{} is {}'.format(i, 1-i, auc))


lgbm_submission = pd.read_csv('submission_notebook.csv.01-04-2018_13:47:13', header = 0)
ffm_submision = pd.read_csv('new_test.sp.prd', header = 0, usecols=['click'])

ffm_submision['is_attributed'] = i_at_max_auc * lgbm_submission['is_attributed'] + \
                                 (1.0-i_at_max_auc) * ffm_submision['click']


ffm_submision.to_csv(get_dated_filename('ensemble_submission.csv'), index=False)
