# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
from sklearn import metrics
#from ggplot import *

from os import walk

import time

def get_dated_filename(filename):
    return '{}.{}'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))

#print(os.listdir("../input"))

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
import math


def logit(x):
    y = x / (1 - x)
    return math.log(y  if y>0 else 1)


def logistic(x):
    return 1 / (1 + math.exp(-x))


vlogit = np.vectorize(logit)
vlogistic = np.vectorize(logistic)

ratio_based_on_val = False

i_at_max_auc = 0.5

if ratio_based_on_val:

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
    for ratio in range(0, 11, 1):
        i = ratio / 10.0
        fpr, tpr, thresholds = metrics.roc_curve(val['is_attributed'].values,
                                                 vlogistic(i * vlogit(lgbm_prediction_values) + (1-i) * vlogit(ffm_prediction_values)))
        auc = metrics.auc(fpr, tpr)
        if auc > max_auc:
            i_at_max_auc = i
            max_auc = auc
        print('ensembled auc of {}:{} is {}'.format(i, 1-i, auc))


    lgbm_submission = pd.read_csv('submission_notebook.csv.01-04-2018_13:47:13', header = 0)
    ffm_submision = pd.read_csv('new_test.sp.prd', header = 0, usecols=['click'])
    #ffm_submision = pd.read_csv('submission_notebook.csv.02-04-2018_22:48:59', header = 0)

    lgbm_submission['is_attributed'] = vlogistic(i_at_max_auc * vlogit(lgbm_submission['is_attributed']) + \
                                     (1.0-i_at_max_auc) * vlogit(ffm_submision['click']))

ensemble_list = [{'file':'submission_notebook.csv.04-04-2018_03-33-26', 'click':False},
                 #{'file': 'new_test.sp.prd', 'click': True},
                 #{'file':'submission_notebook.csv.03-04-2018_01:32:23', 'click':False},
                 {'file': 'submission_notebook.csv.04-04-2018_03-34-54', 'click': False}]


def logistic_func(x):
    return 1/(1+math.exp(-x))

def inv_logistic_func(x):
    return math.log(x/(1-x))

mprd = collections.defaultdict(list)

#for ensemble in ensemble_list:
ensemble_dir_path = './ensemble_predictions'

f = []

for (dirpath, dirnames, filenames) in walk(ensemble_dir_path):
    f.extend(filenames)
    break

for filename in f:
    print('processing ', filename)
    lgbm_submission = pd.read_csv('%s/%s' % (ensemble_dir_path,filename), header = 0)
    #ffm_submision = pd.read_csv('new_test.sp.prd', header = 0, usecols=['click'])
    #ffm_submision = pd.read_csv('submission_notebook.csv.02-04-2018_22:48:59', header = 0)
    #print('min:', lgbm_submission['is_attributed'].min())
    #if sum is None:
    #    sum = lgbm_submission
    #else:
    #    sum['is_attributed'] = vlogit(sum['is_attributed']) + vlogit(lgbm_submission['is_attributed'])

    i = 0
    for index, row in lgbm_submission.reset_index().iterrows():
        i = i+1
        if i % 10000 ==0:
            print('processing... ', i)

        value = row['is_attributed']
        if value == 0:
            value = 0.00001
        mprd[row['click_id']].append(value)


for key in mprd:
    mprd[key] = logistic_func(sum(map(inv_logistic_func, mprd[key]))/len(mprd[key]))

    

res = pd.DataFrame({'click_id':list(mprd.keys()),'is_attributed':list(mprd.values())})
res['click_id'] = res['click_id'].astype('uint32')

res.to_csv(get_dated_filename('ensemble_submission.csv'), index=False)
