# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys

on_kernel = False

if on_kernel:
    sys.path.insert(0, '../input/wordbatch-133/wordbatch/')
    sys.path.insert(0, '../input/randomstate/randomstate/')
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from dateutil import parser
import matplotlib
from pprint import pprint
import wordbatch
from wordbatch.extractors import WordHash
from wordbatch.models import FM_FTRL
from wordbatch.models import FTRL
from wordbatch.models import NN_ReLU_H1
from pathlib import Path

from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
# from wordbatch.data_utils import *
import threading
from sklearn.metrics import roc_auc_score

matplotlib.use('Agg')


def get_dated_filename(filename):
    return '{}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))
    # return filename


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
from contextlib import contextmanager

import os, psutil


def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    gc.collect()
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)

    return memoryUse

@contextmanager
def timer(name):
    t0 = time.time()
    print('start ',name)
    yield
    print('[{}] done in {} s'.format(name, time.time() - t0))
    print('mem after {}: {}'.format(name, cpuStats()))


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

from pympler import muppy
from pympler import summary

use_sample = False
persist_intermediate = False
print_verbose = False

gen_test_input = True

read_path_with_hist = False

TRAIN_SAMPLE_DATA_LEN = 100001

# path = '../input/'
path = '../input/talkingdata-adtracking-fraud-detection/'
path_train_hist = '../input/data_with_hist/'
path_test_hist = '../input/data_with_hist/'

ft_cache_path = '../input/ft_cache/'
try:
    os.mkdir( ft_cache_path )
except:
    None

path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

#categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
categorical = ['app', 'device', 'os', 'channel', 'hour']
# with ip:
# categorical = ['app', 'device', 'os', 'channel', 'hour', 'ip']


cvr_columns_lists = [
    ['ip', 'app', 'device', 'os', 'channel'],
    # ['app', 'os'],
    # ['app','channel'],
    # ['app', 'device'],
    ['ip', 'device'],
    ['ip']
    # , ['os'], ['channel']
]
agg_types = ['non_attr_count', 'cvr']

acro_names = {
    'ip': 'I',
    'app': 'A',
    'device': 'D',
    'os': 'O',
    'channel': 'C',
    'hour': 'H',
    'ip_day_hourcount': 'IDH-',
    'ip_day_hour_oscount': 'IDHO-',
    'ip_day_hour_appcount': 'IDHA-',
    'ip_day_hour_app_oscount': 'IDHAO-',
    'ip_app_oscount': "IAO-",
    'ip_appcount': "IA-",
    'ip_devicecount': "ID-",
    'app_channelcount': "AC-",
    'app_day_hourcount': 'ADH-',
    'ip_in_test_hhcount': "IITH-",
    'next_click': 'NC',
    'app_channel': 'AC',
    'os_channel': 'OC',
    'app_device': 'AD',
    'app_os_channel': 'AOC',
    'ip_app': 'IA',
    'app_os': 'AO'
}

hist_st = []
iii = 0
for type in agg_types:
    for cvr_columns in cvr_columns_lists:
        new_col_name = '_'.join(cvr_columns) + '_' + type
        hist_st.append(new_col_name)

field_sample_filter_channel_filter = {'filter_type': 'filter_field',
                                      'filter_field': 'channel',
                                      'filter_field_values': [107, 477, 265]}
field_sample_filter_app_filter1 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [12, 18, 14]}
field_sample_filter_app_filter2 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [12]}
field_sample_filter_app_filter3 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [18, 14]}
field_sample_filter_app_filter4 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [8, 11]}

train_time_range_start = '2017-11-09 04:00:00'
train_time_range_end = '2017-11-09 15:00:00'

val_time_range_start = '2017-11-08 04:00:00'
val_time_range_end = '2017-11-08 15:00:00'

test_time_range_start = '2017-11-10 04:00:00'
test_time_range_end = '2017-11-10 15:00:00'

id_8_4am = 82259195
id_8_3pm = 118735619
id_9_4am = 144708152
id_9_3pm = 181878211

public_train_from = 109903890
public_train_to =  147403890
public_val_from = 147403890
public_val_to = 149903890

default_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 7,
    'max_depth': 4,
    'min_child_samples': 100,
    'max_bin': 150,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'nthread': 5,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}

new_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 9,
    'max_depth': 5,
    'min_child_samples': 100,
    'max_bin': 150,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'nthread': 5,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}

public_kernel_lgbm_params = dict(new_lgbm_params)
public_kernel_lgbm_params.update({
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced
    })

new_lgbm_params1 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 9,
    'max_depth': 5,
    'min_child_samples': 100,
    'max_bin': 150,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'nthread': 5,
    'verbose': 9,
    'early_stopping_round': 100, #20,
    # 'is_unbalance': True,
    'scale_pos_weight': 200.0
}

new_new_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.2,
    # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,
    'verbose': 0,
    'scale_pos_weight': 200.0
}

lgbm_params_from_search_0_35 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 0.5758903957135874, 'learning_rate': 1.1760632807147045, 'max_depth': 9,
 'min_child_samples': 200, 'min_child_weight': 6, 'min_split_gain': 1.0, 'n_estimators': 86, 'num_leaves': 31,
 'reg_alpha': 8.954987962970492, 'reg_lambda': 1000.0, 'scale_pos_weight': 6.1806180811037486, 'subsample': 1.0,
 'subsample_for_bin': 740701, 'subsample_freq': 0}


lgbm_params_from_search_2_11 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 0.01, 'learning_rate': 1.040731554567982, 'max_depth': 0, 'min_child_samples': 51,
    'min_child_weight': 7, 'min_split_gain': 1.0109536459124555, 'n_estimators': 300, 'num_leaves': 31,
    'reg_alpha': 1.2420399086947274, 'reg_lambda': 1000.0, 'scale_pos_weight': 4.652061504081354,
    'subsample': 1.0, 'subsample_for_bin': 571876, 'subsample_freq': 0}


lgbm_params_from_search_96_8_10 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 0.6021467085338628, 'learning_rate': 0.37861782981986614, 'max_depth': 5,
    'min_child_samples': 171, 'min_child_weight': 1, 'min_split_gain': 0.004337458691335552,
    'n_estimators': 249, 'num_leaves': 10, 'reg_alpha': 8.1447463220916e-05, 'reg_lambda': 64.24440555484793,
    'scale_pos_weight': 4.350080991126866, 'subsample': 0.5332391076871872, 'subsample_for_bin': 470584,
    'subsample_freq': 1
}


lgbm_params_from_search_0_81 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 1.0,
    'learning_rate': 1.3817527732999606,
    'max_depth': 7,
    'min_child_samples': 92,
    'min_child_weight': 10,
    'min_split_gain': 1.0096460744834064,
    'n_estimators': 61,
    'num_leaves': 26,
    'reg_alpha': 6.082283392201092,
    'reg_lambda': 1000.0,
    'scale_pos_weight': 1.9673649490776584,
    'subsample': 1.0,
    'subsample_for_bin': 800000,
    'subsample_freq': 1
}

lgbm_params_from_search_101 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 9,
    'min_child_weight': 3, 'subsample_for_bin': 542556, 'learning_rate': 0.30803266868575857,
    'subsample_freq': 0, 'max_depth': 8, 'subsample': 1.0, 'num_leaves': 7, 'reg_lambda': 28.66709162037324,
    'early_stopping_round': 371, 'min_split_gain': 0.0028642748250716932, 'reg_alpha': 7.351107378081451e-05,
    'scale_pos_weight': 29.983824928443436, 'colsample_bytree': 0.8639270134191158, 'min_child_samples': 200}

shuffle_sample_filter = {'filter_type': 'sample', 'sample_count': 6}
shuffle_sample_filter_1_to_2 = {'filter_type': 'sample', 'sample_count': 2}

shuffle_sample_filter_1_to_10 = {'filter_type': 'sample', 'sample_count': 10}
shuffle_sample_filter_1_to_20 = {'filter_type': 'sample', 'sample_count': 20}
shuffle_sample_filter_1_to_10k = {'filter_type': 'sample', 'sample_count': 1}

hist_ft_sample_filter = {'filter_type': 'hist_ft'}

skip = range(1, 140000000)
print("Loading Data")
# skiprows=skip,

import pickle

search_features_list = [

    # ====================
    # my best features
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'channel', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'os', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['channel', 'hour', 'ip'], 'op': 'var'},
    {'group': ['app', 'os', 'channel', 'ip'], 'op': 'skew'},
    {'group': ['app', 'channel', 'ip'], 'op': 'mean'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

best_single_group_in_search = [
    {'group': ['app', 'device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip'], 'op': 'mean'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

ft_coms_97478=[
    {'group': ['app', 'device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip'], 'op': 'var'},
    {'group': ['device', 'os', 'ip'], 'op': 'mean'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os','hour','ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_from_public=[
    # importance 27 .. 4:
    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'cumcount'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique'},


    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},

    {'group': ['ip', 'os'], 'op': 'cumcount'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique'},

    # count:
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},

    # var:
    {'group': ['ip','day','channel','hour'], 'op': 'var'},
    {'group': ['ip','app', 'os', 'hour'], 'op': 'var'},
    {'group': ['ip','app', 'channel', 'day'], 'op': 'var'},

    # mean:
    {'group': ['ip','app', 'channel','hour'], 'op': 'mean'},

    #{'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

ft_coms_from_public_astype=[
    # importance 27 .. 4:
    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'cumcount'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique'},


    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},

    {'group': ['ip', 'os'], 'op': 'cumcount'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique'},

    # count:
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count', 'astyep':'uint16'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count', 'astyep':'uint16'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count', 'astyep':'uint16'},

    # var:
    {'group': ['ip','day','channel','hour'], 'op': 'var'},
    {'group': ['ip','app', 'os', 'hour'], 'op': 'var'},
    {'group': ['ip','app', 'channel', 'day'], 'op': 'var'},

    # mean:
    {'group': ['ip','app', 'channel','hour'], 'op': 'mean'},

    #{'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

ft_coms_search_99_1=[
    # importance 27 .. 4:
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    # importance 3:
    {'group': ['app', 'channel', 'hour'], 'op': 'cumcount'},
    {'group': ['app', 'hour'], 'op': 'mean'},
    {'group': ['device', 'os', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_search_99_2_2=[
    # importance 4:
    {'group': ['device', 'channel'], 'op': 'cumcount'},
    {'group': ['device', 'os', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'os', 'channel', 'ip'], 'op': 'count'},
    {'group': ['app', 'ip', 'is_attributed'], 'op': 'count'},
    # importance 27 .. 5:
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    # importance 3:

    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_search_99_2=[
    # importance 27 .. 5:
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    # importance 3:

    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
add_features_list_origin = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count', 'astyep':'uint16'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
    # =====================
    # try word batch featuers:
    # =====================
    # {'group': ['ip', 'day', 'hour'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['ip', 'app'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['ip', 'app', 'os'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['ip', 'device'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['app', 'channel'], 'with_hist': False, 'counting_col': 'os'},
    # ======================

    # {'group':['app'], 'with_hist': False, 'counting_col':'channel'},
    # {'group': ['os'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['device'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['channel'], 'with_hist': False, 'counting_col': 'os'},
    # {'group': ['hour'], 'with_hist': False, 'counting_col': 'os'},

    # {'group':['ip','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip','os', 'app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip','hour','channel'], 'with_hist': with_hist_profile, 'counting_col':'os'},
    # {'group':['ip','hour','os'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip','hour','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['channel','app'], 'with_hist': with_hist_profile, 'counting_col':'os'},
    # {'group':['channel','os'], 'with_hist': with_hist_profile, 'counting_col':'app'},
    # {'group':['channel','app','os'], 'with_hist': with_hist_profile, 'counting_col':'device'},
    # {'group':['os','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
]



class ConfigScheme:
    def __init__(self, predict=False, train=True, ffm_data_gen=False,
                 train_filter=None,
                 val_filter=shuffle_sample_filter,
                 test_filter=None,
                 lgbm_params=default_lgbm_params,
                 discretization=0,
                 mock_test_with_val_data_to_test=False,
                 train_start_time=train_time_range_start,
                 train_end_time=train_time_range_end,
                 val_start_time=val_time_range_start,
                 val_end_time=val_time_range_end,
                 gen_ffm_test_data=False,
                 add_hist_statis_fts=False,
                 seperate_hist_files=False,
                 train_wordbatch=False,
                 log_discretization=False,
                 predict_wordbatch=False,
                 use_interactive_features=False,
                 wordbatch_model='FM_FTRL',
                 train_wordbatch_streaming=False,
                 new_train=False,
                 train_from=None,
                 train_to=None,
                 val_from=None,
                 val_to=None,
                 new_predict = False,
                 run_theme = '',
                 add_features_list = add_features_list_origin):
        self.predict = predict
        self.train = train
        self.ffm_data_gen = ffm_data_gen
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.lgbm_params = lgbm_params
        self.discretization = discretization
        self.mock_test_with_val_data_to_test = mock_test_with_val_data_to_test
        self.train_start_time = train_start_time
        self.train_end_time = train_end_time
        self.val_start_time = val_start_time
        self.val_end_time = val_end_time
        self.gen_ffm_test_data = gen_ffm_test_data
        self.add_hist_statis_fts = add_hist_statis_fts
        self.seperate_hist_files = seperate_hist_files
        self.train_wordbatch = train_wordbatch
        self.log_discretization = log_discretization
        self.predict_wordbatch = predict_wordbatch
        self.use_interactive_features = use_interactive_features
        self.wordbatch_model = wordbatch_model
        self.train_wordbatch_streaming = train_wordbatch_streaming
        self.new_train = new_train
        self.train_from=train_from
        self.train_to=train_to
        self.val_from=val_from
        self.val_to=val_to
        self.new_predict = new_predict
        self.run_theme = run_theme
        self.add_features_list = add_features_list


train_config_94_8 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_2,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )

train_config_94_13 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_2_11,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )
train_config_97 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_2_11,
                                 new_train= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_gen'
                                 )

train_config_94_14 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_96_8_10,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )

train_config_98 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms'
                                  )
train_config_96 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='lgbm_params_search'
                                  )

train_config_99 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_20,
                                 shuffle_sample_filter_1_to_20,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms'
                                  )
train_config_99_4 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_20,
                                 shuffle_sample_filter_1_to_20,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms_plus_lgbm_searcher'
                                  )
train_config_100 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_97478
                                  )
train_config_100_3 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_search_99_1
                                  )


train_config_102 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_101,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )


train_config_103 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict'
                                 )

train_config_104 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=public_train_from,
                                 train_to=public_train_to,
                                 val_from=public_val_from,
                                 val_to=public_val_to,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )
train_config_105 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )

train_config_106 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )
train_config_106_3 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )
train_config_106_5 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )

train_config_106_6 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )

train_config_106_7 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=ft_coms_from_public
                                  )

train_config_108 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )

train_config_108_2 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )


train_config_108_3 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_2,
                                  shuffle_sample_filter_1_to_2,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=public_train_from,
                                 train_to=public_train_to,
                                 val_from=public_val_from,
                                 val_to=public_val_to,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )


train_config_108_4 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_2,
                                  shuffle_sample_filter_1_to_2,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )

train_config_106_8 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )




train_config_103_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict'
                                 )
train_config_103_6 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=ft_coms_from_public
                                 )
train_config_103_7 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=ft_coms_from_public_astype
                                 )
def use_config_scheme(str):
    print('config values: ')
    pprint(vars(eval(str)))
    print('using config var name: ', str)
    return eval(str)


config_scheme_to_use = use_config_scheme('train_config_103_7')

print('test log 103_7')

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}
if config_scheme_to_use.add_hist_statis_fts:
    for ft in hist_st:
        dtypes[ft] = 'float32'

if config_scheme_to_use.add_hist_statis_fts and not config_scheme_to_use.seperate_hist_files:
    path_train = path_train_hist + 'train_with_cvr.csv.gzip'
    path_train_sample = path_train_hist + 'train_with_cvr_sample.csv.gzip'
    path_test = path_test_hist + 'test_with_cvr.csv.gzip'
    path_test_sample = path_test_hist + 'test_with_cvr_sample.csv'

    train_cols = train_cols + hist_st
    test_cols = test_cols + hist_st


def gen_categorical_features(data):
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    print("Creating new time features in train: 'hour' and 'day'...")
    data['hour'] = data["click_time"].dt.hour.astype('uint8')
    data['day'] = data["click_time"].dt.day.astype('uint8')

    add_hh_feature = True
    if add_hh_feature:
        data['in_test_hh'] = (3
                              - 2 * data['hour'].isin(most_freq_hours_in_test_data)
                              - 1 * data['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')
        # categorical.append('in_test_hh')
    return data


def gen_iteractive_categorical_features(data):
    if config_scheme_to_use.use_interactive_features:
        interactive_features_list = [
            ['app', 'channel'],
            ['os', 'channel'],
            ['app', 'device'],
            ['app', 'os', 'channel'],
            ['ip', 'app'],
            ['app', 'os']
        ]

        for interactive_feature_items in interactive_features_list:
            interactive_features_name = '_'.join(interactive_feature_items)
            first_setting = True
            for interactive_feature_item in interactive_feature_items:
                if first_setting:
                    data[interactive_features_name] = data[interactive_feature_item].astype(str)
                    first_setting = False
                else:
                    data[interactive_features_name] = data[interactive_features_name] + \
                                                      '_' + data[interactive_feature_item].astype(str)
            if not interactive_features_name in categorical:
                categorical.append(interactive_features_name)
    return data


def post_statistics_features(data):
    return data



def df_get_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    return counts[unqtags]


def add_statistic_feature(group_by_cols, training, qcut_count=0, #0.98,
                          discretization=0, discretization_bins=None,
                          log_discretization=False,
                          op='count',
                          use_ft_cache = False,
                          ft_cache_prefix = '',
                          only_ft_cache = False,
                          astype=None):
    feature_name_added = '_'.join(group_by_cols) + op

    ft_cache_file_name = ft_cache_prefix + '_' + feature_name_added
    ft_cache_file_name = ft_cache_file_name + '_sample' if use_sample else ft_cache_file_name
    ft_cache_file_name = ft_cache_file_name + '.csv.bz2'

    if use_ft_cache and Path(ft_cache_path + ft_cache_file_name).is_file():
        ft_cache_data = pd.read_csv(ft_cache_path + ft_cache_file_name,
                                    dtype='float32',
                                    header=0, engine='c',
                                    compression='bz2')
        training[feature_name_added] = ft_cache_data
        print('loaded {} from file {}'.format(feature_name_added, ft_cache_path + ft_cache_file_name))
        return training, [feature_name_added], None


    counting_col = group_by_cols[len(group_by_cols) - 1]
    group_by_cols = group_by_cols[0:len(group_by_cols) - 1]
    features_added = []
    discretization_bins_used = {}
    print('count ip with group by:', group_by_cols)

    if op == 'cumcount':
        gp = training[group_by_cols + [counting_col]].\
            groupby(by=group_by_cols)[[counting_col]].cumcount()
        training[feature_name_added] = gp.values
    elif op=='nextclick':
        with timer("Adding next click times"):
            D = 2 ** 26
            #data['category'] = (data['ip'].astype(str) + "_" + data['app'].astype(str) + "_" + \
            #                    data['device'].astype(str) \
            #                    + "_" + data['os'].astype(str) + "_" + data['channel'].astype(str)) \
            #                       .apply(hash) % D
            joint_col = None

            for col in group_by_cols:
                if joint_col is None:
                    joint_col = training[col].astype(str)
                else:
                    joint_col = joint_col + "_" + training[col].astype(str)
            training['category'] = joint_col.apply(hash) % D
            del joint_col
            gc.collect()
            click_buffer = np.full(D, 3000000000, dtype=np.uint32)
            training['epochtime'] = training['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, time in zip(reversed(training['category'].values),
                                      reversed(training['epochtime'].values)):
                next_clicks.append(click_buffer[category] - time)
                click_buffer[category] = time
            del (click_buffer)
            training[feature_name_added] = list(reversed(next_clicks))

            training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            print('next click added:', training[feature_name_added].describe())



    else:
        tempstr = 'training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]]'
        temp1 = eval(tempstr + '.' + op + '()')
        n_chans = temp1.reset_index().rename(columns={counting_col: feature_name_added})
        training = training.merge(n_chans, on=group_by_cols if len(group_by_cols) >1 else group_by_cols[0],
                                  how='left')
        del n_chans

    gc.collect()
    if not log_discretization and discretization == 0:
        if training[feature_name_added].max() <= 65535 and \
            op in ['count', 'nunique','cumcount']:
            training[feature_name_added] = training[feature_name_added].astype('uint16')

    qcut_op_types = ['count', 'nextclick', 'nunique']
    if not log_discretization and qcut_count != 0 and discretization == 0 and op in qcut_op_types:
        colmax = training[feature_name_added].max()
        quantile_cut = training[feature_name_added].quantile(qcut_count)
        training[feature_name_added] = training[feature_name_added].apply(
            lambda x: x if x < quantile_cut else quantile_cut)
            # fix colmax transform to test
            # lambda x: x if x < quantile_cut else colmax)


    if log_discretization:
        if training[feature_name_added].min() < 0:
            print('!!!! invalid time in {}, fix it.....'.format(feature_name_added))
            training[feature_name_added] = training[feature_name_added].apply(lambda x: np.max([0, x]))
        training[feature_name_added] = np.log2(1 + training[feature_name_added].values).astype(int)
        print('log dicretizing feature:', feature_name_added)
    elif discretization != 0:
        if print_verbose:
            print('before qcut', feature_name_added, training[feature_name_added].describe())
        if discretization_bins is None:
            ret, discretization_bins_used[feature_name_added] = pd.qcut(training[feature_name_added], discretization,
                                                                        labels=False, duplicates='drop', retbins=True)
            training[feature_name_added] = ret.fillna(0).astype('uint16')
        else:
            training[feature_name_added] = pd.cut(training[feature_name_added],
                                                  discretization_bins[feature_name_added],
                                                  labels=False).fillna(0).astype('uint16')
        if print_verbose:
            print('after qcut', feature_name_added, training[feature_name_added].describe())

    features_added.append(feature_name_added)
    for ft in feature_name_added:
        if astype is not None:
            training[ft] = training[ft].astype(astype)

    print('added features:', features_added)
    if print_verbose:
        print(training[feature_name_added].describe())
    print('nan count: ', training[feature_name_added].isnull().sum())

    print('columns after added: ', training.columns.values)

    if use_ft_cache:
        pd.DataFrame(training[feature_name_added]).to_csv(ft_cache_path + ft_cache_file_name,
                                                          index=False,compression='bz2')
        print('saved {} to file {}'.format(feature_name_added, ft_cache_path + ft_cache_file_name))
        if only_ft_cache:
            del training[feature_name_added]
            del features_added[-1]
            gc.collect()

    return training, features_added, discretization_bins_used


def generate_counting_history_features(data,
                                       discretization=0, discretization_bins=None,
                                       add_features_list=None,
                                       use_ft_cache = False,
                                       ft_cache_prefix = '',
                                       only_ft_cache = False):
    print('discretization bins to use:', discretization_bins)

    new_features = []

    discretization_bins_used = None

    for add_feature in add_features_list:
        with timer('adding feature:' + str(add_feature)):
            data, features_added, discretization_bins_used_current_feature = add_statistic_feature(
                add_feature['group'],
                data,
                discretization=discretization,
                discretization_bins=discretization_bins,
                log_discretization=config_scheme_to_use.log_discretization,
                op = add_feature['op'],
                use_ft_cache = use_ft_cache,
                ft_cache_prefix = ft_cache_prefix,
                only_ft_cache = only_ft_cache,
                astype = add_feature['astype'] if 'astype' in add_feature else None)
            new_features = new_features + features_added
            if discretization_bins_used_current_feature is not None:
                if discretization_bins_used is None:
                    discretization_bins_used = {}
                discretization_bins_used = \
                    dict(list(discretization_bins_used.items()) + list(discretization_bins_used_current_feature.items()))
            gc.collect()

    if discretization_bins is None:
        print('discretization bins used:', discretization_bins_used)
    else:
        print('discretizatoin bins passed in params, so no discretization_bins_used returned')

    data = post_statistics_features(data)

    # add next click feature:
    add_next_click = False
    if add_next_click:
        with timer("Adding next click times"):
            D = 2 ** 26
            data['category'] = (data['ip'].astype(str) + "_" + data['app'].astype(str) + "_" + \
                                data['device'].astype(str) \
                                + "_" + data['os'].astype(str) + "_" + data['channel'].astype(str)).apply(hash) % D
            click_buffer = np.full(D, 3000000000, dtype=np.uint32)
            data['epochtime'] = data['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, time in zip(reversed(data['category'].values), reversed(data['epochtime'].values)):
                next_clicks.append(click_buffer[category] - time)
                click_buffer[category] = time
            del (click_buffer)
            data['next_click'] = list(reversed(next_clicks))

            if discretization != 0:
                print('min of next click: {}, max: {}'.format(data['next_click'].min(), data['next_click'].max()))
                if data['next_click'].min() < 0:
                    print('!!!! invalid time in next click, fix it.....')
                    data['next_click'] = data['next_click'].apply(lambda x: np.max([0, x]))
                data['next_click'] = np.log2(1 + data['next_click'].values).astype(int)
            data.drop('epochtime', inplace=True, axis=1)
            data.drop('category', inplace=True, axis=1)

            #print('next click ', data['next_click'])
            if print_verbose:
                print(data['next_click'].describe())

            new_features = new_features + ['next_click']

    gc.collect()

    click_count_later = False
    if click_count_later:
        # build current day profile features:
        with timer("building current day profile features"):
            D = 2 ** 26
            data['category'] = (data['ip'].astype(str) + "_" + data['app'].astype(str) + "_" + \
                                data['device'].astype(str) \
                                + "_" + data['os'].astype(str) + "_" + data['channel'].astype(str)).apply(hash) % D
            click_count_buffer = np.full(D, 0, dtype=np.uint16)
            click_count_later = []
            for category in reversed(data['category'].values):
                click_count_later.append(click_count_buffer[category])
                click_count_buffer[category] += 1
            del (click_count_buffer)
            data['click_count_later'] = list(reversed(click_count_later))
        gc.collect()
        new_features = new_features + ['click_count_later']

    print('data dtypes:',data.dtypes)
    return data, new_features, discretization_bins_used


# test['hour'] = test["click_time"].dt.hour.astype('uint8')
# test['day'] = test["click_time"].dt.day.astype('uint8')


def convert_features_to_text(data, predictors):
    with timer('convert_features_to_text'):
        i = 0
        str_array = None
        for feature in predictors:
            if not feature in acro_names:
                print('{} missing acronym'.format(feature))
                exit(-1)
            if str_array is None:
                str_array = acro_names[feature] + "_" + data[feature].astype(str)
            else:
                str_array = str_array + " " + acro_names[feature] + "_" + data[feature].astype(str)

            gc.collect()
            print('mem after gc:', cpuStats())
            i += 1

        str_array = str_array.values
        return str_array


def train_lgbm(train, val, new_features, do_val_prediction=False):

    target = 'is_attributed'

    predictors1 = categorical + new_features

    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st

    gc.collect()

    print("Preparing the datasets for training...")

    predictors_to_train = [predictors1]

    for predictors in predictors_to_train:
        print('training with :', predictors)
        dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical
                             )
        dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical
                             )

        evals_results = {}
        print("Training the model...")

        lgb_model = lgb.train(config_scheme_to_use.lgbm_params,
                              dtrain,
                              valid_sets=[dtrain, dvalid],
                              valid_names=['train', 'valid'],
                              evals_result=evals_results,
                              num_boost_round=1000,
                              early_stopping_rounds=30,
                              verbose_eval=50,
                              feval=None)

        # Nick's Feature Importance Plot
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=[7, 10])
        # lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
        # plt.title("Light GBM Feature Importance")
        # plt.savefig('feature_import.png')

        # Feature names:
        print('Feature names:', lgb_model.feature_name())
        # Feature importances:
        print('Feature importances:', list(lgb_model.feature_importance()))
        try:
            pprint(sorted(zip(lgb_model.feature_name(),list(lgb_model.feature_importance())),
                          key=lambda x: x[1]))
        except:
            print('error sorting and zipping fts')

        importance_dict = dict(zip(lgb_model.feature_name(), list(lgb_model.feature_importance())))

        feature_imp = pd.DataFrame(lgb_model.feature_name(), list(lgb_model.feature_importance()))

        if persist_intermediate:
            print('dumping model')
            lgb_model.save_model(get_dated_filename('model.txt'))

        val_prediction = None
        if do_val_prediction:
            print("Writing the val_prediction into a csv file...")
            # if persist_intermediate:

            print('gen val prediction')
            val_prediction = lgb_model.predict(val[predictors1], num_iteration=lgb_model.best_iteration)
            # pd.Series(val_prediction).to_csv(get_dated_filename("val_prediction.csv"), index=False)
            val_auc = roc_auc_score(val['is_attributed'], val_prediction)
            if persist_intermediate:
                val['predict'] = val_prediction
                val.to_csv(get_dated_filename("val_prediction.csv"), index=False)

    if do_val_prediction:
        return lgb_model, val_prediction, predictors1, importance_dict, val_auc
    else:
        return lgb_model, val_prediction, predictors1, importance_dict, 0



# In[ ]:



def gen_ffm_data():
    train, val, new_features, discretization_bins_used = gen_train_df(False, True)
    train_len = len(train)
    val_len = len(val)
    gc.collect()
    test_len = 0
    if config_scheme_to_use.gen_ffm_test_data:
        test, _ = gen_test_df(False, True, discretization_bins_used)
        test_len = len(test)
    gc.collect()

    print('train({}) val({}) test({}) generated'.format(train_len, val_len, test_len))


    # train = train.append(val)
    # test = train.append(test)

    # del train
    # del val

    # gc.collect()

    # print(test)

def get_train_df():
    train = pd.read_csv(path_train_sample if use_sample else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.train_from) \
                            if not use_sample and config_scheme_to_use.train_from is not None else None,
                        nrows=config_scheme_to_use.train_to - config_scheme_to_use.train_from \
                            if not use_sample and config_scheme_to_use.train_from is not None else None,
                        parse_dates=["click_time"])

    print('mem after loaded train data:', cpuStats())

    if config_scheme_to_use.train_filter and \
                    config_scheme_to_use.train_filter['filter_type'] == 'sample':
        sample_count = config_scheme_to_use.train_filter['sample_count']
        train = train.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        gc.collect()
        print('mem after filtered train data:', cpuStats())

    gc.collect()
    return train

def get_val_df():
    val = pd.read_csv(path_train_sample if use_sample else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.val_from) \
                            if not use_sample and config_scheme_to_use.val_from is not None else None,
                        nrows=config_scheme_to_use.val_to - config_scheme_to_use.val_from \
                            if not use_sample and config_scheme_to_use.val_from is not None else None,
                        parse_dates=["click_time"])

    print('mem after loaded val data:', cpuStats())

    if config_scheme_to_use.val_filter and \
        config_scheme_to_use.val_filter['filter_type'] == 'sample':
        sample_count = config_scheme_to_use.val_filter['sample_count']
        val = val.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        gc.collect()
        print('mem after filtered val data:', cpuStats())
    gc.collect()
    return val

def get_test_df():
    test = pd.read_csv(path_test_sample if use_sample else path_test,
                       dtype=dtypes,
                       header=0,
                       usecols=test_cols,
                       parse_dates=["click_time"])
    test['is_attributed'] = 0
    return test

def get_combined_df(gen_test_data):

    print('loading train data...')
    train = get_train_df()
    train_len = len(train)

    print('loading val data')
    val = get_val_df()
    val_len = len(val)

    train = train.append(val)

    del val
    gc.collect()
    print('mem after appended val data:', cpuStats())

    if gen_test_data:
        test = get_test_df()
        train = train.append(test)
        del test
        gc.collect()

    return train, train_len, val_len

def train_and_predict(com_fts_list, use_ft_cache = False, only_cache=False,
                                         use_base_data_cache=False):
    with timer('load combined data df'):
        combined_df, train_len, val_len = get_combined_df(config_scheme_to_use.new_predict)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache = use_ft_cache,
                                           add_features_list=com_fts_list)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]

    with timer('train lgbm model...'):
        lgb_model, val_prediction, predictors, importances, val_auc = train_lgbm(train, val, new_features, False)

    if config_scheme_to_use.new_predict:
        with timer('predict test data:'):
            test = combined_df[train_len + val_len:]

            predict_result = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)
            submission = pd.DataFrame({'is_attributed':predict_result,
                                       'click_id':test['click_id'].astype('uint32').values})

            print("Writing the submission data into a csv file...")

            submission.to_csv(get_dated_filename("submission_notebook"), index=False)

            print("All done...")

    return importances, val_auc


def gen_ft_caches_seperately(com_fts_list):

    with timer('loading train df:'):
        train = get_train_df()
    with timer('gen categorical features for train'):
        train = gen_categorical_features(train)
    with timer('gen statistical hist features for train'):
        train, new_features, discretization_bins_used = \
        generate_counting_history_features(train,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list,
                                           use_ft_cache = True,
                                           ft_cache_prefix='train',
                                           only_ft_cache = True)

    del train


    gc.collect()

    with timer('loading val df:'):
        val = get_val_df()
    with timer('gen categorical features for val'):
        val = gen_categorical_features(val)
    with timer('gen statistical hist features for val'):
        val, _, _ = \
        generate_counting_history_features(val,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list,
                                           discretization_bins=discretization_bins_used,
                                           use_ft_cache = True,
                                           ft_cache_prefix='val',
                                           only_ft_cache = True)
    gc.collect()

def train_and_predict_gen_fts_seperately(com_fts_list, use_ft_cache = False, only_cache=False,
                                         use_base_data_cache=False):

    if use_base_data_cache and \
            hasattr(train_and_predict_gen_fts_seperately, 'base_train_data_cache') and \
                    train_and_predict_gen_fts_seperately.base_train_data_cache is not None:
        with timer('loading base train data from cache:'):
            train = train_and_predict_gen_fts_seperately.base_train_data_cache.copy()
    else:
        print('no base data cache, load it.')
        with timer('loading train df:'):
            train = get_train_df()
        with timer('gen categorical features for train'):
            train = gen_categorical_features(train)

        if use_base_data_cache:
            with timer('setting base train data cache'):
                train_and_predict_gen_fts_seperately.base_train_data_cache = train.copy()


    with timer('gen statistical hist features for train'):
        train, new_features, discretization_bins_used = \
        generate_counting_history_features(train,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='train')

    gc.collect()
    if use_base_data_cache and \
            hasattr(train_and_predict_gen_fts_seperately, 'base_val_data_cache') and \
                    train_and_predict_gen_fts_seperately.base_val_data_cache is not None:
        with timer('loading base train data from cache:'):
            val = train_and_predict_gen_fts_seperately.base_val_data_cache.copy()
    else:
        with timer('loading val df:'):
            val = get_val_df()
        with timer('gen categorical features for val'):
            val = gen_categorical_features(val)
        if use_base_data_cache:
            with timer('setting base train data cache'):
                train_and_predict_gen_fts_seperately.base_val_data_cache = val.copy()


    with timer('gen statistical hist features for val'):
        val, _, _ = \
        generate_counting_history_features(val,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list,
                                           discretization_bins=discretization_bins_used,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='val')
    gc.collect()

    with timer('train lgbm model...'):
        if not only_cache:
            lgb_model, val_prediction, predictors, importances, val_auc = train_lgbm(train, val, new_features, False)

    del train
    del val
    gc.collect()
    print('mem after train and gc:', cpuStats())

    if config_scheme_to_use.new_predict and not only_cache:

        with timer('predict test data:'):
            with timer('loading test df:'):
                test = get_test_df()
            with timer('gen categorical features for test'):
                test = gen_categorical_features(test)
            with timer('gen statistical hist features for test'):
                test, _, _ = \
                    generate_counting_history_features(test,
                                                       discretization=config_scheme_to_use.discretization,
                                                       add_features_list=com_fts_list,
                                                       discretization_bins=discretization_bins_used,
                                                       use_ft_cache = use_ft_cache,
                                                       ft_cache_prefix='test')

            predict_result = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)
            submission = pd.DataFrame({'is_attributed':predict_result,
                                       'click_id':test['click_id'].astype('uint32').values})

            print("Writing the submission data into a csv file...")

            submission.to_csv(get_dated_filename("submission_notebook"), index=False)

            print("All done...")

    if only_cache:
        return {}, 0
    else:
        return importances, val_auc



if config_scheme_to_use.ffm_data_gen:
    gen_ffm_data()


def grid_search_features_combination(only_gen_ft_cache = False, use_lgbm_searcher = False):
    import itertools
    from random import shuffle

    # grid search for feature generations combinations:
    com_fts_list_to_use = []
    raw_cols0 = ['app', 'device', 'os', 'channel', 'hour', 'ip']
    raw_cols1 = ['app', 'device', 'os', 'channel', 'in_test_hh', 'ip']
    ops = ['mean','var','nextclick','nunique','cumcount']
    #ops = ['mean','var','skew','nunique','cumcount']
    #ops = ['mean','var','nunique','cumcount']

    raw_cols_groups = [raw_cols0]
    #raw_cols_groups = [raw_cols0, raw_cols1]

    for raw_cols in raw_cols_groups:
        # non-count() coms:
        for op in ops:
            for cols_count in range(2, 7):
                for cols_coms in itertools.combinations(raw_cols, cols_count):
                    com_fts_list_to_use.append({'group':list(cols_coms), 'op':op})

        #print('added non-count coms(len: {}): {}'.format(len(com_fts_list_to_use), com_fts_list_to_use))

        print('\n\n\n')
        #for count():
        for cols_count in range(1, 7):
            for cols_coms in itertools.combinations(raw_cols, cols_count):
                temp = []
                temp.extend(cols_coms)
                temp.append('is_attributed')
                com_fts_list_to_use.append({'group': list(temp), 'op': 'count'})

    #print('added count coms(len: {}): {}'.format(len(com_fts_list_to_use), com_fts_list_to_use))

    shuffle(com_fts_list_to_use)
    #print('shuffled coms(len: {}): {}'.format(len(com_fts_list_to_use), com_fts_list_to_use))
    #exit(0)


    size = 400 if not only_gen_ft_cache else 100000

    if use_sample:
        com_fts_list_to_use = com_fts_list_to_use[0:size*3+2]


    i = 0
    val_auc_list = []
    importances_list = []
    for pos in range(0, len(com_fts_list_to_use), size):
        print('==================================')
        print('#{}. training with statistical features combinations:\n{}'.format(i, '\n'.\
              join([str(a) for a in com_fts_list_to_use[pos:pos + size]])))
        print('==================================')
        if only_gen_ft_cache:
            with timer('------gening fts------' + str(i)):
                gen_ft_caches_seperately(com_fts_list_to_use[pos:pos + size])
        else:
            with timer('------training------' + str(i)):
                additional_groups= [
                    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'],'op': 'nextclick'}
                    #, {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
                    ]

                if use_lgbm_searcher:
                    lgbm_params_search(additional_groups + com_fts_list_to_use[pos:pos + size])
                    importances = {}
                    auc = 0
                else:
                    jointly = True
                    if jointly:
                        importances, auc = train_and_predict(additional_groups +
                                                            com_fts_list_to_use[pos:pos + size],
                                                            use_ft_cache=False,
                                                            only_cache=only_gen_ft_cache,
                                                            use_base_data_cache=False)
                    else:
                        importances, auc = train_and_predict_gen_fts_seperately(additional_groups +
                                                                        com_fts_list_to_use[pos:pos + size],
                                                                        use_ft_cache=False,
                                                                        only_cache=only_gen_ft_cache,
                                                                        use_base_data_cache=False)

                importances_list.append(importances)
                val_auc_list.append(auc)
                gc.collect()
                i+=1
        print('\n\n\n')

    if not only_gen_ft_cache:
        i = 0
        for importances, auc in zip(importances_list, val_auc_list):
            print('#',i)
            i+= 1
            print('features importances:')
            pprint(importances)
            print('val auc:', auc)



def lgbm_params_search(com_fts_list):
    ITERATIONS = 1000
    # Classifier
    search_spaces_0 = {
                'learning_rate': (10.0 ** 0.01, 10.0 ** 1.0, 'log-uniform'),
                'num_leaves': (4, 31),
                'max_depth': (0, 50),
                'min_child_samples': (10,200),
                #'max_bin': (64, 255), # unsupported in skylearn
                'subsample': (0.01, 1.0, 'uniform'),
                'subsample_freq': (0, 1),
                'colsample_bytree': (0.01, 1.0, 'uniform'),
                'min_child_weight': (0, 10),
                'subsample_for_bin': (200000,800000),
                'min_split_gain': (10.0 ** 0, 10.0 ** 0.01, 'log-uniform'),
                'reg_alpha':  (10.0 ** 0, 10.0 ** 1.0, 'log-uniform'),
                'reg_lambda': (10.0 ** 0, 1000.0, 'log-uniform'),
                # 'is_unbalance': True,
                'n_estimators': (50, 100), # alias: num_boost_round
                'scale_pos_weight': (10.0 ** 1e-6, 500.0, 'log-uniform')
            }
    search_spaces_1 = {
                        'learning_rate': (0.01, 1.0, 'log-uniform'),
                        'min_child_weight': (0, 10),
                        'max_depth': (3, 10),
                        'num_leaves': (4, 11),
                        'subsample': (0.01, 1.0, 'uniform'),
                        'colsample_bytree': (0.01, 1.0, 'uniform'),
                        'reg_lambda': (1e-9, 1000, 'log-uniform'),
                        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
                        'min_child_weight': (0, 5),
                        'min_child_samples': (10, 200),
                        # 'max_bin': (64, 255), # unsupported in skylearn
                        'subsample_freq': (0, 1),
                        'subsample_for_bin': (200000, 800000),
                        'min_split_gain': (0, 0.01, 'uniform'),
                        # 'is_unbalance': True,
                        'n_estimators': (50, 500),  # alias: num_boost_round
                        'scale_pos_weight': (1e-6, 500.0, 'log-uniform')
                    }
    with timer('create bayes cv tunner'):
        bayes_cv_tuner = BayesSearchCV(
            estimator=lgb.LGBMClassifier(
                boosting_type= 'gbdt',
                objective= 'binary',
                metric= 'auc',
                n_jobs = 5,
                silent = False
            ),
            search_spaces=search_spaces_1,
            scoring='roc_auc',
            cv=StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            ),
            n_jobs=3,
            n_iter=ITERATIONS,
            verbose=9,
            refit=True,
            random_state=42
        )

    with timer('load combined data df'):
        combined_df, train_len, val_len = get_combined_df(config_scheme_to_use.new_predict)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]

    predictors1 = categorical + new_features

    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st

    train = train.append(val)
    train[predictors1].values

    def status_print(optim_result):
        """Status callback durring bayesian hyperparameter search"""

        # Get all the models tested so far in DataFrame format
        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)

        # Get current parameters and the best parameters
        best_params = pd.Series(bayes_cv_tuner.best_params_)
        print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
            len(all_models),
            np.round(bayes_cv_tuner.best_score_, 4),
            bayes_cv_tuner.best_params_
        ))
        all_models.sort_values('mean_test_score')

        # Save all model results
        all_models.to_csv("xgBoost_cv_results.csv")

    # Fit the model
    result = bayes_cv_tuner.fit(train[predictors1].values, train['is_attributed'].values, callback=status_print)


def run_model():
    print('run theme: ', config_scheme_to_use.run_theme)

    if config_scheme_to_use.run_theme == 'grid_search_ft_gen':
        grid_search_features_combination(True)
    elif config_scheme_to_use.run_theme == 'grid_search_ft_coms':
        grid_search_features_combination(False)
    elif config_scheme_to_use.run_theme == 'grid_search_ft_coms_plus_lgbm_searcher':
        grid_search_features_combination(False, use_lgbm_searcher=True)
    elif config_scheme_to_use.run_theme == 'train_and_predict':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list)
    elif config_scheme_to_use.run_theme == 'lgbm_params_search':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        lgbm_params_search(config_scheme_to_use.add_features_list)
    elif config_scheme_to_use.run_theme == 'train_and_predict_gen_fts_seperately':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict_gen_fts_seperately(config_scheme_to_use.add_features_list)
    else:
        print("nothing to run... exit")


with timer('run_model...'):
    run_model()

print('run_model done')