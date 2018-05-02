# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
import copy
from scipy import special as sp

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
import mmh3

matplotlib.use('Agg')


def get_dated_filename(filename):
    #print('got file name: {}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S")))
    #return '{}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))


    print('got file name: {}_{}.csv'.format(filename, config_scheme_to_use.config_name))
    return '{}_{}.csv'.format(filename, config_scheme_to_use.config_name)
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
import csv

from pympler import muppy
from pympler import summary

dump_train_data = False

use_sample = False
debug = False
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
path_test_supplement = path + 'test_supplement.csv'
path_test_supplement_sample = path + 'test_supplement_sample.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

#categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
categorical = ['app', 'device', 'os', 'channel', 'hour']
# with ip:
# categorical = ['app', 'device', 'os', 'channel', 'hour', 'ip']

pick_hours={4, 5, 10, 13, 14}
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

most_freq_values_in_test_data  = {
    'device': [],
    'app':[],
    'os':[],
    'channel':[
        107,265,232,477,178,153,134,259,128,442,127,379,205,466,121,137,439,489,145,480,135,469,219,122,215,
        435,244,237,347,409,140,334,452,328,377,211,236,424,173,130,459,116,19,115,315,401,125,212,340,
        349,258,266,234,213,105,445,376,386,481,319,3,343,463,478,412,278,430,101,111,400,364,497,124,
        402,487,325,243,417,326,18,113,448,242,150,17,317,488,21
    ], # 88, 99.1%
    'ip':[]
}
least_freq_values_in_test_data  = {
    'device': [],
    'app':[],
    'os':[],
    'channel':[
        172, 149, 138, 322, 490, 169, 251, 458, 483, 281, 222, 181, 223, 256, 352, 407, 233, 216, 123, 4, 420, 114, 272,
        408, 410, 341, 455, 0, 261, 356, 262, 126, 457, 174, 208, 311, 416, 498, 414, 277, 353, 451, 332, 391, 479, 15,
        110, 22, 465, 253, 456, 460, 108, 203, 160, 320, 484, 450, 274, 5, 24, 419, 120, 268, 446, 330, 360, 361, 421,
        224, 30, 453, 118, 13, 129, 182, 171, 245, 406, 282, 411, 449, 225, 333, 210, 371, 404, 280, 373, 467
    ], # 90, < 1%
    'ip':[]
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
9308569
id_8_4am = 82259195
id_8_3pm = 118735619
id_9_4am = 144708152
id_9_3pm = 181878211
id_7_4am = 22536989
id_7_3pm = 56845833
id_9_4pm = 184903891 -1

sample_from_list = [0, 50000]
sample_to_list = [49998, 90000]

public_train_from = 109903890
public_train_to =  147403890
public_val_from = 147403890
public_val_to = 149903890

debug_train_from = 0
debug_train_to = 90000
debug_val_from=90000
debug_val_to=100000

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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}

lgbm_params_search_128_114 = dict(new_lgbm_params)
lgbm_params_search_128_114.update({
    'colsample_bytree': 0.8793460386326015, 'learning_rate': 0.19814501809928017, 'max_depth': 9,
    'min_child_samples': 188, 'min_child_weight': 4, 'num_leaves': 11, 'reg_alpha': 0.02387225386312356,
    'reg_lambda': 1.2196200544739068e-09, 'scale_pos_weight': 231.48637373544372,
    'subsample': 0.7079619705989065}
)
lgbm_params_search_128_610 = dict(new_lgbm_params)
lgbm_params_search_128_114.update({
    'colsample_bytree': 0.7773614836495996, 'learning_rate': 0.2, 'max_depth': 10, 'min_child_samples': 10,
    'min_child_weight': 0, 'num_leaves': 11, 'reg_alpha': 1.0, 'reg_lambda': 1e-09,
    'scale_pos_weight': 249.99999999999994, 'subsample': 0.6870745956370757}
)
lgbm_params_l1 = dict(new_lgbm_params)
lgbm_params_l1.update({
    'reg_alpha': 1.0
})

lgbm_params_pub_asraful_kernel = dict(new_lgbm_params)
lgbm_params_pub_asraful_kernel.update({
        'learning_rate': 0.10,
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

new_lgbm_params_feature_fraction = {**new_lgbm_params, ** {
    'feature_fraction': 0.5
}}

new_lgbm_params_iter_600 = {
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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 600,
    'num_boost_round': 600,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}
new_lgbm_params_100_cat_smooth = {
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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0,
    'cat_smooth':100
}
new_lgbm_params_early_300 = {
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
    'nthread': 10,
    'verbose': 9,
    'num_boost_round':300,
    'early_stopping_round': 300,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}

new_lgbm_params_early_415 = dict(new_lgbm_params)
new_lgbm_params_early_415['num_boost_round'] = 415
new_lgbm_params_early_415['early_stopping_round'] = 415


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
        'scale_pos_weight':200, # because training data is extremely unbalanced
        'early_stopping_round': 30
    })

new_lagbm_params_from_115_0 = dict(new_lgbm_params)
new_lagbm_params_from_115_0.update( \
    {'colsample_bytree': 1.0, 'learning_rate': 0.1773256374384233, 'max_depth': 3, 'min_child_samples': 200, 'min_child_weight': 0,
    'min_split_gain': 0.0007911719321269061, 'num_leaves': 11, 'reg_alpha': 2.355979159306278e-08,
    'reg_lambda': 0.9016760858543618, 'scale_pos_weight': 260.6441151527916, 'subsample': 1.0, 'subsample_for_bin': 457694, 'subsample_freq': 0} )

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



shuffle_sample_filter = {'filter_type': 'sample', 'sample_count': 6}
shuffle_sample_filter_1_to_6 = {'filter_type': 'sample', 'sample_count': 6}

shuffle_sample_filter_1_to_2 = {'filter_type': 'sample', 'sample_count': 2}
shuffle_sample_filter_1_to_3 = {'filter_type': 'sample', 'sample_count': 3}

shuffle_sample_filter_1_to_10 = {'filter_type': 'sample', 'sample_count': 10}
shuffle_sample_filter_1_to_20 = {'filter_type': 'sample', 'sample_count': 20}
shuffle_sample_filter_1_to_10k = {'filter_type': 'sample', 'sample_count': 1}

hist_ft_sample_filter = {'filter_type': 'hist_ft'}

random_sample_filter_0_5 = {'filter_type': 'random_sample', 'frac': 0.5}


skip = range(1, 140000000)
print("Loading Data")
# skiprows=skip,

import pickle

add_features_list_origin_no_channel_next_click = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]

add_features_list_smooth_cvr = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'os', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'device', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['hour', 'is_attributed'], 'op': 'smoothcvr'},
    #{'group': ['ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['device', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['channel', 'is_attributed'], 'op': 'smoothcvr'},

    # for debuging the smooth cvrs:
    #{'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'smoothcvr'}, #cheating ft, low val auc, avoid it

    #{'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'mean'},
    #{'group': ['hour', 'is_attributed'], 'op': 'mean'},
    #{'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'count'},
    #{'group': ['hour', 'is_attributed'], 'op': 'count'}

    ]


add_features_list_origin_no_channel_next_click_best_ct_nu_from_search = [
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['os', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'device', 'os', 'hour', 'ip'], 'op': 'nunique'},

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]
add_features_list_origin_no_channel_next_click_best_ct_nu_from_search_28 = [
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'},

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]

add_features_list_origin_no_channel_next_click_ip_freq_ch = \
    add_features_list_origin_no_channel_next_click + [
        {'group': ['ip', 'in_test_frequent_channel', 'is_attributed'], 'op': 'count'}
    ]

add_features_list_pub_asraful_kernel  = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},

    {'group': ['ip', 'os'], 'op': 'cumcount'},
    {'group': ['ip','device','os', 'app'], 'op': 'cumcount'},

    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'app', 'os', 'hour'], 'op': 'var'}
    ]

add_features_add_best_nunique = add_features_list_origin_no_channel_next_click + [
    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'}
]


add_features_from_pub_ftrl = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'channel'], 'op': 'nunique'}
    ]


add_features_list_origin_no_channel_next_click_10mincvr = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['id_10min', 'is_attributed'], 'op': 'mean'}
]



add_features_list_origin_no_channel_next_click_days = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'day', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day','in_test_hh', 'is_attributed'], 'op': 'count'}
    ]


add_features_list_origin_no_channel_next_click_no_app = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]


add_features_list_origin_no_channel_next_click_stnc = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    # st nc:
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.98'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.02'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'min'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'var'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'mean'}
    #,{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'skew'}
    ]

add_features_list_origin_no_channel_next_click_varnc = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    # st nc:
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.98'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.02'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'var'},
    #,{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'skew'}
    ]

add_features_list_origin_no_channel_next_click_next_n_click = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextnclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    # st nc:
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.98'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.02'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'var'},
    #,{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'skew'}
    ]

add_features_list_origin = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
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
                 add_features_list = add_features_list_origin,
                 use_ft_cache = False,
                 use_ft_cache_from = None,
                 qcut = 0,
                 add_second_ft = False,
                 use_lgbm_fts = False,
                 sync_mode = False,
                 normalization = False,
                 add_10min_ft = False,
                 pick_hours_weighted = False,
                 adversial_val_weighted = False,
                 adversial_val_ft=False,
                 add_in_test_frequent_dimensions = None,
                 add_lgbm_fts_from_saved_model = False,
                 train_smoothcvr_cache_from = None,
                 train_smoothcvr_cache_to = None,
                 test_smoothcvr_cache_from = None,
                 test_smoothcvr_cache_to = None
                 ):
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
        self.use_ft_cache = use_ft_cache
        self.use_ft_cache_from = use_ft_cache_from
        self.qcut = qcut
        self.add_second_ft = add_second_ft
        self.use_lgbm_fts = use_lgbm_fts
        self.sync_mode = sync_mode
        self.normalization = normalization
        self.add_10min_ft = add_10min_ft
        self.pick_hours_weighted = pick_hours_weighted
        self.adversial_val_weighted = adversial_val_weighted
        self.adversial_val_ft = adversial_val_ft
        self.add_in_test_frequent_dimensions = add_in_test_frequent_dimensions
        self.add_lgbm_fts_from_saved_model = add_lgbm_fts_from_saved_model
        self.train_smoothcvr_cache_from = train_smoothcvr_cache_from
        self.train_smoothcvr_cache_to = train_smoothcvr_cache_to
        self.test_smoothcvr_cache_from = test_smoothcvr_cache_from
        self.test_smoothcvr_cache_to = test_smoothcvr_cache_to



train_config_103_11 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_103_12 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click_stnc
                                   )

train_config_103_13 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click_varnc,
                                   use_ft_cache=True
                                   )

train_config_103_14 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click_next_n_click,
                                   use_ft_cache=False
                                   )

train_config_103_15 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                   qcut = 0.98
                                   )

train_config_103_16 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click_no_app,
                                   use_ft_cache=False
                                   )

train_config_103_20 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                   add_second_ft=True
                                   )

train_config_103_20 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                   add_second_ft=True
                                   )


train_config_103_21 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lagbm_params_from_115_0,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_103_22 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_with_test_supplement',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_103_23 = ConfigScheme(False, False, False,
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
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_103_26 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_9_3pm,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict_with_test_supplement',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_103_27 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_early_300,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_9_3pm,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict_with_test_supplement',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_119 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter_1_to_6,
                                 None,
                                 lgbm_params={**new_lgbm_params, **{'num_boost_round':20}},
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_lgbm_fts',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )


train_config_120 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter_1_to_6,
                                 None,
                                 lgbm_params={**new_lgbm_params, **{'num_boost_round':20}},
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                use_lgbm_fts=True
                                   )


train_config_120_2 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter_1_to_6,
                                 None,
                                 lgbm_params={**new_lgbm_params, **{'num_boost_round':30}},
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                use_lgbm_fts=True,
                                  sync_mode=True
                                   )


train_config_116 = ConfigScheme(False, False, False,
                                None,
                                shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )

train_config_116_3 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )

train_config_116_4 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                  discretization=50,
                                  )


train_config_116_5 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                  wordbatch_model='FTRL',
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )


train_config_116_6 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                  wordbatch_model='NN_ReLU_H1',
                                  new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )

train_config_117_1 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_117_3 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_117_4 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_from_pub_ftrl,
                                   use_ft_cache=False
                                   )

train_config_117_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_8_3pm,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_117_6 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                log_discretization=True,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_117_8 = copy.deepcopy(train_config_117_3)
train_config_117_8.log_discretization = True

train_config_117_9 = copy.deepcopy(train_config_117_8)
train_config_117_9.add_lgbm_fts_from_saved_model = True
train_config_117_9.add_features_list = add_features_list_origin_no_channel_next_click_days

train_config_121_1 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_3,
                                 shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_121_2 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_10,
                                 shuffle_sample_filter_1_to_10,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_121_5 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_3,
                                 shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_121_6 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_3,
                                 shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_122 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model_ffm',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )


train_config_124 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )



train_config_124_1 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=[id_8_4am,id_9_4am],
                                 train_to=[id_8_3pm, id_9_3pm],
                                 val_from=id_7_4am,
                                 val_to=id_7_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )

train_config_124_2 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=[id_7_4am, id_8_4am,id_9_4am],
                                 train_to=[id_7_3pm, id_8_3pm, id_9_3pm],
                                 val_from=1,
                                 val_to=id_7_4am - 1,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )
train_config_124_3 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )

train_config_124_4 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                  normalization=True
                                   )


train_config_124_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_early_300,
                                 new_predict= True,
                                 train_from=id_7_3pm,
                                 train_to=id_9_3pm,
                                 val_from=id_9_3pm,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )

train_config_124_6 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_early_300,
                                 new_predict= True,
                                 train_from=id_8_4am,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )


train_config_124_9 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_from_pub_ftrl
                                   )



train_config_125 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_10mincvr,
                                add_10min_ft=True
                                   )
train_config_125_4 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_10mincvr,
                                add_10min_ft=True
                                   )


train_config_124_7 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4am,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )

train_config_124_8 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days,
                                 pick_hours_weighted = True
                                   )

train_config_124_10 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_iter_600,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days,
                                 pick_hours_weighted = True
                                   )

train_config_124_11 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_add_best_nunique
                                   )

train_config_124_12 = train_config_124_10

train_config_124_14 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_feature_fraction,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_add_best_nunique
                                   )

train_config_124_16 = ConfigScheme(False, False, False,
                                 None,
                                 None,
                                 None,
                                 lgbm_params=new_lgbm_params_early_415,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days,
                                 pick_hours_weighted = True
                                   )

train_config_124_17 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_7_3pm,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )
train_config_124_18 = copy.deepcopy(train_config_124_3)
train_config_124_18.adversial_val_weighted = True

train_config_124_19 = copy.deepcopy(train_config_124)
train_config_124_19.lgbm_params = lgbm_params_search_128_114

train_config_124_20 = copy.deepcopy(train_config_124_17)
train_config_124_20.train_from = 0
train_config_124_20.train_to = id_7_3pm

train_config_124_21 = copy.deepcopy(train_config_124_3)
train_config_124_21.adversial_val_ft = True


train_config_124_22 = copy.deepcopy(train_config_124)
train_config_124_22.adversial_val_weighted = True


train_config_124_23 = copy.deepcopy(train_config_124)
train_config_124_23.add_in_test_frequent_dimensions = ['channel']
train_config_124_23.add_features_list = add_features_list_origin_no_channel_next_click_ip_freq_ch



train_config_124_25 = copy.deepcopy(train_config_124_23)
train_config_124_25.adversial_val_weighted = True

train_config_124_26 = copy.deepcopy(train_config_124)
train_config_124_26.add_features_list = add_features_list_origin_no_channel_next_click_best_ct_nu_from_search


train_config_124_28 = copy.deepcopy(train_config_124)
train_config_124_28.add_features_list = add_features_list_origin_no_channel_next_click_best_ct_nu_from_search_28


train_config_124_29 = copy.deepcopy(train_config_124)
train_config_124_29.add_features_list = add_features_list_smooth_cvr

train_config_124_30 = copy.deepcopy(train_config_124)
train_config_124_30.add_features_list = add_features_list_smooth_cvr
train_config_124_30.train_smoothcvr_cache_from = id_8_4am
train_config_124_30.train_smoothcvr_cache_to = id_8_3pm
train_config_124_30.test_smoothcvr_cache_from = id_9_4am
train_config_124_30.test_smoothcvr_cache_to = id_9_3pm

train_config_124_31 = copy.deepcopy(train_config_124)
train_config_124_31.lgbm_params = lgbm_params_search_128_610

train_config_126_1 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_2 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=[id_7_4am, id_8_4am,id_9_4am],
                                 train_to=[id_7_3pm, id_8_3pm, id_9_3pm],
                                 val_from=1,
                                 val_to=id_7_4am - 1,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_3 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_4 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_from_pub_ftrl
                                   )

train_config_126_5 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_7_3pm,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_6 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_7_3pm,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 pick_hours_weighted = True
                                  )

train_config_126_11 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params_feature_fraction,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )
train_config_126_12 = copy.deepcopy(train_config_126_6)
train_config_126_12.val_from = id_7_4am
train_config_126_12.val_to = id_7_3pm


train_config_126_13 = copy.deepcopy(train_config_126_1)
train_config_126_13.add_features_list = add_features_list_pub_asraful_kernel


train_config_126_14 = copy.deepcopy(train_config_126_1)
train_config_126_14.add_features_list = add_features_list_pub_asraful_kernel
train_config_126_14.lgbm_params =  lgbm_params_pub_asraful_kernel

train_config_126_15 = copy.deepcopy(train_config_126_1)
train_config_126_15.add_features_list = add_features_list_smooth_cvr
#train_config_126_15.train_smoothcvr_cache_from = id_7_4am
#train_config_126_15.train_smoothcvr_cache_to = id_7_3pm


train_config_121_7 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_121_8 = copy.deepcopy(train_config_121_7)
train_config_121_9 = copy.deepcopy(train_config_121_7)
train_config_121_10 = copy.deepcopy(train_config_121_7)
train_config_121_10.lgbm_params=lgbm_params_l1

train_config_121_11 = copy.deepcopy(train_config_121_7)
train_config_121_11.lgbm_params=lgbm_params_l1
train_config_121_11.adversial_val_weighted = True

train_config_121_12 = copy.deepcopy(train_config_121_8)
train_config_121_12.lgbm_params=lgbm_params_l1
train_config_121_12.adversial_val_weighted = True

train_config_121_13 = copy.deepcopy(train_config_121_8)

train_config_121_13.train_smoothcvr_cache_from = id_7_4am
train_config_121_13.train_smoothcvr_cache_to = id_7_3pm

train_config_126_9 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params_100_cat_smooth,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )


train_config_128 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='lgbm_params_search',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

def use_config_scheme(str):
    ret = eval(str)
    if debug:
        ret.train_from = debug_train_from
        ret.train_to = debug_train_to
        ret.val_from=debug_val_from
        ret.val_to=debug_val_to
    print('using config var name and test log: ', str)
    ret.config_name = str
    if ret.use_ft_cache_from is None:
        ret.use_ft_cache_from = ret.config_name
    print('config values: ')
    pprint(vars(ret))


    return ret


config_scheme_to_use = use_config_scheme('train_config_121_13')


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
    if config_scheme_to_use.add_second_ft:
        data['second'] = data["click_time"].dt.second.astype('int8')
        categorical.append('second')

    add_hh_feature = True
    if add_hh_feature:
        data['in_test_hh'] = (3
                              - 2 * data['hour'].isin(most_freq_hours_in_test_data)
                              - 1 * data['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')
        # categorical.append('in_test_hh')

    if config_scheme_to_use.add_in_test_frequent_dimensions is not None:
        for dimension in config_scheme_to_use.add_in_test_frequent_dimensions:
            data['in_test_frequent_' + dimension] =(3
                              - 2 * data[dimension].isin(most_freq_values_in_test_data[dimension])
                              - 1 * data[dimension].isin(least_freq_values_in_test_data[dimension])).astype('uint8')
    #126 8
    #data['hour'] = data['hour'] // 3

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
                print('added iterative fts:',interactive_features_name )
    return data


def post_statistics_features(data):
    print('ip_in_test_frequent_channel_is_attributed_count describe:')
    if config_scheme_to_use.add_in_test_frequent_dimensions is not None and \
        'channel' in config_scheme_to_use.add_in_test_frequent_dimensions:
        print(data['ip_in_test_frequent_channel_is_attributedcount'].describe())
    return data



def df_get_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    return counts[unqtags]


def get_recursive_alpha_beta(sumi, count):

    def getalpha(sumi, count, alpha0, beta0):
        return alpha0 * (sp.psi(sumi + alpha0) - sp.psi(alpha0)) / \
            (sp.psi(count + alpha0 + beta0) - sp.psi(alpha0 + beta0))

    def getbeta(sumi, count, alpha0, beta0):
        return beta0 * (sp.psi(count - sumi + beta0) - sp.psi(beta0)) / \
            (sp.psi(count + alpha0 + beta0) - sp.psi(alpha0 + beta0))

    alpha = 10.0
    beta = 10000.0
    for i in range(1000):
        alpha0 = alpha
        beta0 = beta
        alpha = getalpha(sumi, count, alpha0, beta0)
        beta = getbeta(sumi, count, alpha0, beta0)
    return alpha, beta

def add_statistic_feature(group_by_cols, training, qcut_count=config_scheme_to_use.qcut, #0, #0.98,
                          discretization=0, discretization_bins=None,
                          log_discretization=False,
                          op='count',
                          use_ft_cache = False,
                          ft_cache_prefix = '',
                          only_ft_cache = False,
                          astype=None):
    input_len = len(training)
    feature_name_added = '_'.join(group_by_cols) + op

    ft_cache_file_name = config_scheme_to_use.use_ft_cache_from + "_" + ft_cache_prefix + '_' + feature_name_added
    ft_cache_file_name = ft_cache_file_name + '_sample' if use_sample else ft_cache_file_name
    ft_cache_file_name = ft_cache_file_name + '.csv.bz2'

    loaded_from_cache = False

    if use_ft_cache and Path(ft_cache_path + ft_cache_file_name).is_file():
        ft_cache_data = pd.read_csv(ft_cache_path + ft_cache_file_name,
                                    dtype='float32',
                                    header=0, engine='c',
                                    compression='bz2')
        training[feature_name_added] = ft_cache_data
        print('loaded {} from file {}'.format(feature_name_added, ft_cache_path + ft_cache_file_name))
        loaded_from_cache=True
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
    elif op=='nextnclick':
        with timer("Adding next n click times"):
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
            if debug:
                print('data:',training[0:10])
                print('debug str',joint_col[0:10])
                print('debug str', (training['ip'].astype(str) + "_" + training['app'].astype(str) + "_" + training['device'].astype(str) \
            + "_" + training['os'].astype(str)+ "_" + training['channel'].astype(str))[0:10])

            training['category'] = joint_col.apply(mmh3.hash) % D
            if debug:
                print('debug category',training['category'][0:10])

            del joint_col
            gc.collect()

            n = 3
            click_buffers = []
            for i in range(n):
                click_buffers.append(np.full(D, 3000000000, dtype=np.uint32))
            training['epochtime'] = training['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, time in zip(reversed(training['category'].values),
                                      reversed(training['epochtime'].values)):
                # shift values in buffers queue and append new value from the tail
                for i in range(n - 1):
                    click_buffers[i][category] = click_buffers[i+1][category]
                next_clicks.append(click_buffers[0][category] - time)
                click_buffers[n-1][category] = time
            del (click_buffers)
            training[feature_name_added] = list(reversed(next_clicks))

            #training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            #features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            print('next click added:', training[feature_name_added].describe())
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
            if debug:
                print('data:',training[0:10])
                print('debug str',joint_col[0:10])
                print('debug str', (training['ip'].astype(str) + "_" + training['app'].astype(str) + "_" + training['device'].astype(str) \
            + "_" + training['os'].astype(str)+ "_" + training['channel'].astype(str))[0:10])

            training['category'] = joint_col.apply(mmh3.hash) % D
            if debug:
                print('debug category',training['category'][0:10])

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

            #training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            #features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            print('next click added:', training[feature_name_added].describe())
    elif op=='smoothcvr':
        with timer('gen cvr grouping cache:'):
            if not hasattr(add_statistic_feature, 'train_cvr_cache'):
                add_statistic_feature.train_cvr_cache = dict()

            if not hasattr(add_statistic_feature, 'alpha'):
                add_statistic_feature.alpha, add_statistic_feature.beta = \
                    get_recursive_alpha_beta(training['is_attributed'].sum(), training['is_attributed'].count())
                print('total alpha/beta: {}/{}'.format(add_statistic_feature.alpha, add_statistic_feature.beta))
                print('total cvr:{}, alpha/(alpha+beta):{}'.format(training['is_attributed'].mean(),
                    add_statistic_feature.alpha/ (add_statistic_feature.alpha + add_statistic_feature.beta)))

            if feature_name_added in add_statistic_feature.train_cvr_cache:
                temp_sum = add_statistic_feature.train_cvr_cache[feature_name_added]
            else:
                if ft_cache_prefix != 'train':
                    print("!!!!!!non-train should only use cache, which should be there!!!!!!!")
                    exit(-1)
                temp_count = training[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[['is_attributed']].count()
                temp_sum = training[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[['is_attributed']].sum()
                temp_sum[feature_name_added] = (temp_sum['is_attributed'] + add_statistic_feature.alpha) /  \
                        (temp_count['is_attributed'] + add_statistic_feature.alpha + add_statistic_feature.beta).astype('float16')
                del temp_count
                del temp_sum['is_attributed']
                gc.collect()
                add_statistic_feature.train_cvr_cache[feature_name_added] = temp_sum
        #n_chans = temp1.reset_index().rename(columns={'is_attributed': feature_name_added})
        training = training.merge(temp_sum.reset_index(), on=group_by_cols if len(group_by_cols) >1 else group_by_cols[0],
                                  how='left')
        if not hasattr(add_statistic_feature, 'global_cvr'):
            add_statistic_feature.global_cvr = training['is_attributed'].mean()
        training[feature_name_added] = training[feature_name_added].fillna(add_statistic_feature.global_cvr)
        print('smooth cvr: {} gened'.format(feature_name_added))
        print(training[feature_name_added].describe())
    else:
        tempstr = 'training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]]'
        if len(op) > 2 and op[0:2] == 'qt':
            temp1 = eval('{}.quantile({})'.format(tempstr, float(op[2:])) )
            #temp1 = eval('{}.apply(lambda x: np.percentile(x.sample(min(len(x), 100)), q={}))'.format(tempstr, float(op[2:])) )
        else:
            temp1 = eval(tempstr + '.' + op + '()')

        n_chans = temp1.reset_index().rename(columns={counting_col: feature_name_added})
        training = training.merge(n_chans, on=group_by_cols if len(group_by_cols) >1 else group_by_cols[0],
                                  how='left')
        del n_chans

        if config_scheme_to_use.normalization and op == 'count':
            if hasattr(add_statistic_feature, first_df_count):
                rate = add_statistic_feature.first_df_count / input_len
            else:
                add_statistic_feature.first_df_count = input_len
                rate = 1.0
            training[feature_name_added] = (training[feature_name_added] * rate).astype('uint32')


    gc.collect()
    no_type_cast = False
    if not no_type_cast and not log_discretization and discretization == 0:
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
    for ft in features_added:
        if astype is not None:
            training[ft] = training[ft].astype(astype)

    print('added features:', features_added)
    if print_verbose:
        print(training[feature_name_added].describe())
    print('nan count: ', training[feature_name_added].isnull().sum())

    print('columns after added: ', training.columns.values)

    if use_ft_cache and not loaded_from_cache:
        pd.DataFrame(training[feature_name_added]).to_csv(ft_cache_path + ft_cache_file_name,
                                                          index=False,compression='bz2')
        print('saved {} to file {}'.format(feature_name_added, ft_cache_path + ft_cache_file_name))
        if only_ft_cache:
            del training[feature_name_added]
            del features_added[-1]
            gc.collect()

    return training, features_added, discretization_bins_used

def clear_smoothcvr_cache():
    if hasattr(add_statistic_feature, 'train_cvr_cache'):
        del add_statistic_feature.train_cvr_cache
        del add_statistic_feature.global_cvr
        gc.collect()

def gen_smoothcvr_cache(frm, to):
    with timer('loading train df:'):
        train = pd.read_csv(path_train_sample if use_sample else path_train,
                            dtype=dtypes,
                            header=0,
                            usecols=train_cols,
                            skiprows=range(1, frm) \
                                if not use_sample and frm is not None else None,
                            nrows=to - frm \
                                if not use_sample and frm is not None else None,
                            parse_dates=["click_time"])

    print('mem after loaded train data:', cpuStats())

    with timer('gen categorical features for train'):
        train = gen_categorical_features(train)


    for add_feature in config_scheme_to_use.add_features_list:
        feature_name_added = '_'.join(add_feature['group']) + add_feature['op']

        counting_col = add_feature['group'][len(add_feature['group']) - 1]
        group_by_cols = add_feature['group'][0:len(add_feature['group']) - 1]

        if add_feature['op'] != 'smoothcvr':
            continue
        print('processing:',add_feature)
        with timer('gen cvr grouping cache:'):
            if not hasattr(add_statistic_feature, 'train_cvr_cache'):
                add_statistic_feature.train_cvr_cache = dict()

            if not hasattr(add_statistic_feature, 'alpha'):
                add_statistic_feature.alpha, add_statistic_feature.beta = \
                    get_recursive_alpha_beta(train['is_attributed'].sum(), train['is_attributed'].count())
                print('total alpha/beta: {}/{}'.format(add_statistic_feature.alpha, add_statistic_feature.beta))
                print('total cvr:{}, alpha/(alpha+beta):{}'.format(train['is_attributed'].mean(),
                                                                   add_statistic_feature.alpha / (
                                                                   add_statistic_feature.alpha + add_statistic_feature.beta)))

            if feature_name_added in add_statistic_feature.train_cvr_cache:
                temp_sum = add_statistic_feature.train_cvr_cache[feature_name_added]
            else:
                temp_count = train[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[
                    ['is_attributed']].count()
                temp_sum = train[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[['is_attributed']].sum()
                temp_sum[feature_name_added] = (temp_sum['is_attributed'] + add_statistic_feature.alpha) / \
                                               (temp_count[
                                                    'is_attributed'] + add_statistic_feature.alpha + add_statistic_feature.beta).astype(
                                                   'float16')
                del temp_count
                del temp_sum['is_attributed']
                gc.collect()
                add_statistic_feature.train_cvr_cache[feature_name_added] = temp_sum

    print('setting global cvr for fillna...')
    if not hasattr(add_statistic_feature, 'global_cvr'):
        add_statistic_feature.global_cvr = train['is_attributed'].mean()

    del train
    gc.collect()


def generate_counting_history_features(data,
                                       discretization=0, discretization_bins=None,
                                       add_features_list=None,
                                       use_ft_cache = False,
                                       ft_cache_prefix = '',
                                       only_ft_cache = False):
    print('discretization bins to use:', discretization_bins)

    new_features = []

    if config_scheme_to_use.add_10min_ft:
        with timer('adding 10min feature:'):
            data['id_10min'] = data.click_time.astype(int) // (10 ** 9 * 600 *5)
            categorical.append('id_10min')

    if config_scheme_to_use.adversial_val_ft:
        with timer('add ad_val ft:'):
            ad_val_bst = lgb.Booster(model_file='ad_val_model.txt')
            ad_val_predictors = ['app', 'device', 'os', 'channel', 'ip', 'hour']
            data['ad_val'] = ad_val_bst.predict(data[ad_val_predictors])
            print('add ad_val fts:', data['ad_val'].describe())
            new_features.append('ad_val')

            del ad_val_bst
            gc.collect()

    discretization_bins_used = None
    i = -1
    for add_feature in add_features_list:
        i += 1
        #with timer('adding feature:' + str(add_feature)):
        with timer('adding feature: {}/{}, {}'.format(i, len(add_features_list), str(add_feature))):
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

    print('\n\n\n-------------\n{} DEBUG CVR:\n-------------\n\n\n'.format(ft_cache_prefix))
    if 'hour_is_attributedsmoothcvr' in data.columns and \
            'hour_is_attributedmean' in data.columns and \
            'hour_is_attributedcount' in data.columns:
        print('gened hour_is_attributedsmoothcvr:')
        print(data[['hour_is_attributedsmoothcvr', 'hour_is_attributedmean',
                    'hour_is_attributedcount','hour']].sample(20).to_string())


    if 'ip_app_device_os_is_attributedsmoothcvr' in data.columns and \
        'ip_app_device_os_is_attributedmean' in data.columns and \
        'ip_app_device_os_is_attributedcount' in  data.columns:
        print('gened ip_app_device_os_is_attributedsmoothcvr:')
        print(data[['ip_app_device_os_is_attributedsmoothcvr',
              'ip_app_device_os_is_attributedmean','ip_app_device_os_is_attributedcount']].sample(20).to_string())


    print('\n\n\n-------------\nDEBUG CVR DONE\n-------------\n\n\n')


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
                                + "_" + data['os'].astype(str) + "_" + data['channel'].astype(str)).apply(mmh3.hash) % D
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
                                + "_" + data['os'].astype(str) + "_" + data['channel'].astype(str)).apply(mmh3.hash) % D
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


def convert_features_to_text(data, predictors, hash = False):
    NR_BINS = 1000000

    def hashstr(input):
        #return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16) % (NR_BINS - 1) + 1)
        return str(mmh3.hash_from_buffer(input, signed=False) % (NR_BINS - 1) + 1)

    with timer('convert_features_to_text'):
        i = 0
        str_array = None
        assign_name_id = 0
        for feature in predictors:
            acro_name_to_dump = ''
            if not feature in acro_names:
                print('{} missing acronym, assign name AN{}'.format(feature, assign_name_id))
                acro_name_to_dump = 'AN' + str(assign_name_id)
                assign_name_id += 1
            else:
                acro_name_to_dump =  acro_names[feature]
            if str_array is None:
                str_array = acro_name_to_dump + "_" + data[feature].astype(str)
                if hash:
                    str_array = str_array.apply(hashstr)
            else:
                if not hash:
                    str_array = str_array + " " + acro_name_to_dump + "_" + data[feature].astype(str)
                else:
                    temp = (acro_name_to_dump + "_" + data[feature].astype(str)). \
                        apply(hashstr)
                    str_array = str_array + " " + temp

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
        if dump_train_data:
            train[predictors].to_csv("train_ft_dump.csv.bz2", compression='bz2',index=False)
            val[predictors].to_csv("val_ft_dump.csv.bz2", compression='bz2',index=False)
            print('dumping done')
            exit(0)
        print('training with :', predictors)
        # based on pub discussion, convert to float32 reduce half of mem spike of lgbm
        # https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/55325

        # dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,

        train_weights = None
        val_weights = None
        if config_scheme_to_use.pick_hours_weighted or config_scheme_to_use.adversial_val_weighted:
            with timer('setting weight'):
                if config_scheme_to_use.pick_hours_weighted:
                    train_weights = train['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5)
                    val_weights = val['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5)
                elif config_scheme_to_use.adversial_val_weighted:
                    ad_val_bst = lgb.Booster(model_file='ad_val_model.txt')
                    ad_val_predictors =  ['app','device',  'os', 'channel','ip','hour']
                    train_weights = ad_val_bst.predict(train[ad_val_predictors])
                    val_weights = ad_val_bst.predict(val[ad_val_predictors])

                    # normalization and only weight positive:
                    train_weights = (train_weights / np.min(train_weights) - 1.0 ) * train['is_attributed'] + 1.0
                    val_weights = (val_weights / np.min(val_weights) - 1.0) * val['is_attributed'] + 1.0

                    print('train weights:', pd.DataFrame({'weights':train_weights}).describe())
                    print('val weights:', pd.DataFrame({'weights:':val_weights}).describe())



        dtrain = lgb.Dataset(train[predictors].values.astype(np.float32), label=train[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical,
                             weight=train_weights
                             )
        dvalid = lgb.Dataset(val[predictors].values.astype(np.float32), label=val[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical,
                             weight=val_weights
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
                              verbose_eval=10,
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
            print('split importance:')
            pprint(sorted(zip(lgb_model.feature_name(),list(lgb_model.feature_importance())),
                          key=lambda x: x[1]))

            print('gain importance:')
            pprint(sorted(zip(lgb_model.feature_name(),list(lgb_model.feature_importance(importance_type='gain'))),
                          key=lambda x: x[1]))

        except:
            print('error sorting and zipping fts')

        importance_dict = dict(zip(lgb_model.feature_name(), list(lgb_model.feature_importance())))

        feature_imp = pd.DataFrame(lgb_model.feature_name(), list(lgb_model.feature_importance()))

        persist_model = True
        if persist_model:
            print('dumping model')
            lgb_model.save_model(get_dated_filename('model.txt'))
            print('dumping predictors of this model')
            pickle.dump(predictors1, open(get_dated_filename('predictors.pickle'), 'wb'))

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


def get_train_df():
    train = None
    if config_scheme_to_use.train_from is not None and isinstance(config_scheme_to_use.train_from, list):
        for data_from, data_to in \
                zip(sample_from_list, sample_to_list) if use_sample else \
                zip(config_scheme_to_use.train_from, config_scheme_to_use.train_to):
            train0 = pd.read_csv(path_train_sample if use_sample else path_train,
                                dtype=dtypes,
                                header=0,
                                usecols=train_cols,
                                skiprows=range(1, data_from),
                                nrows=data_to - data_from,
                                parse_dates=["click_time"])
            if train is None:
                train = train0
            else:
                train = train.append(train0)

            del train0
    else:
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
    elif config_scheme_to_use.train_filter and \
                    config_scheme_to_use.train_filter['filter_type'] == 'random_sample':
        train = train.sample(frac = config_scheme_to_use.train_filter['frac'])

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
    elif config_scheme_to_use.val_filter and \
                    config_scheme_to_use.val_filter['filter_type'] == 'random_sample':
        val = val.sample(frac = config_scheme_to_use.val_filter['frac'])
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
def get_test_supplement_df():
    test_supplement = pd.read_csv(path_test_supplement,
                       dtype=dtypes,
                       header=0,
                       skiprows=range(1,54583762),
                       usecols=test_cols,
                       nrows=1000 if use_sample else None,
                       parse_dates=["click_time"])
    test_supplement['is_attributed'] = 0
    return test_supplement

def get_combined_df(gen_test_data, load_test_supplement=False):

    test_len = 0
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
        test_len = len(test)
        train = train.append(test)
        del test
        gc.collect()

        if load_test_supplement:
            test_supplement = get_test_supplement_df()
            train = train.append(test_supplement)
            del test_supplement
            gc.collect()

    return train, train_len, val_len, test_len


def fit_batch(clf, X, y, w):
    if not isinstance(clf, FM_FTRL):
        clf.partial_fit(X, y)
    else:
        clf.partial_fit(X, y, sample_weight=w)

def predict_batch(clf, X):  return clf.predict(X)

def evaluate_batch(clf, X, y):
    auc= roc_auc_score(y, predict_batch(clf, X))

    print( "ROC AUC:", auc)
    return auc

class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return


def process_chunk_data(chunk, wb1, new_features, additional_categorical, ffm_data_file_handle=None, with_click_id = False):
    pick_hours = {4, 5, 10, 13, 14}
    D = 2 ** 20 # changed from 2**22 from 116_7
    batchsize = 10000000 // 2 # ATTENTION: in python3 / always returns float, for valid slice index ,it has to be int

    wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                         "lowercase": False, "n_features": D,
                                                         "norm": None, "binary": True})
                             , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)

    with timer('process chunk'):
        print('start adding interactive features')
        with timer('adding interactive features'):
            if config_scheme_to_use.use_interactive_features:
                print('gen_iteractive_categorical_features...')
                chunk = gen_iteractive_categorical_features(chunk)
        gc.collect()
        print('mem after iter fts:', cpuStats())

        predictors1 = categorical + additional_categorical + new_features
        print('converting chunk of len {} with features {}: '.format(len(chunk), predictors1))
        with timer('to text'):
            str_array = convert_features_to_text(chunk, predictors1, hash= ffm_data_file_handle is not None)
            print('converted to str array: ', str_array[10])

        if ffm_data_file_handle is not None:
            with timer('writing to str array csv file'):
                if with_click_id:
                    to_output = chunk[['click_id', 'is_attributed']].copy(True)
                    to_output['click_id'] = to_output['click_id'].astype(int)
                    to_output = to_output.set_index('click_id')
                else:
                    to_output = pd.DataFrame(chunk['is_attributed']).copy(True)
                to_output['str'] = to_output.index.astype(str) + ' ' + to_output['is_attributed'].astype(str) + ' ' + str_array.astype(str)

                np.savetxt(ffm_data_file_handle, to_output['str'].values, '%s')
                #to_output.to_csv(ffm_data_file_handle, header=False, index=False)
                return None, None, None

        print('gen weighted labels and recycle chunk')
        labels = []
        weights = []

        if 'is_attributed' in chunk.columns:
            labels = chunk['is_attributed'].values
            weights = np.multiply([1.0 if x == 1 else 0.2 for x in chunk['is_attributed'].values],
                                  chunk['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
        del (chunk)
        gc.collect()
        print('mem after converted to text and recycled chunk:', cpuStats())

        print('wordbatch processing')
        with timer('wordbatch processing'):
            X = wb.transform(str_array)
            del (str_array)
            gc.collect()
            print('mem after gc str array:', cpuStats())

    del wb
    gc.collect()
    print('mem after gc wb', cpuStats())

    return X, labels, weights

def ffm_data_gen(com_fts_list, use_ft_cache=False):

    with timer('load combined data df'):
        combined_df, train_len, val_len, test_len = get_combined_df(config_scheme_to_use.new_predict)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]
    test = combined_df[train_len + val_len:]

    predictors1 = categorical + new_features + ['is_attributed']
    print('dump train/val fe data for fts:', predictors1)

    if use_sample:
        train[predictors1].to_csv('train_fe_sample.csv', index=False)
        val[predictors1].to_csv('val_fe_sample.csv', index=False)
    else:
        train[predictors1].to_csv('train_fe.csv', index=False)
        val[predictors1].to_csv('val_fe.csv', index=False)

    print('dump dtypes')

    y = {k: str(v) for k, v in train.dtypes.to_dict().items()}
    print(y)
    del y['click_time']
    # del y['Unnamed: 0']
    pickle.dump(y, open('output_dtypes.pickle', 'wb'))

    print('dump test fe data...')

    click_id_df = pd.read_csv(path_test_sample if use_sample else path_test,
                         dtype='uint64',
                         header=0,
                         usecols=['click_id'])
    #pay attention to adding column of dataframe here, need to reset_index() first
    test_to_dump = test[predictors1].copy(True).reset_index(drop=True)
    test_to_dump['click_id'] = click_id_df['click_id']
    test_to_dump.to_csv('test_fe.csv' + '.sample' if use_sample else 'test_fe.csv', index=False)

    print('gen fe data for ffm done.')

def train_and_predict_online_model(com_fts_list, use_ft_cache=False, use_lgbm_fts =config_scheme_to_use.use_lgbm_fts,
                                   gen_ffm_data = False):
    batchsize = 10000000 // 2 # ATTENTION: in python3 / always returns float, for valid slice index ,it has to be int
    # https://stackoverflow.com/questions/42646915/typeerror-slice-indices-must-be-integers-or-none-or-have-an-index-method
    # wordbatch/batcher.py: 			data_split = [data.iloc[x * minibatch_size:(x + 1) * minibatch_size] for x in
	#                       					  range(int(ceil(len_data / minibatch_size)))]
    D = 2 ** 20 # changed from 2**22 from 116_7

    with timer('load combined data df'):
        combined_df, train_len, val_len, test_len = get_combined_df(config_scheme_to_use.new_predict)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]
    test = combined_df[train_len + val_len:]

    # ensure intermediate dump dir
    inter_dump_path = './inter_dump/'
    try:
        os.mkdir(inter_dump_path)
        print('created dir', inter_dump_path)
    except:
        None

    with timer('dump iter data'):
        train.to_csv(inter_dump_path + "train_iter_dump.csv.bz2", compression='bz2',index=False)
        val.to_csv(inter_dump_path + "val_iter_dump.csv.bz2", compression='bz2',index=False)
        if config_scheme_to_use.new_predict:
            test.to_csv(inter_dump_path + "test_iter_dump.csv.bz2", compression='bz2',index=False)

        y = {k: str(v) for k, v in train.dtypes.to_dict().items()}
        y1 = {k: str(v) for k, v in test.dtypes.to_dict().items()}

        y.update(y1)
        del y['click_time']

        print('dyptes:', y)
        dtypes = y

    del train
    del val
    del test

    gc.collect()
    print('mem after dump data', cpuStats())



    wb = None
    clf = None
    print('creating model...')
    with timer('creating model'):
        wb = wordbatch.WordBatch(None, extractor=(WordHash, {"ngram_range": (1, 1), "analyzer": "word",
                                                             "lowercase": False, "n_features": D,
                                                             "norm": None, "binary": True})
                                 , minibatch_size=batchsize // 80, procs=8, freeze=True, timeout=1800, verbose=0)

        clf = None
        if config_scheme_to_use.wordbatch_model == 'FM_FTRL':
            clf = FM_FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, alpha_fm=0.02, L2_fm=0.0, init_fm=0.01,
                          weight_fm=1.0,D_fm=8, e_noise=0.0, iters=3, inv_link="sigmoid", e_clip=1.0,
                          threads=4, use_avx=1, verbose=9) # iters changed to 2 from 116_7
                          #threads=4, use_avx=1, verbose=0)
        elif config_scheme_to_use.wordbatch_model == 'NN_ReLU_H1':
            clf = NN_ReLU_H1(alpha=0.05, D = D, verbose=9, e_noise=0.0, threads=4, inv_link="sigmoid")
        elif config_scheme_to_use.wordbatch_model == 'FTRL':
            clf = FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, iters=3, threads=4, verbose=9)
        else:
            print('invalid wordbatch_model param:', config_scheme_to_use.wordbatch_model)
            exit(-1)

    print('adding ip to categorical feature only in non lgbm models')
    categorical.append('ip')

    print('start streaming training...')
    p = None
    lgbm_ft_categorical = []
    with timer('train wordbatch model...'):
        train_chunks = pd.read_csv(inter_dump_path + "train_iter_dump.csv.bz2",
                                   dtype=dtypes,
                                   chunksize = batchsize,
                                   header=0,
                                   parse_dates=["click_time"]
                                  )
        if use_lgbm_fts:
            train_lgbm_fts_chunks = pd.read_csv("./lgbmft_dump/train_lgbm_ft_dump.csv.bz2",
                                   dtype='uint8',
                                   chunksize=batchsize,
                                   header=0
                                   )

        gen_ffm_data_file_handle = None
        if gen_ffm_data:
            gen_ffm_data_file_handle = open('train_fe.csv', 'w')

        for chunk in train_chunks:
            if use_lgbm_fts:
                lgbm_ft_chunk = next(train_lgbm_fts_chunks)
                lgbm_ft_categorical = list(lgbm_ft_chunk.columns)
                chunk = pd.concat([chunk, lgbm_ft_chunk], axis=1)

            X, labels, weights = process_chunk_data(chunk, wb, new_features, lgbm_ft_categorical, gen_ffm_data_file_handle)
            del (chunk)
            gc.collect()
            print('mem after converted to text and recycled chunk:', cpuStats())

            print('joining previous training threads...')
            with timer('join previous training threads:'):
                if p != None:
                    p.join()
                gc.collect()

            if not gen_ffm_data:
                print('start trainnig thread')
                with timer('start training thread'):
                    p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
                    p.start()
                    if config_scheme_to_use.sync_mode:
                        p.join()

        if gen_ffm_data:
            gen_ffm_data_file_handle.close()
            gen_ffm_data_file_handle = None

        print('joining train threads')
        if p != None:
            p.join()

        del train_chunks
        if use_lgbm_fts:
            del train_lgbm_fts_chunks
        del X
        del labels
        del weights
        gc.collect()

    print('evaling')
    with timer('eval wordbatch model...'):
        val_chunks = pd.read_csv(inter_dump_path + "val_iter_dump.csv.bz2",
                                   dtype=dtypes,
                                   chunksize = batchsize,
                                   header=0,
                                   parse_dates=["click_time"]
                                  )
        if use_lgbm_fts:
            val_lgbm_fts_chunks = pd.read_csv("./lgbmft_dump/val_lgbm_ft_dump.csv.bz2",
                                   dtype='uint8',
                                   chunksize=batchsize,
                                   header=0
                                   )
        if gen_ffm_data:
            gen_ffm_data_file_handle = open('val_fe.csv', 'w')
        for chunk in val_chunks:
            if use_lgbm_fts:
                chunk = pd.concat([chunk, next(val_lgbm_fts_chunks)], axis=1)

            X, labels, weights = process_chunk_data(chunk, wb, new_features,lgbm_ft_categorical, gen_ffm_data_file_handle)
            del (chunk)
            gc.collect()
            print('mem after converted to text and recycled chunk:', cpuStats())

            print('joining previous eval threads...')
            with timer('join previous eval threads:'):
                if p != None:
                    p.join()
                gc.collect()

            if not gen_ffm_data:
                print('start eval thread')
                with timer('start eval thread'):
                    p = threading.Thread(target=evaluate_batch, args=(clf, X, labels))
                    p.start()
                    if config_scheme_to_use.sync_mode:
                        p.join()

        if gen_ffm_data:
            gen_ffm_data_file_handle.close()
            gen_ffm_data_file_handle = None

        print('joining eval threads')
        if p != None:
            p.join()

        del val_chunks
        if use_lgbm_fts:
            del val_lgbm_fts_chunks

        del X
        del labels
        del weights
        gc.collect()

    if not config_scheme_to_use.new_predict:
        return
    print('predicting')
    click_ids = []
    test_preds = []
    with timer('predict wordbatch model...'):
        test_chunks = pd.read_csv(inter_dump_path + "test_iter_dump.csv.bz2",
                                   dtype=dtypes,
                                   chunksize = batchsize,
                                   header=0,
                                   parse_dates=["click_time"]
                                  )
        if use_lgbm_fts:
            test_lgbm_fts_chunks = pd.read_csv("./lgbmft_dump/test_lgbm_ft_dump.csv.bz2",
                                   dtype='uint8',
                                   chunksize=batchsize,
                                   header=0
                                   )
        if gen_ffm_data:
            gen_ffm_data_file_handle = open('test_fe.csv', 'w')
        for chunk in test_chunks:
            if use_lgbm_fts:
                chunk = pd.concat([chunk, next(test_lgbm_fts_chunks)], axis=1)

            X, labels, weights = process_chunk_data(chunk, wb, new_features,lgbm_ft_categorical, gen_ffm_data_file_handle, with_click_id = True)
            del (chunk)
            gc.collect()
            print('mem after converted to text and recycled chunk:', cpuStats())

            print('joining previous predict threads...')
            with timer('join previous predict threads:'):
                if p != None:
                    ret = p.join()
                    if ret is not None:
                        test_preds += list(ret)
                gc.collect()

            if not gen_ffm_data:
                print('start predict thread')
                with timer('start predict thread'):
                    p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
                    p.start()
                    if config_scheme_to_use.sync_mode:
                        test_preds += list(p.join())
                        p = None

        if gen_ffm_data:
            gen_ffm_data_file_handle.close()
            gen_ffm_data_file_handle = None

        print('joining eval threads')
        if p != None:
            test_preds += list(p.join())

        del test_chunks
        if use_lgbm_fts:
            del test_lgbm_fts_chunks

        del X
        del labels
        del weights
        gc.collect()

        if not gen_ffm_data:
            df_sub = pd.read_csv(path_test_sample if use_sample else path_test,
                               dtype='uint64',
                               header=0,
                               usecols=['click_id'])
            df_sub['is_attributed'] = test_preds
            df_sub.to_csv(get_dated_filename("wordbatch_fm_ftrl.csv"), index=False)
        print('done streaming prediction', ' for gen_ffm_data' if gen_ffm_data else '')

def train_and_predict(com_fts_list, use_ft_cache = False, only_cache=False,
                                         use_base_data_cache=False, gen_fts = False, load_test_supplement = False):
    with timer('load combined data df'):
        combined_df, train_len, val_len, test_len = get_combined_df(config_scheme_to_use.new_predict or gen_fts,
                                                                    load_test_supplement = load_test_supplement)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]
    if dump_train_data and config_scheme_to_use.new_predict:
        test = combined_df[train_len + val_len:]
        test[categorical + new_features].to_csv("test_ft_dump.csv.bz2", compression='bz2',index=False)

    with timer('train lgbm model...'):
        lgb_model, val_prediction, predictors, importances, val_auc = train_lgbm(train, val, new_features, False)

    if config_scheme_to_use.new_predict:
        with timer('predict test data:'):
            if not dump_train_data: # because for dump case, it'll be set above
                test = combined_df[train_len + val_len: train_len+val_len+test_len]

            print('NAN next click count in test:', len(test.query('ip_app_device_os_is_attributednextclick > 1489000000')))

            predict_result = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)
            submission = pd.DataFrame({'is_attributed':predict_result,
                                       'click_id':test['click_id'].astype('uint32').values})

            print("Writing the submission data into a csv file...")

            submission.to_csv(get_dated_filename("submission_notebook"), index=False)

            print("All done...")

    if gen_fts:
        #train
        with timer('predict train LGBM features:'):
            predict_result = lgb_model.predict(train[predictors], num_iteration=lgb_model.best_iteration, pred_leaf=True)

        with timer('create df of train LGBM features:'):
            ft_df = pd.DataFrame(predict_result, columns=['T' + str(i) for i in range(len(predict_result[0]))])


        with timer('dump df of train LGBM features:'):
            # ensure lgbm ft dump dir
            inter_dump_path = './lgbmft_dump/'
            try:
                os.mkdir(inter_dump_path)
                print('created dir', inter_dump_path)
            except:
                None

            ft_df.to_csv(inter_dump_path + "train_lgbm_ft_dump.csv.bz2", compression='bz2',index=False)

        del predict_result
        del ft_df
        gc.collect(
        )
        print('mem after train lgbm fts gen', cpuStats())

        with timer('predict val LGBM features:'):
            predict_result = lgb_model.predict(val[predictors], num_iteration=lgb_model.best_iteration, pred_leaf=True)

        with timer('create df of val LGBM features:'):
            ft_df = pd.DataFrame(predict_result, columns=['T' + str(i) for i in range(len(predict_result[0]))])


        with timer('dump df of val LGBM features:'):
            # ensure lgbm ft dump dir
            inter_dump_path = './lgbmft_dump/'
            ft_df.to_csv(inter_dump_path + "val_lgbm_ft_dump.csv.bz2", compression='bz2',index=False)


        del predict_result
        del ft_df
        gc.collect(
        )
        print('mem after val lgbm fts gen', cpuStats())

        with timer('predict test LGBM features:'):
            test = combined_df[train_len + val_len: train_len + val_len + test_len]
            predict_result = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration, pred_leaf=True)

        with timer('create df of test LGBM features:'):
            ft_df = pd.DataFrame(predict_result, columns=['T' + str(i) for i in range(len(predict_result[0]))])


        with timer('dump df of test LGBM features:'):
            # ensure lgbm ft dump dir
            inter_dump_path = './lgbmft_dump/'

            ft_df.to_csv(inter_dump_path + "test_lgbm_ft_dump.csv.bz2", compression='bz2',index=False)

        print('done gen test lgbm fts')

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

def ffm_data_gen_seperately(com_fts_list, use_ft_cache=False):
    with timer('loading train df:'):
        train = get_train_df()
    with timer('gen categorical features for train'):
        train = gen_categorical_features(train)

    with timer('gen statistical hist features for train'):
        train, new_features, discretization_bins_used = \
        generate_counting_history_features(train,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='train')

    predictors1 = categorical + new_features + ['is_attributed']

    if config_scheme_to_use.add_lgbm_fts_from_saved_model:
        lgb_model = lgb.Booster(model_file='train_config_124_3_model.txt')
        lgb_predictors = pickle.load(open('train_config_124_3_model_predictors.pickle', 'rb'))
        lgb_fts_count = 20

        with timer('predict train LGBM features:'):
            predict_result = lgb_model.predict(train[lgb_predictors], num_iteration=lgb_fts_count, pred_leaf=True)
            ft_df = pd.DataFrame(predict_result, dtype='uint8',
                                 columns=['T' + str(i) for i in range(len(predict_result[0]))])
            predictors1 = predictors1 + list(ft_df.columns)
            del predict_result
            gc.collect()
            train = pd.concat([train, ft_df], axis=1)
            del ft_df
            gc.collect()

    print('dump train fe data for fts:', predictors1)
    if use_sample:
        train[predictors1].to_csv('train_fe_sample.csv', index=False)
    else:
        train[predictors1].to_csv('train_fe.csv', index=False)

    print('dump dtypes')

    y = {k: str(v) for k, v in train.dtypes.to_dict().items()}
    print(y)
    del y['click_time']
    # del y['Unnamed: 0']
    pickle.dump(y, open('output_dtypes.pickle', 'wb'))

    del train
    gc.collect()
    print('mem after train dump:', cpuStats())

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
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='val')

    if config_scheme_to_use.add_lgbm_fts_from_saved_model:
        with timer('predict val LGBM features:'):
            predict_result = lgb_model.predict(val[lgb_predictors], num_iteration=lgb_fts_count, pred_leaf=True)
            ft_df = pd.DataFrame(predict_result, dtype='uint8',
                                 columns=['T' + str(i) for i in range(len(predict_result[0]))])
            del predict_result
            gc.collect()
            val = pd.concat([val, ft_df], axis=1)
            del ft_df
            gc.collect()

    print('dump val fe data for fts:', predictors1)

    if use_sample:
        val[predictors1].to_csv('val_fe_sample.csv', index=False)
    else:
        val[predictors1].to_csv('val_fe.csv', index=False)

    del val
    gc.collect()
    print('mem after val dump:', cpuStats())

    print('dump test fe data...')
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
    if config_scheme_to_use.add_lgbm_fts_from_saved_model:
        with timer('predict val LGBM features:'):
            predict_result = lgb_model.predict(test[lgb_predictors], num_iteration=lgb_fts_count, pred_leaf=True)
            ft_df = pd.DataFrame(predict_result, dtype='uint8',
                                 columns=['T' + str(i) for i in range(len(predict_result[0]))])
            del predict_result
            gc.collect()
            test = pd.concat([test, ft_df], axis=1)
            del ft_df
            gc.collect()

    click_id_df = pd.read_csv(path_test_sample if use_sample else path_test,
                         dtype='uint64',
                         header=0,
                         usecols=['click_id'])
    #pay attention to adding column of dataframe here, need to reset_index() first
    test_to_dump = test[predictors1].copy(True).reset_index(drop=True)
    test_to_dump['click_id'] = click_id_df['click_id']
    test_to_dump.to_csv('test_fe.csv' + '.sample' if use_sample else 'test_fe.csv', index=False)

    print('gen fe data for ffm done.')

def train_and_predict_gen_fts_seperately(com_fts_list, use_ft_cache = False, only_cache=False,
                                         use_base_data_cache=False):

    if config_scheme_to_use.train_smoothcvr_cache_from is not None:
        gen_smoothcvr_cache(config_scheme_to_use.train_smoothcvr_cache_from, config_scheme_to_use.train_smoothcvr_cache_to)

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

        if config_scheme_to_use.test_smoothcvr_cache_from is not None:
            clear_smoothcvr_cache()
            gen_smoothcvr_cache(config_scheme_to_use.test_smoothcvr_cache_from,
                                config_scheme_to_use.test_smoothcvr_cache_to)

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



def grid_search_features_combination(only_gen_ft_cache = False, use_lgbm_searcher = False):
    import itertools
    from random import shuffle

    # grid search for feature generations combinations:
    com_fts_list_to_use = []
    raw_cols0 = ['app', 'device', 'os', 'channel', 'hour', 'ip']
    raw_cols1 = ['app', 'device', 'os', 'channel', 'in_test_hh', 'ip']
    #for train_config_121_7
    #ops = ['nunique']
    # for train_config_121_8
    ops = []


    #ops = ['mean','var','nextclick','nunique','cumcount']

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

        #train_config_121_7,train_config_121_8
        add_count = False
        if add_count:
            for cols_count in range(1, 7):
                for cols_coms in itertools.combinations(raw_cols, cols_count):
                    temp = []
                    temp.extend(cols_coms)
                    temp.append('is_attributed')
                    com_fts_list_to_use.append({'group': list(temp), 'op': 'count'})

        add_cvr = False # has to use with combined
        if add_cvr:
            for cols_count in range(1, 7):
                for cols_coms in itertools.combinations(raw_cols, cols_count):
                    temp = []
                    temp.extend(cols_coms)
                    temp.append('is_attributed')
                    # add both mean and var:
                    com_fts_list_to_use.append({'group': list(temp), 'op': 'mean', 'astype':'float32'})
                    com_fts_list_to_use.append({'group': list(temp), 'op': 'var','astype':'float32'})

        add_smooth_cvr = True  # has to use with combined
        if add_smooth_cvr:
            for cols_count in range(1, 7):
                for cols_coms in itertools.combinations(raw_cols, cols_count):
                    temp = []
                    temp.extend(cols_coms)
                    temp.append('is_attributed')
                    # add both mean and var:
                    com_fts_list_to_use.append({'group': list(temp), 'op': 'mean', 'astype': 'float32'})
                    com_fts_list_to_use.append({'group': list(temp), 'op': 'var', 'astype': 'float32'})
    #print('added count coms(len: {}): {}'.format(len(com_fts_list_to_use), com_fts_list_to_use))

    do_shuffle = False
    if do_shuffle:
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
                    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'],'op': 'nextclick'},
                    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
                    ]

                if use_lgbm_searcher:
                    lgbm_params_search(additional_groups + com_fts_list_to_use[pos:pos + size])
                    importances = {}
                    auc = 0
                else:
                    jointly = False
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
    # search 128
    search_spaces_128 = {
                        'learning_rate': (0.03, 0.2, 'log-uniform'),
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
                        # 'is_unbalance': True,
                        #'n_estimators': (50),  # alias: num_boost_round
                        'scale_pos_weight': (98.0, 250.0, 'log-uniform')
                    }
    '''
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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}
    '''
    with timer('create bayes cv tunner'):
        bayes_cv_tuner = BayesSearchCV(
            estimator=lgb.LGBMClassifier(
                boosting_type= 'gbdt',
                objective= 'binary',
                metric= 'auc',
                n_jobs = 5,
                silent = False,
                subsample_for_bin = 200000,
                subsample_freq=1,
                min_split_gain=0,
                n_estimators =50
            ),
            search_spaces=search_spaces_128,
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
        combined_df, train_len, val_len, test_len = get_combined_df(config_scheme_to_use.new_predict)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           add_features_list=com_fts_list,
                                           ft_cache_prefix='joint')

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
        train_and_predict(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache)
    elif config_scheme_to_use.run_theme == 'train_and_predict_with_test_supplement':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list,
                          use_ft_cache=config_scheme_to_use.use_ft_cache,
                          load_test_supplement=True)
    elif config_scheme_to_use.run_theme == 'train_and_predict_gen_lgbm_fts':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache, gen_fts=True)
    elif config_scheme_to_use.run_theme == 'lgbm_params_search':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        lgbm_params_search(config_scheme_to_use.add_features_list)
    elif config_scheme_to_use.run_theme == 'train_and_predict_gen_fts_seperately':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict_gen_fts_seperately(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache)
    elif config_scheme_to_use.run_theme == 'online_model':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict_online_model(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache)
    elif config_scheme_to_use.run_theme == 'online_model_ffm':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict_online_model(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache, gen_ffm_data=True)
        gc.collect()
        print('train_fe.csv, test_fe.csv generated for ffm to train, mem after:', cpuStats())

        print('running ffm model in another process: ./mark/mark1/mark1 -r 0.11 -s 12 -t 40 test_fe.csv train_fe.csv')
        os.system('./mark/mark1/mark1 -r 0.11 -s 12 -t 40 test_fe.csv train_fe.csv')
        print('done...')
    elif config_scheme_to_use.run_theme == 'ffm_data_gen':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        ffm_data_gen(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache)
    elif config_scheme_to_use.run_theme == 'ffm_data_gen_seperately':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        ffm_data_gen_seperately(config_scheme_to_use.add_features_list,
                                use_ft_cache=config_scheme_to_use.use_ft_cache)
    else:
        print("nothing to run... exit")


with timer('run_model...'):
    run_model()

print('run_model done')
