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

# from wordbatch.data_utils import *
import threading
from sklearn.metrics import roc_auc_score

matplotlib.use('Agg')


def get_dated_filename(filename):
    return '{}.{}'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))
    # return filename


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    print('start ',name)
    yield
    print('[{}] done in {} s'.format(name, time.time() - t0))


print(os.listdir("../input"))

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

gen_test_input = True

read_path_with_hist = False

TRAIN_SAMPLE_DATA_LEN = 100001

# path = '../input/'
path = '../input/talkingdata-adtracking-fraud-detection/'
path_train_hist = '../input/data_with_hist/'
path_test_hist = '../input/data_with_hist/'

path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

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
    'early_stopping_round': 20,
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

shuffle_sample_filter = {'filter_type': 'sample', 'sample_count': 6}
shuffle_sample_filter_1_to_2 = {'filter_type': 'sample', 'sample_count': 2}

shuffle_sample_filter_1_to_10 = {'filter_type': 'sample', 'sample_count': 1}
shuffle_sample_filter_1_to_10k = {'filter_type': 'sample', 'sample_count': 1}

hist_ft_sample_filter = {'filter_type': 'hist_ft'}

skip = range(1, 140000000)
print("Loading Data")
# skiprows=skip,

import pickle


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
                 new_predict = False):
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


train_config_88_4 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 seperate_hist_files=False, add_hist_statis_fts=False,
                                 train_wordbatch=True,
                                 predict_wordbatch=True,
                                 log_discretization=True,
                                 discretization=0
                                 )

train_config_87 = ConfigScheme(False, True, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               seperate_hist_files=True, add_hist_statis_fts=True,
                               lgbm_params=new_lgbm_params
                               )
train_config_89 = ConfigScheme(False, False, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               train_wordbatch=True,
                               log_discretization=True
                               )
train_config_89_4 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 seperate_hist_files=False, add_hist_statis_fts=False,
                                 train_wordbatch=True,
                                 predict_wordbatch=True,
                                 log_discretization=True,
                                 use_interactive_features=True
                                 )
train_config_89_8 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 seperate_hist_files=False, add_hist_statis_fts=False,
                                 train_wordbatch=True,
                                 predict_wordbatch=True,
                                 log_discretization=True,
                                 use_interactive_features=True,
                                 wordbatch_model='FTRL'
                                 )

train_config_89_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 seperate_hist_files=False, add_hist_statis_fts=False,
                                 train_wordbatch=False,
                                 predict_wordbatch=True,
                                 log_discretization=True,
                                 use_interactive_features=True,
                                 train_wordbatch_streaming=True,
                                 train_start_time=None
                                 )

train_config_89_6 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 seperate_hist_files=False, add_hist_statis_fts=False,
                                 train_start_time=val_time_range_start,
                                 train_end_time=val_time_range_end,
                                 val_start_time=train_time_range_start,
                                 val_end_time=train_time_range_end,
                                 train_wordbatch=True,
                                 log_discretization=True,
                                 use_interactive_features=True
                                 )
train_config_87_3 = ConfigScheme(True, True, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 seperate_hist_files=True, add_hist_statis_fts=True,
                                 lgbm_params=new_lgbm_params
                                 )
train_config_93_1 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm
                                 )
train_config_94_1 = ConfigScheme(False, False, False,
                                 None,
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


def use_config_scheme(str):
    print('config values: ')
    pprint(vars(eval(str)))
    print('using config var name: ', str)
    return eval(str)


config_scheme_to_use = use_config_scheme('train_config_93_1')

print('test log 93_1')

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


def add_historical_statistical_features(data):
    print('adding historical statistic features...')
    feature_names = []
    cvr_columns_lists = [['ip', 'device'], ['app', 'channel']]

    for cvr_columns in cvr_columns_lists:
        sta_ft = data[cvr_columns + ['hour', 'day', 'is_attributed']].groupby(cvr_columns + ['day', 'hour'])[
            ['is_attributed']].mean().reset_index()

        sta_ft['day'] = sta_ft['day'] + 1

        new_col_name = '_'.join(cvr_columns + ['cvr'])
        sta_ft = sta_ft.rename(columns={'is_attributed': new_col_name})
        data = data.merge(sta_ft, on=cvr_columns + ['day', 'hour'], how='left')

        data[new_col_name] = data[new_col_name].astype('float32')

        del sta_ft
        gc.collect()

        feature_names.append(new_col_name)
        print('new feature {} added'.format(new_col_name))
    return data, feature_names


def df_get_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    return counts[unqtags]


def add_statistic_feature(group_by_cols, training, counting_col='channel', qcut_count=0.98,
                          discretization=0, discretization_bins=None,
                          log_discretization=False):
    features_added = []
    feature_name_added = '_'.join(group_by_cols) + 'count'
    discretization_bins_used = {}
    print('count ip with group by:', group_by_cols)
    n_chans = training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]] \
        .count().reset_index().rename(columns={counting_col: feature_name_added})
    training = training.merge(n_chans, on=group_by_cols, how='left')
    del n_chans

    gc.collect()
    if not log_discretization and discretization == 0:
        if training[feature_name_added].max() <= 65535:
            training[feature_name_added] = training[feature_name_added].astype('uint16')

    if not log_discretization and qcut_count != 0 and discretization == 0:
        colmax = training[feature_name_added].max()
        quantile_cut = training[feature_name_added].quantile(qcut_count)
        training[feature_name_added] = training[feature_name_added].apply(
            lambda x: x if x < quantile_cut else colmax)

    if log_discretization:
        if training[feature_name_added].min() < 0:
            print('!!!! invalid time in {}, fix it.....'.format(feature_name_added))
            training[feature_name_added] = training[feature_name_added].apply(lambda x: np.max([0, x]))
        training[feature_name_added] = np.log2(1 + training[feature_name_added].values).astype(int)
        print('log dicretizing feature:', feature_name_added)
    elif discretization != 0:
        print('before qcut', feature_name_added, training[feature_name_added].describe())
        if discretization_bins is None:
            ret, discretization_bins_used[feature_name_added] = pd.qcut(training[feature_name_added], discretization,
                                                                        labels=False, duplicates='drop', retbins=True)
            training[feature_name_added] = ret.fillna(0).astype('uint16')
        else:
            training[feature_name_added] = pd.cut(training[feature_name_added],
                                                  discretization_bins[feature_name_added],
                                                  labels=False).fillna(0).astype('uint16')
        print('after qcut', feature_name_added, training[feature_name_added].describe())

    features_added.append(feature_name_added)

    print('added features:', features_added)
    print(training[feature_name_added].describe())
    print('nan count: ', training[feature_name_added].isnull().sum())

    return training, features_added, discretization_bins_used


def generate_counting_history_features(data,
                                       discretization=0, discretization_bins=None):
    print('discretization bins to use:', discretization_bins)

    new_features = []

    add_features_list = [

        # ====================
        # my best features
        {'group': ['ip', 'day', 'hour'], 'with_hist': False, 'counting_col': 'channel'},
        {'group': ['ip', 'day', 'hour', 'os'], 'with_hist': False, 'counting_col': 'channel'},
        {'group': ['ip', 'day', 'hour', 'app'], 'with_hist': False, 'counting_col': 'channel'},
        {'group': ['ip', 'day', 'hour', 'app', 'os'], 'with_hist': False, 'counting_col': 'channel'},
        {'group': ['app', 'day', 'hour'], 'with_hist': False, 'counting_col': 'channel'},
        {'group': ['ip', 'in_test_hh'], 'with_hist': False, 'counting_col': 'channel'}
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

    discretization_bins_used = None

    for add_feature in add_features_list:
        data, features_added, discretization_bins_used_current_feature = add_statistic_feature(
            add_feature['group'],
            data,
            counting_col=add_feature['counting_col'],
            discretization=discretization,
            discretization_bins=discretization_bins,
            log_discretization=config_scheme_to_use.log_discretization)
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


def train_lgbm(train, val, new_features):

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

        feature_imp = pd.DataFrame(lgb_model.feature_name(), list(lgb_model.feature_importance()))

        if persist_intermediate:
            print('dumping model')
            lgb_model.save_model(get_dated_filename('model.txt'))

        do_val_prediction = False
        val_prediction = None
        if do_val_prediction:
            print("Writing the val_prediction into a csv file...")
            # if persist_intermediate:

            print('gen val prediction')
            val_prediction = lgb_model.predict(val[predictors1], num_iteration=lgb_model.best_iteration)
            val['predict'] = val_prediction
            # pd.Series(val_prediction).to_csv(get_dated_filename("val_prediction.csv"), index=False)
            val.to_csv(get_dated_filename("val_prediction.csv"), index=False)

    return lgb_model, val_prediction, predictors1


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


def get_combined_df(gen_test_data):

    print('loading train data...')
    train = pd.read_csv(path_train_sample if use_sample else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.train_from) \
                            if not use_sample and config_scheme_to_use.train_from is not None else None,
                        nrows=config_scheme_to_use.train_to - config_scheme_to_use.train_from \
                            if not use_sample and config_scheme_to_use.train_from is not None else None,
                        parse_dates=["click_time"])


    if config_scheme_to_use.train_filter and \
                    config_scheme_to_use.train_filter['filter_type'] == 'sample':
        sample_count = config_scheme_to_use.train_filter['sample_count']
        train = train.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()

    gc.collect()

    train_len = len(train)

    print('loading val data')
    val = pd.read_csv(path_train_sample if use_sample else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.val_from) \
                            if not use_sample and config_scheme_to_use.val_from is not None else None,
                        nrows=config_scheme_to_use.val_to - config_scheme_to_use.val_from \
                            if not use_sample and config_scheme_to_use.val_from is not None else None,
                        parse_dates=["click_time"])


    if config_scheme_to_use.val_filter and \
        config_scheme_to_use.val_filter['filter_type'] == 'sample':
        sample_count = config_scheme_to_use.val_filter['sample_count']
        val = val.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()

    gc.collect()

    val_len = len(val)

    train = train.append(val)

    del val
    gc.collect()

    if gen_test_data:
        test = pd.read_csv(path_test_sample if use_sample else path_test,
                            dtype=dtypes,
                            header=0,
                            usecols=test_cols,
                            parse_dates=["click_time"])
        train = train.append(test)
        del test
        gc.collect()

    return train, train_len, val_len


if config_scheme_to_use.new_train:
    with timer('load combined data df'):
        combined_df, train_len, val_len = get_combined_df(config_scheme_to_use.new_predict)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)
    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]

    with timer('train lgbm model...'):
        lgb_model, val_prediction, predictors = train_lgbm(train, val, new_features)

    if config_scheme_to_use.new_predict:
        with timer('predict test data:'):
            test = combined_df[train_len + val_len:]

            predict_result = lgb_model.predict(test[predictors], num_iteration=lgb_model.best_iteration)
            submission = pd.DataFrame({'is_attributed':predict_result,
                                       'click_id':test['click_id'].astype('uint32').values})

            print("Writing the submission data into a csv file...")

            submission.to_csv(get_dated_filename("submission_notebook.csv"), index=False)

            print("All done...")



if config_scheme_to_use.ffm_data_gen:
    gen_ffm_data()
