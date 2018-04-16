
# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys

on_kernel = True

if on_kernel:
    sys.path.insert(0, '../input/wordbatch-133/wordbatch/')
    sys.path.insert(0, '../input/randomstate/randomstate/')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

#from wordbatch.data_utils import *
import threading
from sklearn.metrics import roc_auc_score

matplotlib.use('Agg')

def get_dated_filename(filename):
    return '{}.{}'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))
    #return filename


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle
from contextlib import contextmanager
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {} s'.format(name, time.time() - t0))

print(os.listdir("../input"))


import os, psutil
def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    gc.collect()
    #all_objects = muppy.get_objects()
    #sum1 = summary.summarize(all_objects)
    #summary.print_(sum1)

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

use_sample = True
persist_intermediate = False

gen_test_input = True

read_path_with_hist = False

TRAIN_SAMPLE_DATA_LEN = 100001

#path = '../input/'
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
#with ip:
#categorical = ['app', 'device', 'os', 'channel', 'hour', 'ip']


cvr_columns_lists = [
    ['ip', 'app', 'device', 'os', 'channel'],
    ['app', 'os'],
    ['app','channel'],
    ['app', 'device'],
    ['ip','device'],
    ['ip'], ['os'], ['channel']
]
agg_types = ['non_attr_count', 'cvr']

acro_names = {
                 'ip':'I',
                 'app':'A',
                 'device':'D',
                 'os': 'O',
                 'channel' : 'C',
                 'hour' :'H',
                 'ip_day_hourcount' : 'IDH-',
                 'ip_day_hour_oscount' : 'IDHO-',
                 'ip_day_hour_appcount' : 'IDHA-',
                 'ip_day_hour_app_oscount': 'IDHAO-',
                 'ip_app_oscount':"IAO-",
                 'ip_appcount':"IA-",
                 'ip_devicecount':"ID-",
                 'app_channelcount':"AC-",
                 'app_day_hourcount' : 'ADH-',
                 'ip_in_test_hhcount' : "IITH-",
                 'next_click' : 'NC',
                 'app_channel': 'AC',
                 'os_channel': 'OC',
                 'app_device': 'AD',
                 'app_os_channel' :'AOC',
                 'ip_app': 'IA',
                 'app_os':'AO'
              }

hist_st = []
iii = 0
for type in agg_types:
    for cvr_columns in cvr_columns_lists:
        new_col_name = '_'.join(cvr_columns)  + '_' + type
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



shuffle_sample_filter = {'filter_type': 'sample', 'sample_count': 6}
shuffle_sample_filter_1_to_2 = {'filter_type': 'sample', 'sample_count': 2}

shuffle_sample_filter_1_to_10 = {'filter_type': 'sample', 'sample_count': 1}
shuffle_sample_filter_1_to_10k = {'filter_type': 'sample', 'sample_count': 1}

hist_ft_sample_filter = {'filter_type': 'hist_ft'}

skip = range(1, 140000000)
print("Loading Data")
#skiprows=skip,

import pickle




class ConfigScheme:
    def __init__(self, predict = False, train = True, ffm_data_gen = False,
               train_filter = None,
               val_filter = shuffle_sample_filter,
               test_filter = None,
               lgbm_params = default_lgbm_params,
               discretization = 0,
               mock_test_with_val_data_to_test = False,
               train_start_time = train_time_range_start,
               train_end_time = train_time_range_end,
               val_start_time = val_time_range_start,
               val_end_time = val_time_range_end,
               gen_ffm_test_data = False,
               add_hist_statis_fts = False,
               seperate_hist_files = False,
               train_wordbatch = False,
               log_discretization = False,
               predict_wordbatch = False,
               use_interactive_features = False,
               wordbatch_model = 'FM_FTRL',
               train_wordbatch_streaming = False):
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


train_config_88_4 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_wordbatch=True,
                               predict_wordbatch = True,
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
                               predict_wordbatch = True,
                               log_discretization=True,
                               use_interactive_features=True
                               )
train_config_89_8 = ConfigScheme(False, False, False,
                               None,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_wordbatch=True,
                               predict_wordbatch = True,
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
                               predict_wordbatch = True,
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

def use_config_scheme(str):
    print('config values: ')
    pprint(vars(eval(str)))
    print('using config var name: ', str)
    return eval(str)

config_scheme_to_use = use_config_scheme('train_config_89_4')

print('test log 89_4 no ip')

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
        dtypes[ft]= 'float32'

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
        data['in_test_hh'] = (   3
                               - 2*data['hour'].isin(  most_freq_hours_in_test_data )
                               - 1*data['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
        #categorical.append('in_test_hh')
    return data

def gen_iteractive_categorical_features(data):
    if config_scheme_to_use.use_interactive_features:
        interactive_features_list = [
            ['app','channel'],
            ['os','channel'],
            ['app','device'],
            ['app','os','channel'],
            ['ip','app'],
            ['app','os']
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
        #print(sta_ft.describe())
        #sta_ft.info()

        sta_ft['day'] = sta_ft['day'] + 1

        new_col_name = '_'.join(cvr_columns + ['cvr'])
        sta_ft = sta_ft.rename(columns={'is_attributed': new_col_name})
        data = data.merge(sta_ft, on=cvr_columns + ['day', 'hour'], how='left')

        data[new_col_name] = data[new_col_name].astype('float32')

        import gc
        del sta_ft
        gc.collect()

        #print(data)
        #print(data.describe())
        #data.info()
        feature_names.append(new_col_name)
        print('new feature {} added'.format(new_col_name))
    return data, feature_names

def prepare_data(data, training_day, profile_days, filter_config=None,
                 with_hist_profile=True, only_for_ip_with_hist = False, for_test = False,
                 start_time=None, end_time=None,
                 start_hist_time=None):
    if filter_config is not None:
        if filter_config['filter_type'] == 'sample':
            sample_count = filter_config['sample_count']
            #sample 1/4 of the data:
            data = data.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()

        elif filter_config['filter_type'] == 'filter_field':
            query_str = ' | '.join(['%s == \'%s\'' % (filter_config['filter_field'], value)
                                    for value in filter_config['filter_field_values']])
            data = data.query(query_str)
        elif filter_config['filter_type'] == 'hist_ft':
            for ft in hist_st:
                if aa is None:
                    aa = pd.notnull(data[ft])
                else:
                    aa = aa | pd.notnull(data[ft])
            data = data.loc[aa]

        len_train = len(data)
        print('len after filter %s: %s', (filter_config['filter_type'], len_train))

        gc.collect()

    train_ip_contains_training_day = None
    train_ip_contains_training_day_attributed = None

    if with_hist_profile:
        #train_ip_contains_training_day = data.groupby('ip').filter(lambda x: x['day'].max() == training_day)

        #print('train_ip_contains_training_day', train_ip_contains_training_day)
        #print('train_ip_contains_training_day unique ips:', len(train_ip_contains_training_day['ip'].unique()))

        #if only_for_ip_with_hist:
        #    data = train_ip_contains_training_day.groupby('ip').filter(lambda x: x['day'].min() < training_day)

        #train_ip_contains_training_day = train_ip_contains_training_day  \
        #    .query('day < {0} & day > {1}'.format(training_day, training_day - 1 - profile_days) )

        print('original len', len(data))
        print('original unique ips:', len(data['ip'].unique()))

        train_ip_contains_training_day = data.set_index('click_time').ix[start_hist_time:start_time].reset_index()
        print('train_ip_contains_training_day len', len(train_ip_contains_training_day))
        print('train_ip_contains_training_day unique ips:', len(train_ip_contains_training_day['ip'].unique()))

        print('split attributed data:')
        train_ip_contains_training_day_attributed = train_ip_contains_training_day.query('is_attributed == 1')
        print('len:',len(train_ip_contains_training_day_attributed))

    #only use data on 9 to train, but data before 9 as features
    if start_time is None:
        train = data.query('day == {}'.format(training_day))
    else:
        xx = parser.parse(start_time)
        yy = parser.parse(end_time)

        print('filter time range: {} - {}, len before filter:{}'.format(start_time, end_time, len(data)))
        train = data.set_index('click_time').ix[start_time:end_time].reset_index()
        print('filter time len after filter:{}'.format(len(train)))
    print('training data len:', len(train))
    print('train unique ips:', len(train['ip'].unique()))
    
    return train, \
           train_ip_contains_training_day, train_ip_contains_training_day_attributed


def df_get_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(0) + 1),
                                     return_inverse=True, return_counts=True)
    return counts[unqtags]

def add_statistic_feature(group_by_cols, training, training_hist, training_hist_attribution,
                          with_hist, counting_col='channel', cast_type=True, qcut_count=0.98,
                          discretization=0, discretization_bins = None,
                          log_discretization = False):
    features_added = []
    feature_name_added = '_'.join(group_by_cols) + 'count'
    discretization_bins_used = {}
    print('count ip with group by:', group_by_cols)
    n_chans = training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]] \
        .count().reset_index().rename(columns={counting_col: feature_name_added})
    training = training.merge(n_chans, on=group_by_cols, how='left')
    del n_chans

    #training[feature_name_added] = df_get_counts(training,group_by_cols )
    gc.collect()
    if not log_discretization and discretization == 0:
        if training[feature_name_added].max() <= 65535:
            training[feature_name_added] = training[feature_name_added].astype('uint16')

    if not log_discretization and qcut_count != 0 and  discretization==0:
        colmax = training[feature_name_added].max()
        #print('before qcut', feature_name_added, training[feature_name_added].describe())
        quantile_cut = training[feature_name_added].quantile(qcut_count)
        training[feature_name_added] = training[feature_name_added].apply(
            lambda x: x if x < quantile_cut else colmax)
        #print('after qcut', feature_name_added, training[feature_name_added].describe())

    if log_discretization:
        #print('feature: {}:{}'.format(feature_name_added, training[feature_name_added]))
        #print('feature: {} describe:{}'.format(feature_name_added, training[feature_name_added].describe()))
        if training[feature_name_added].min() < 0:
            print('!!!! invalid time in {}, fix it.....'.format(feature_name_added))
            training[feature_name_added] = training[feature_name_added].apply(lambda x: np.max([0, x]))
        training[feature_name_added] = np.log2(1 + training[feature_name_added].values).astype(int)
        print('log dicretizing feature:', feature_name_added)
    elif discretization != 0:
        print('before qcut', feature_name_added, training[feature_name_added].describe())
        if discretization_bins is None:
            ret, discretization_bins_used[feature_name_added]= pd.qcut(training[feature_name_added], discretization,
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

def generate_counting_history_features(data, history, history_attribution,
                                       with_hist_profile = True, remove_hist_profile_count=0,
                                       discretization=0, discretization_bins=None):
    print('discretization bins to use:', discretization_bins)
        
    new_features = []

    add_features_list = [

        #====================
        # my best features
        {'group':['ip','day','hour'], 'with_hist': False, 'counting_col':'channel'},
        {'group':['ip','day','hour', 'os'], 'with_hist': False, 'counting_col':'channel'},
        {'group':['ip','day','hour','app'], 'with_hist': False, 'counting_col':'channel'},
        {'group':['ip','day','hour','app','os'], 'with_hist': False, 'counting_col':'channel'},
        {'group':['app','day','hour'], 'with_hist': False, 'counting_col':'channel'},
        {'group': ['ip', 'in_test_hh'], 'with_hist': with_hist_profile, 'counting_col': 'channel'}
        #=====================


        # try word batch featuers:
        #=====================
        #{'group': ['ip', 'day', 'hour'], 'with_hist': False, 'counting_col': 'channel'},
        #{'group': ['ip', 'app'], 'with_hist': False, 'counting_col': 'channel'},
        #{'group': ['ip', 'app', 'os'], 'with_hist': False, 'counting_col': 'channel'},
        #{'group': ['ip', 'device'], 'with_hist': False, 'counting_col': 'channel'},
        #{'group': ['app', 'channel'], 'with_hist': False, 'counting_col': 'os'},
        #======================



        #{'group':['app'], 'with_hist': False, 'counting_col':'channel'},
        #{'group': ['os'], 'with_hist': False, 'counting_col': 'channel'},
        #{'group': ['device'], 'with_hist': False, 'counting_col': 'channel'},
        #{'group': ['channel'], 'with_hist': False, 'counting_col': 'os'},
        #{'group': ['hour'], 'with_hist': False, 'counting_col': 'os'},

        #{'group':['ip','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        #{'group':['ip','os', 'app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        #{'group':['ip'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        #{'group':['ip','hour','channel'], 'with_hist': with_hist_profile, 'counting_col':'os'},
        #{'group':['ip','hour','os'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        #{'group':['ip','hour','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        #{'group':['channel','app'], 'with_hist': with_hist_profile, 'counting_col':'os'},
        #{'group':['channel','os'], 'with_hist': with_hist_profile, 'counting_col':'app'},
        #{'group':['channel','app','os'], 'with_hist': with_hist_profile, 'counting_col':'device'},
        #{'group':['os','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        ]

    discretization_bins_used = None

    for add_feature in add_features_list:
        data, features_added, discretization_bins_used_current_feature = add_statistic_feature(
            add_feature['group'],
            #data[add_feature['group'] + [add_feature['counting_col']]],
            data,
            history, history_attribution, add_feature['with_hist'],
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


    if remove_hist_profile_count != 0:
        data = data.query('ipcount_in_hist > {}'.format(remove_hist_profile_count))

    if discretization_bins is None:
        print('discretization bins used:',discretization_bins_used )
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
        data['epochtime'] = pd.to_datetime(data['click_time']).astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, time in zip(reversed(data['category'].values), reversed(data['epochtime'].values)):
            next_clicks.append(click_buffer[category] - time)
            click_buffer[category] = time
        del (click_buffer)
        data['next_click'] = list(reversed(next_clicks))

        if discretization!=0:
            print('min of next click: {}, max: {}'.format(data['next_click'].min(), data['next_click'].max()))
            if data['next_click'].min() < 0:
                print('!!!! invalid time in next click, fix it.....')
                data['next_click'] = data['next_click'].apply(lambda x: np.max([0, x]))
            data['next_click'] = np.log2(1 + data['next_click'].values).astype(int)
        data.drop('epochtime',inplace=True,axis=1)
        data.drop('category',inplace=True,axis=1)

    print('next click ', data['next_click'])
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
            click_count_later  = []
            for category in reversed(data['category'].values):
                click_count_later.append(click_count_buffer[category])
                click_count_buffer[category] += 1
            del (click_count_buffer)
            data['click_count_later'] = list(reversed(click_count_later))
        gc.collect()
        new_features = new_features + ['click_count_later']

    return data, new_features, discretization_bins_used

#test['hour'] = test["click_time"].dt.hour.astype('uint8')
#test['day'] = test["click_time"].dt.day.astype('uint8')



def gen_train_df(with_hist_profile = True, persist_fe_data = False):


    train = pd.read_csv(path_train_sample if use_sample else path_train, dtype=dtypes,
                        compression='gzip' if \
                            config_scheme_to_use.add_hist_statis_fts and \
                            not config_scheme_to_use.seperate_hist_files else None,
            header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)
    lentrain = len(train)

    # filter first before join cvr cols
    train = train.set_index('ip').loc[lambda x: (x.index + 401) % 6 == 0].reset_index()
    print('len after filter',len(train))

    if config_scheme_to_use.seperate_hist_files:
        for ft in hist_st:
            csv_file = Path(path_train_hist +ft+ '.train.csv')
            csv_gzip_file = Path(path_train_hist + ft + '.train.csv.gzip')
            csv_bz2_file = Path(path_train_hist + ft + '.train.csv.bz2')
            ft_data = None
            if csv_file.is_file():
                ft_data = next(pd.read_csv(path_train_hist +ft+ '.train.csv', dtype={ft:'float32'},
                                    header=0, engine='c',
                                    chunksize = lentrain))  # .sample(1000)
            elif csv_gzip_file.is_file():
                ft_data = next(pd.read_csv(path_train_hist +ft+ '.train.csv.gzip', dtype={ft:'float32'},
                                    header=0, engine='c',compression='gzip',
                                    chunksize = lentrain))  # .sample(1000)
            elif csv_bz2_file.is_file():
                ft_data = next(pd.read_csv(path_train_hist +ft+ '.train.csv.bz2', dtype={ft:'float32'},
                                    header=0, engine='c',compression='bz2',
                                    chunksize = lentrain))  # .sample(1000)
                print(path_train_hist +ft+ '.train.csv.bz2' + ' loaded')
            else:
                print('{} not found!!!'.format(ft))
                exit(-1)

            train[ft] = ft_data
            del ft_data
            gc.collect()



    len_train = len(train)
    print('The initial size of the train set is', len_train)
    print('Binding the training and test set together...')
    train = gen_categorical_features(train)


    train_data, train_ip_contains_training_day, train_ip_contains_training_day_attributed =  \
        prepare_data(train, 8, 2, config_scheme_to_use.train_filter, with_hist_profile,
                     start_time=config_scheme_to_use.train_start_time,
                     end_time=config_scheme_to_use.train_end_time, start_hist_time='2017-11-06 0:00:00')

    train_data, new_features, discretization_bins_used = generate_counting_history_features(train_data, train_ip_contains_training_day,
                                                                  train_ip_contains_training_day_attributed,
                                                                  with_hist_profile,
                                                                  discretization=config_scheme_to_use.discretization)

    train_data = train_data.set_index('click_time'). \
        ix[config_scheme_to_use.train_start_time:config_scheme_to_use.train_end_time].reset_index()

    print('train data:', train_data)
    print('new features:', new_features)
    #print('train data ip count in hist:', train_data['ipcount_in_hist'].describe())
    #print('train data min ', train_ip_contains_training_day.groupby('ip')['day'].min())

    #gen val data:
    #val = train.set_index('ip').loc[lambda x: (x.index) % 17 == 0].reset_index()
    #print(val)
    #print('The size of the validation set is ', len(val))

    del train_ip_contains_training_day
    del train_ip_contains_training_day_attributed
    gc.collect()


    val, train_ip_contains_training_day, train_ip_contains_training_day_attributed =  \
        prepare_data(train, 9, 2, config_scheme_to_use.val_filter, with_hist_profile,
                     start_time=config_scheme_to_use.val_start_time,
                     end_time=config_scheme_to_use.val_end_time, start_hist_time='2017-11-07 0:00:00')

    print('len val:', len(val))
    val, new_features1, _ = generate_counting_history_features(val, train_ip_contains_training_day,
                                                           train_ip_contains_training_day_attributed,
                                                           with_hist_profile,
                                                           discretization=config_scheme_to_use.discretization,
                                                           discretization_bins=discretization_bins_used)

    val = val.set_index('click_time').\
              ix[config_scheme_to_use.val_start_time:config_scheme_to_use.val_end_time].reset_index()
    train = train_data

    del train_ip_contains_training_day
    del train_ip_contains_training_day_attributed
    del train_data
    gc.collect()
    #train = train.set_index('ip').loc[lambda x: (x.index) % 17 != 0].reset_index()
    #print('The size of the train set is ', len(train))

    target = 'is_attributed'
    train[target] = train[target].astype('uint8')
    train.info()

    if persist_fe_data:
        predictors1 = categorical + new_features + ['is_attributed']
        if use_sample:
            train[predictors1 ].to_csv(get_dated_filename('train_fe_sample.csv'), index=False)
            val[predictors1].to_csv(get_dated_filename('val_fe_sample.csv'), index=False)
        else:
            train[predictors1].to_csv(get_dated_filename('train_fe.csv'), index=False)
            val[predictors1].to_csv(get_dated_filename('val_fe.csv'), index=False)

        print('save dtypes')

        y = {k: str(v) for k, v in train.dtypes.to_dict().items()}
        print(y)
        del y['click_time']
        #del y['Unnamed: 0']
        pickle.dump(y,open(get_dated_filename('fe_dtypes.pickle'),'wb'))

    print('all new features: {}'.format('_'.join(new_features)))
    return train, val, new_features, discretization_bins_used


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

def convert_features_to_text_iter(data, predictors):

    with timer('convert_features_to_text'):
        i = 0

        lendata = len(data)
        #str_array = np.zeros(lendata, dtype=np.dtype('U'))
        str_array = []

        for ix, row in data.iterrows():
            row_str = None
            for feature in predictors:
                if not feature in acro_names:
                    print('{} missing acronym'.format(feature))
                    exit(-1)
                if row_str is None:
                    row_str = acro_names[feature] + "_" + str(row[feature])
                else:
                    row_str = row_str + " " + acro_names[feature] + "_" + str(row[feature])

            #str_array[i] = row_str
            str_array.append(row_str)

            del row_str
                #gc.collect()
                #print('mem after gc:', cpuStats())
            i += 1

        gc.collect()
        print('mem after gc:', cpuStats())
        ret = np.array(str_array)
        del str_array

        return ret


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




def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def train_wordbatch_model_streaming():
    pick_hours = {4, 5, 10, 13, 14}
    batchsize = 10000000
    D = 2 ** 22

    if use_sample:
        batchsize = TRAIN_SAMPLE_DATA_LEN // 5

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
                          threads=4, use_avx=1, verbose=0)
        elif config_scheme_to_use.wordbatch_model == 'NN_ReLU_H1':
            clf = NN_ReLU_H1(alpha=0.05, D = D, verbose=9, e_noise=0.0, threads=4, inv_link="sigmoid")
        elif config_scheme_to_use.wordbatch_model == 'FTRL':
            clf = FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, iters=3, threads=4, verbose=9)
        else:
            print('invalid wordbatch_model param:', config_scheme_to_use.wordbatch_model)
            exit(-1)

    target = 'is_attributed'
    #predictors1 = categorical + new_features

    #if config_scheme_to_use.add_hist_statis_fts:
    #    predictors1 = predictors1 + hist_st

    p = None

    file_read_chunk_id = -1

    mini_chunk_id = -1

    with timer('train wordbatch model...'):
        train_chunks = pd.read_csv(path_train_sample if use_sample else path_train, dtype=dtypes,
                                   chunksize = batchsize,
                                   header=0,
                                   usecols=train_cols,
                                   parse_dates=["click_time"])
        for chunk in train_chunks:
            file_read_chunk_id += 1
            print('procssing chunk #{} in file'.format(file_read_chunk_id))
            # convert features to text:
            if config_scheme_to_use.train_filter is not None and \
                            config_scheme_to_use.train_filter['filter_type'] == 'sample':
                sample_count = config_scheme_to_use.train_filter['sample_count']
                # sample 1/4 of the data:
                chunk = chunk.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()

            if config_scheme_to_use.train_start_time is not None:
                chunk = chunk.set_index('click_time'). \
                     ix[config_scheme_to_use.train_start_time:config_scheme_to_use.train_end_time].reset_index()

                if not use_sample and len(chunk) < 500000:
                    print('len {} too short to train after filter {} - {}, skip this batch'.format(
                        len(chunk),
                        config_scheme_to_use.train_start_time,
                        config_scheme_to_use.train_end_time))
                    continue

            chunk = gen_categorical_features(chunk)

            chunk, new_features, discretization_bins_used = \
                generate_counting_history_features(chunk,
                                                   None,
                                                   None,
                                                   False,
                                                   discretization=config_scheme_to_use.discretization)

            print('debug: ip_day_hour_app_oscount nan count: ')
            print(chunk['ip_day_hour_app_oscount'].isnull().sum())


            print('mem after gen ft:', cpuStats())
            chunk.drop('click_time', inplace=True, axis=1)
            print('mem after drop clk time:', cpuStats())
            gc.collect()
            print('mem after gc:', cpuStats())
            chunk.info()

            if config_scheme_to_use.use_interactive_features:
                print('gen_iteractive_categorical_features...')
                chunk = gen_iteractive_categorical_features(chunk)

            gc.collect()
            print('mem after iter fts:', cpuStats())

            if p != None:
                p.join()
                del (X)
                print('mem: after del X', cpuStats())

            chunker_iter = chunker(chunk, batchsize//3)
            mini_chunk_id = 0
            for minichunk in chunker_iter:
                mini_chunk_id +=1
                print('procssing minichunk #{}/{}'.format(mini_chunk_id, file_read_chunk_id))

                predictors1 = categorical + new_features

                if config_scheme_to_use.add_hist_statis_fts:
                    predictors1 = predictors1 + hist_st

                print('converting minichunk {} with features {}: '.format(minichunk, predictors1))
                str_array = convert_features_to_text(minichunk, predictors1)
                print('converted to str array: ', str_array)


                labels = []
                weights = []
                if target in minichunk.columns:
                    labels = minichunk['is_attributed'].values
                    weights = np.multiply([1.0 if x == 1 else 0.2 for x in minichunk['is_attributed'].values],
                                          minichunk['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))
                del (minichunk)
                gc.collect()
                print('mem:', cpuStats())


                if p != None:
                    p.join()

                gc.collect()

                X = wb.transform(str_array)
                del (str_array)
                gc.collect()
                print('mem:', cpuStats())


                p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
                p.start()
            del chunk
            gc.collect()
        #for chunk in train_chunks:


        if p != None:
            p.join()


    del train_chunks
    del X
    del labels
    gc.collect()

    print('mem:', cpuStats())
    # convert features to text:

    val = pd.read_csv(path_train_sample if use_sample else path_train, dtype=dtypes,
                               header=0,
                               usecols=train_cols,
                               parse_dates=["click_time"])

    val, _, _ =  \
        prepare_data(val, 9, 2, config_scheme_to_use.val_filter, False,
                     start_time=config_scheme_to_use.val_start_time,
                     end_time=config_scheme_to_use.val_end_time, start_hist_time='2017-11-07 0:00:00')

    print('len val:', len(val))
    val = gen_categorical_features(val)

    val, new_features1, _ = generate_counting_history_features(val, None,
                                                           None,
                                                           False,
                                                           discretization=config_scheme_to_use.discretization,
                                                           discretization_bins=None)

    val = val.set_index('click_time').\
              ix[config_scheme_to_use.val_start_time:config_scheme_to_use.val_end_time].reset_index()

    if config_scheme_to_use.use_interactive_features:
        val = gen_iteractive_categorical_features(val)

    predictors1 = categorical + new_features1

    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st

    str_array = convert_features_to_text(val, predictors1)
    print('mem:', cpuStats())
    labels = val['is_attributed'].values

    del (val)
    gc.collect()

    print('mem:', cpuStats())

    X = wb.transform(str_array)

    evaluate_batch(clf, X, labels)

    del X
    del str_array
    del labels

    gc.collect()

    print('mem:', cpuStats())

    print('predicting...')
    p = None
    click_ids = []
    test_preds = []

    if config_scheme_to_use.predict_wordbatch:
        test_data, _ = gen_test_df(False, False, None)
        test_len= len(test_data)
    gc.collect()
    if config_scheme_to_use.use_interactive_features:
        test_data = gen_iteractive_categorical_features(test_data)

    if test_data is not None:
        batchsize = batchsize // 10
        with timer('predict wordbatch model...'):
            for chunk in chunker(test_data, batchsize):
                # convert features to text:

                predictors1 = categorical + new_features

                if config_scheme_to_use.add_hist_statis_fts:
                    predictors1 = predictors1 + hist_st

                str_array = convert_features_to_text(chunk, predictors1)
                print('mem:', cpuStats())
                click_ids += chunk['click_id'].tolist()

                if p != None:
                    test_preds += list(p.join())
                    del (X)
                gc.collect()

                X = wb.transform(str_array)
                del (str_array)
                p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
                p.start()
        if p != None:  test_preds += list(p.join())

        df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
        df_sub.to_csv(get_dated_filename("wordbatch_fm_ftrl.csv"), index=False)



def train_wordbatch_model(train, val, test_data, new_features):
    pick_hours = {4, 5, 10, 13, 14}
    batchsize = 10000000
    D = 2 ** 22

    if use_sample:
        batchsize = len(train) // 5

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
                          threads=4, use_avx=1, verbose=0)
        elif config_scheme_to_use.wordbatch_model == 'NN_ReLU_H1':
            clf = NN_ReLU_H1(alpha=0.05, D = D, verbose=9, e_noise=0.0, threads=4, inv_link="sigmoid")
        elif config_scheme_to_use.wordbatch_model == 'FTRL':
            clf = FTRL(alpha=0.05, beta=0.1, L1=0.0, L2=0.0, D=D, iters=3, threads=4, verbose=9)
        else:
            print('invalid wordbatch_model param:', config_scheme_to_use.wordbatch_model)
            exit(-1)

    target = 'is_attributed'
    #predictors1 = categorical + new_features

    #if config_scheme_to_use.add_hist_statis_fts:
    #    predictors1 = predictors1 + hist_st

    if config_scheme_to_use.use_interactive_features:
        train = gen_iteractive_categorical_features(train)
    p = None

    with timer('train wordbatch model...'):
        for chunk in chunker(train, batchsize):
            # convert features to text:


            predictors1 = categorical + new_features

            if config_scheme_to_use.add_hist_statis_fts:
                predictors1 = predictors1 + hist_st

            print('converting chunk {} with features {}: '.format(chunk, predictors1))
            str_array = convert_features_to_text(chunk, predictors1)
            print('converted to str array: ', str_array)
            del(chunk)
            print('mem:', cpuStats())

            labels = []
            weights = []
            if target in train.columns:
                labels = train['is_attributed'].values
                weights = np.multiply([1.0 if x == 1 else 0.2 for x in train['is_attributed'].values],
                                      train['hour'].apply(lambda x: 1.0 if x in pick_hours else 0.5))

            if p != None:
                del (X)
            gc.collect()

            X = wb.transform(str_array)
            del (str_array)

            if p != None:
                p.join()

            p = threading.Thread(target=fit_batch, args=(clf, X, labels, weights))
            p.start()

        if p != None:
            p.join()

    print('mem after train:', cpuStats())

    print('recycle train mem:')

    for ft in predictors1:
        del train[ft]

    del train
    gc.collect()
    print('mem after recycle train mem:', cpuStats())



    # convert features to text:
    if config_scheme_to_use.use_interactive_features:
        val = gen_iteractive_categorical_features(val)

    predictors1 = categorical + new_features

    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st

    str_array = convert_features_to_text(val, predictors1)
    print('mem:', cpuStats())
    labels = val['is_attributed'].values

    del (X)
    del (val)
    gc.collect()

    print('mem:', cpuStats())

    X = wb.transform(str_array)

    evaluate_batch(clf, X, labels)

    del X
    del str_array
    del labels

    gc.collect()

    print('mem:', cpuStats())
    
    return wb, clf
    
def predict_wordbatch(wb, clf):
    batchsize = 10000000

    print('predicting...')
    p = None
    click_ids = []
    test_preds = []

    if config_scheme_to_use.mock_test_with_val_data_to_test:
        path_test_to_use = path_train if not use_sample else path_train_sample
        test_cols_to_use = train_cols
    else:
        path_test_to_use = path_test if not use_sample else path_test_sample
        test_cols_to_use = test_cols

    batchsize = batchsize // 10
    print('predict wordbatch model... batch size:',batchsize)
    with timer('predict wordbatch model...'):

        chunks = pd.read_csv(path_test_to_use, dtype=dtypes, header=0,
                           compression='gzip' if read_path_with_hist else None,
                            chunksize=batchsize,
                           usecols=test_cols_to_use, parse_dates=["click_time"])  # .sample(1000)

        for chunk in chunks:

            chunk = gen_categorical_features(chunk)

            chunk, new_features, _ = generate_counting_history_features(chunk, None,
                                                                        None,
                                                                        False,
                                                                        discretization=config_scheme_to_use.discretization,
                                                                        discretization_bins=None)

            gc.collect()
            print('mem after hist ft:', cpuStats())

            if config_scheme_to_use.use_interactive_features:
                chunk = gen_iteractive_categorical_features(chunk)
                # convert features to text:
                predictors1 = categorical + new_features

                gc.collect()
                print('mem after iter ft:', cpuStats())
            if config_scheme_to_use.add_hist_statis_fts:
                predictors1 = predictors1 + hist_st


            str_array = convert_features_to_text(chunk, predictors1)
            click_ids += chunk['click_id'].tolist()
            del chunk
            gc.collect()
            print('mem after convert text:', cpuStats())

            if p != None:
                test_preds += list(p.join())
                del (X)
            gc.collect()

            X = wb.transform(str_array)
            del (str_array)
            p = ThreadWithReturnValue(target=predict_batch, args=(clf, X))
            p.start()
    if p != None:  test_preds += list(p.join())

    df_sub = pd.DataFrame({"click_id": click_ids, 'is_attributed': test_preds})
    df_sub.to_csv(get_dated_filename("wordbatch_fm_ftrl.csv"), index=False)


train_lgbm = False

def train_lgbm(train, val, new_features):
#if train_lgbm:

    # In[7]:
    target = 'is_attributed'

    predictors0 = ['device', 'app', 'os', 'channel', 'hour', # Starter Vars, Then new features below
                  'ip_day_hourcount','ipcount','ip_appcount', 'ip_app_oscount',
                  "ip_hour_channelcount", "ip_hour_oscount", "ip_hour_appcount","ip_hour_devicecount"]

    predictors1 = categorical + new_features

    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st
    #for ii in new_features:
    #    predictors1 = predictors1 + ii
    #print(predictors1)
    gc.collect()

    #train.fillna(value={x:-1 for x in new_features})

    print("Preparing the datasets for training...")



    predictors_to_train = [predictors1]

    for predictors in predictors_to_train:
        print('training with :', predictors)
        #print('training data: ', train[predictors].values)
        #print('validation data: ', val[predictors].values)
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
                         valid_names=['train','valid'],
                         evals_result=evals_results,
                         num_boost_round=1000,
                         early_stopping_rounds=30,
                         verbose_eval=50,
                         feval=None)

        #del train
        #del val
        #gc.collect()

        # Nick's Feature Importance Plot
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=[7,10])
        #lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
        #plt.title("Light GBM Feature Importance")
        #plt.savefig('feature_import.png')

        # Feature names:
        print('Feature names:', lgb_model.feature_name())
        # Feature importances:
        print('Feature importances:', list(lgb_model.feature_importance()))

        feature_imp = pd.DataFrame(lgb_model.feature_name(),list(lgb_model.feature_importance()))

        if persist_intermediate:
            print('dumping model')
            lgb_model.save_model(get_dated_filename('model.txt'))

        do_val_prediction = False
        val_prediction = None
        if do_val_prediction:
            print("Writing the val_prediction into a csv file...")
            #if persist_intermediate:

            print('gen val prediction')
            val_prediction = lgb_model.predict(val[predictors1], num_iteration=lgb_model.best_iteration)
            val['predict'] = val_prediction
            #pd.Series(val_prediction).to_csv(get_dated_filename("val_prediction.csv"), index=False)
            val.to_csv(get_dated_filename("val_prediction.csv"), index=False)


    return lgb_model, val_prediction


# In[ ]:

for_test = True

def gen_test_df(with_hist_profile = True, persist_fe_data = False,
            discretization_bins=None):
    #del train
    #del test
    #gc.collect()

    train = None

    #prepare test data:
    if with_hist_profile:
        train = pd.read_csv(path_train if not use_sample else path_train_sample, dtype=dtypes,
                            compression='gzip' if config_scheme_to_use.add_hist_statis_fts else None,
                            header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)
    #train = pd.read_csv(path_train if not use_sample else path_train_sample, dtype=dtypes,
    #        header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)

    if config_scheme_to_use.mock_test_with_val_data_to_test:
        path_test_to_use = path_train if not use_sample else path_train_sample
        test_cols_to_use = train_cols
    else:
        path_test_to_use = path_test if not use_sample else path_test_sample
        test_cols_to_use = test_cols


    test = pd.read_csv(path_test_to_use, dtype=dtypes, header=0,
                       compression='gzip' if read_path_with_hist else None,
                       usecols=test_cols_to_use,parse_dates=["click_time"])#.sample(1000)

    if config_scheme_to_use.seperate_hist_files:
        for ft in hist_st:
            csv_file = Path(path_train_hist +ft+ '.test.csv')
            csv_gzip_file = Path(path_train_hist + ft + '.test.csv.gzip')
            csv_bz2_file = Path(path_train_hist + ft + '.test.csv.bz2')
            ft_data = None
            if csv_file.is_file():
                ft_data = pd.read_csv(path_train_hist +ft+ '.test.csv', dtype={ft:'float32'},
                                    header=0, engine='c')  # .sample(1000)
            elif csv_gzip_file.is_file():
                ft_data = pd.read_csv(path_train_hist +ft+ '.test.csv.gzip', dtype={ft:'float32'},
                                    header=0, engine='c',compression='gzip') # .sample(1000)
            elif csv_bz2_file.is_file():
                ft_data = pd.read_csv(path_train_hist +ft+ '.test.csv.bz2', dtype={ft:'float32'},
                                    header=0, engine='c',compression='bz2') # .sample(1000)
                print(path_train_hist +ft+ '.test.csv.bz2' + ' loaded')
            else:
                print('{} not found!!!'.format(ft))
                exit(-1)

            test[ft] = ft_data
            del ft_data
            gc.collect()



    if train is not None:
        train=train.append(test)
    else:
        train = test

    del test
    gc.collect()
    train = gen_categorical_features(train)

    if not config_scheme_to_use.mock_test_with_val_data_to_test:
        train, train_ip_contains_training_day, train_ip_contains_training_day_attributed = \
            prepare_data(train, 10, 2, config_scheme_to_use.test_filter, with_hist_profile, for_test=True,
                         start_time = '2017-11-10 04:00:00',
                         end_time = '2017-11-10 23:59:59', start_hist_time = '2017-11-07 0:00:00'
        )
    else:
        train, train_ip_contains_training_day, train_ip_contains_training_day_attributed = \
            prepare_data(train, 10, 2, config_scheme_to_use.test_filter, with_hist_profile, for_test=True,
                         start_time = val_time_range_start,
                         end_time = val_time_range_end, start_hist_time = '2017-11-07 0:00:00'
        )

    train, new_features, _ = generate_counting_history_features(train, train_ip_contains_training_day,
                                                             train_ip_contains_training_day_attributed,
                                                             with_hist_profile,
                                                             discretization=config_scheme_to_use.discretization,
                                                             discretization_bins=discretization_bins)


    print('filtering testing data:')

    with timer('filtering testing data'):
        if not config_scheme_to_use.mock_test_with_val_data_to_test:
            train = train.set_index('click_time').ix['2017-11-10 04:00:00':'2017-11-10 15:00:00'].reset_index()
        else:
            train = train.set_index('click_time').ix[val_time_range_start:val_time_range_end].reset_index()

    if not config_scheme_to_use.mock_test_with_val_data_to_test:
        train['is_attributed'] = 0

    if persist_fe_data:
        predictors1 = categorical + new_features+ ['is_attributed']
        if config_scheme_to_use.add_hist_statis_fts:
            predictors1 = predictors1 + hist_st
        if config_scheme_to_use.mock_test_with_val_data_to_test:
            if 'click_id' not in train.columns:
                train['click_id'] = train.index

        train[predictors1 + ['click_id']].to_csv(get_dated_filename('test_fe.csv' + '.sample' if use_sample else 'test_fe.csv'), index=False)

    return train, new_features

# In[ ]:



def gen_ffm_data():
    train, val, new_features, discretization_bins_used = gen_train_df(False, True)
    train_len = len(train)
    val_len = len(val)
    gc.collect()
    test_len = 0
    if config_scheme_to_use.gen_ffm_test_data:
        test, _ = gen_test_df(False, True, discretization_bins_used)
        test_len= len(test)
    gc.collect()

    print('train({}) val({}) test({}) generated'.format(train_len, val_len,test_len))


    #train = train.append(val)
    #test = train.append(test)

    #del train
    #del val

    #gc.collect()

    #print(test)


if config_scheme_to_use.train:

    train, val, new_features, discretization_bins_used = gen_train_df(False)

    lgb_model, val_prediction = train_lgbm(train, val, new_features)
    # In[ ]:
    del train
    del val
    gc.collect()

if config_scheme_to_use.train_wordbatch:

    train, val, new_features, discretization_bins_used = gen_train_df(False)

    test = None

    gc.collect()

    wb, clf = train_wordbatch_model(train, val, None, new_features)
    print('mem done train:', cpuStats())

    
    del train
    del val
    
    #del clf 
    
    #del wb
    
    gc.collect()
    print('mem before predict:', cpuStats())

    if config_scheme_to_use.predict_wordbatch:
        predict_wordbatch(wb,clf)

if config_scheme_to_use.train_wordbatch_streaming:
    train_wordbatch_model_streaming()

to_submit = False

if to_submit:
    print('test data:', train)

    print('new features:', new_features)
    print("Preparing data for submission...")

    #submit = pd.read_csv(path_test, dtype='int', usecols=['click_id'])
    #print('submit test len:', len(submit))
    print("Predicting the submission data...")

    train['is_attributed'] = lgb_model.predict(train[predictors1], num_iteration=lgb_model.best_iteration)

    print("Writing the submission data into a csv file...")

    train[['click_id','is_attributed']].to_csv(get_dated_filename("submission_notebook.csv"),index=False)

    print("All done...")


predict_from_saved_model = False

to_predict = True

gc.collect()

if config_scheme_to_use.predict:
    test, new_test_features = gen_test_df(False, discretization_bins = discretization_bins_used)

    #print(test['ipcount_in_hist'].describe())

    print(test)

    if predict_from_saved_model:
        lgb_model = lgb.Booster(model_file='model.txt.01-04-2018_10:59:15')

    #submit = pd.read_csv(path_test if not use_sample else path_test_sample, dtype='int', usecols=['click_id'])

    predictors1 = categorical + new_test_features
    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st
    test['is_attributed'] = lgb_model.predict(test[predictors1], num_iteration=lgb_model.best_iteration)

    print("Writing the submission data into a csv file...") 

    test[['click_id','is_attributed']].to_csv(get_dated_filename("submission_notebook.csv"), index=False)

    print("All done...")


if config_scheme_to_use.ffm_data_gen:
    gen_ffm_data()

