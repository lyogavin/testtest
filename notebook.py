
# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from dateutil import parser
import matplotlib
matplotlib.use('Agg')

def get_dated_filename(filename):
    #return '{}.{}'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))
    return filename


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle

print('test log 43 gen ffm data with qcut 1000')
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
persist_intermediate = False

gen_test_input = True

path = '../input/' 
path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

categorical = ['app', 'device', 'os', 'channel', 'hour']

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


shuffle_sample_filter = {'filter_type': 'sample', 'sample_count': 6}
shuffle_sample_filter_1_to_10 = {'filter_type': 'sample', 'sample_count': 1}
shuffle_sample_filter_1_to_10k = {'filter_type': 'sample', 'sample_count': 1}

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
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
               train_end_time = train_time_range_start,
               val_start_time = val_time_range_start,
               val_end_time = val_time_range_end,
               gen_ffm_test_data = False):
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


train_predict_config = ConfigScheme(True, True, False)
train_config = ConfigScheme(False, True, False,
                            train_filter=shuffle_sample_filter,
                            train_start_time = val_time_range_start,
                            train_end_time=val_time_range_end,
                            val_start_time=train_time_range_start,
                            val_end_time=train_time_range_end)
ffm_data_config = ConfigScheme(False, False, True,shuffle_sample_filter_1_to_10,
                               shuffle_sample_filter_1_to_10,shuffle_sample_filter_1_to_10k,  discretization=100,
                               gen_ffm_test_data=True)
ffm_data_config_train = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=100,
                                     )


ffm_data_config_mock_test = ConfigScheme(False, False, True,shuffle_sample_filter_1_to_10,
                                         shuffle_sample_filter_1_to_10,shuffle_sample_filter_1_to_10k,
                                         discretization=100,
                                         mock_test_with_val_data_to_test=True)


train_predict_new_lgbm_params_config = ConfigScheme(True, True, False, lgbm_params=new_lgbm_params)


train_predict_filter_app_12_config = ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter2,
                                                  val_filter=field_sample_filter_app_filter2,
                                                  test_filter=field_sample_filter_app_filter2)

train_predict_filter_app_18_14_config = ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter3,
                                                     val_filter=field_sample_filter_app_filter3,
                                                     test_filter=field_sample_filter_app_filter3)

train_predict_filter_app_8_11_config = ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter4,
                                                    val_filter=field_sample_filter_app_filter4,
                                                    test_filter=field_sample_filter_app_filter4)

train_predict_filter_app_8_11_new_lgbm_params_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter4,
                 val_filter=field_sample_filter_app_filter4,
                 test_filter=field_sample_filter_app_filter4,
                 lgbm_params=new_lgbm_params
                 )
train_predict_filter_app_12_new_lgbm_params_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter2,
                 val_filter=field_sample_filter_app_filter2,
                 test_filter=field_sample_filter_app_filter2,
                 lgbm_params=new_lgbm_params
                 )

config_scheme_to_use = ffm_data_config_train

# In[2]:

def gen_categorical_features(data):
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    print("Creating new time features in train: 'hour' and 'day'...")
    data['hour'] = data["click_time"].dt.hour.astype('uint8')
    data['day'] = data["click_time"].dt.day.astype('uint8')

    add_hh_feature = False
    if add_hh_feature:
        data['in_test_hh'] = (   3
                               - 2*data['hour'].isin(  most_freq_hours_in_test_data )
                               - 1*data['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
        categorical.append('in_test_hh')
    return data


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

        len_train = len(data)
        print('len after filter %s: %s', (filter_config['filter_type'], len_train))

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

        train = data.set_index('click_time').ix[start_time:end_time].reset_index()
    print('training data len:', len(train))
    print('train unique ips:', len(train['ip'].unique()))
    
    return train, \
           train_ip_contains_training_day, train_ip_contains_training_day_attributed


def add_statistic_feature(group_by_cols, training, training_hist, training_hist_attribution,
                          with_hist, counting_col='channel', cast_type=True, qcut_count=0.98,
                          discretization=0, discretization_bins = None):
    features_added = []
    feature_name_added = '_'.join(group_by_cols) + 'count'
    discretization_bins_used = {}
    print('count ip with group by:', group_by_cols)
    n_chans = training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]] \
        .count().reset_index().rename(columns={counting_col: feature_name_added})
    training = training.merge(n_chans, on=group_by_cols, how='left')
    del n_chans
    gc.collect()
    training[feature_name_added] = training[feature_name_added].astype('uint16')
    if qcut_count != 0 and  discretization==0:
        #print('before qcut', feature_name_added, training[feature_name_added].describe())
        quantile_cut = training[feature_name_added].quantile(qcut_count)
        training[feature_name_added] = training[feature_name_added].apply(
            lambda x: x if x < quantile_cut else 65535).astype('uint16')
        #print('after qcut', feature_name_added, training[feature_name_added].describe())
    if discretization != 0:
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

    if with_hist:
        print('count ip with group by in hist data:', group_by_cols)
        feature_name_added = '_'.join(group_by_cols) + "count_in_hist"
        n_chans = training_hist[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]] \
            .count().reset_index().rename(columns={counting_col: feature_name_added})
        training = training.merge(n_chans, on=group_by_cols, how='left')
        del n_chans
        gc.collect()
        print('count ip attribution with group by in hist data:', group_by_cols)
        feature_name_added1 = '_'.join(group_by_cols) + "count_attribution_in_hist"
        n_chans = training_hist_attribution[group_by_cols + [counting_col]] \
            .groupby(by=group_by_cols)[[counting_col]] \
            .count().reset_index().rename(columns={counting_col: feature_name_added1})
        training = training.merge(n_chans, on=group_by_cols, how='left')
        del n_chans
        gc.collect()

        feature_name_added2 = '_'.join(group_by_cols) + "count_attribution_rate_in_hist"
        training[feature_name_added2] = \
            training[feature_name_added1] / training[feature_name_added] * 1000.0

        if qcut_count != 0:
            print('before qcut', feature_name_added, training[feature_name_added].describe())
            quantile_cut = training[feature_name_added].quantile(qcut_count)
            training[feature_name_added] = training[feature_name_added].apply(lambda x: x if x < quantile_cut else -1)
            print('after qcut', feature_name_added, training[feature_name_added].describe())

        if cast_type:
            training[feature_name_added] = training[feature_name_added].fillna(0).astype('uint16')
        if discretization != 0:
            print('before qcut', feature_name_added, training[feature_name_added].describe())
            training[feature_name_added] = pd.qcut(training[feature_name_added], discretization, labels=False,
                                                   duplicates='drop').fillna(0).astype('uint16')
            print('after qcut', feature_name_added, training[feature_name_added].describe())



        if qcut_count != 0:
            print('before qcut', feature_name_added1, training[feature_name_added1].describe())
            quantile_cut = training[feature_name_added1].quantile(qcut_count)
            training[feature_name_added1] = training[feature_name_added1].apply(lambda x: x if x < quantile_cut else -1)
            print('after qcut', feature_name_added1, training[feature_name_added1].describe())

        if cast_type:
            training[feature_name_added1] = training[feature_name_added1].fillna(0).astype('uint16')
            #training = training.astype({feature_name_added1:'uint16'})
            print(training[feature_name_added1])
        if discretization != 0:
            print('before qcut', feature_name_added1, training[feature_name_added1].describe())
            training[feature_name_added1] = pd.qcut(training[feature_name_added1], discretization, labels=False,
                                                    duplicates='drop').fillna(0).astype('uint16')
            print('after qcut', feature_name_added1, training[feature_name_added1].describe())
        # training[feature_name_added1] = training[feature_name_added1].astype('uint16')


        if cast_type:
            training[feature_name_added2] = training[feature_name_added2].fillna(0).astype('uint16')

        features_added.append(feature_name_added)
        features_added.append(feature_name_added1)
        features_added.append(feature_name_added2)

    print('added features:', features_added)

    return training, features_added, discretization_bins_used

def generate_counting_history_features(data, history, history_attribution,
                                       with_hist_profile = True, remove_hist_profile_count=0,
                                       discretization=0, discretization_bins=None):

    print('discretization bins to use:', discretization_bins)
        
    new_features = []

    add_features_list = [
        {'group':['ip','day','hour'], 'with_hist': False, 'counting_col':'channel'},
        {'group':['ip','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        {'group':['ip','os', 'app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        #{'group':['ip'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        {'group':['ip','hour','channel'], 'with_hist': with_hist_profile, 'counting_col':'os'},
        {'group':['ip','hour','os'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        {'group':['ip','hour','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
        {'group':['channel','app'], 'with_hist': with_hist_profile, 'counting_col':'os'},
        {'group':['channel','os'], 'with_hist': with_hist_profile, 'counting_col':'app'},
        {'group':['channel','app','os'], 'with_hist': with_hist_profile, 'counting_col':'device'},
        {'group':['os','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'}
        ]

    new_features_data = []
    discretization_bins_used = None

    for add_feature in add_features_list:
        new_data, features_added, discretization_bins_used_current_feature = add_statistic_feature(add_feature['group'],
                                                     data[add_feature['group'] + [add_feature['counting_col']]],
                                                     history, history_attribution, add_feature['with_hist'],
                                                     counting_col=add_feature['counting_col'],
                                                     discretization=discretization,
                                                     discretization_bins=discretization_bins)
        new_features = new_features + features_added
        if discretization_bins_used_current_feature is not None:
            if discretization_bins_used is None:
                discretization_bins_used = {}
            discretization_bins_used = \
                dict(list(discretization_bins_used.items()) + list(discretization_bins_used_current_feature.items()))
        new_features_data.append({'data':new_data[features_added], 'features':features_added})
        gc.collect()

    for new_data in new_features_data:
        for feature in new_data['features']:
            data[feature] = new_data['data'][feature]

    if remove_hist_profile_count != 0:
        data = data.query('ipcount_in_hist > {}'.format(remove_hist_profile_count))

    if discretization_bins is None:
        print('discretization bins used:',discretization_bins_used )
    else:
        print('discretizatoin bins passed in params, so no discretization_bins_used returned')
    return data, new_features, discretization_bins_used

#test['hour'] = test["click_time"].dt.hour.astype('uint8')
#test['day'] = test["click_time"].dt.day.astype('uint8')



def gen_train_df(with_hist_profile = True, persist_fe_data = False):
    train = pd.read_csv(path_train_sample if use_sample else path_train, dtype=dtypes,
            header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)


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

    print('train data:', train)
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

    #sys.exit(0)
    return train, val, new_features, discretization_bins_used


train_lgbm = False

def train_lgbm(train, val, new_features):
#if train_lgbm:

    # In[7]:
    target = 'is_attributed'

    predictors0 = ['device', 'app', 'os', 'channel', 'hour', # Starter Vars, Then new features below
                  'ip_day_hourcount','ipcount','ip_appcount', 'ip_app_oscount',
                  "ip_hour_channelcount", "ip_hour_oscount", "ip_hour_appcount","ip_hour_devicecount"]

    predictors1 = categorical + new_features
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
            usecols=test_cols_to_use,parse_dates=["click_time"])#.sample(1000)
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

    if not config_scheme_to_use.mock_test_with_val_data_to_test:
        train = train.set_index('click_time').ix['2017-11-10 04:00:00':'2017-11-10 15:00:00'].reset_index()
    else:
        train = train.set_index('click_time').ix[val_time_range_start:val_time_range_end].reset_index()

    if not config_scheme_to_use.mock_test_with_val_data_to_test:
        train['is_attributed'] = 0

    if persist_fe_data:
        predictors1 = categorical + new_features+ ['is_attributed']
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
    test['is_attributed'] = lgb_model.predict(test[predictors1], num_iteration=lgb_model.best_iteration)

    print("Writing the submission data into a csv file...")

    test[['click_id','is_attributed']].to_csv(get_dated_filename("submission_notebook.csv"), index=False)

    print("All done...")


if config_scheme_to_use.ffm_data_gen:
    gen_ffm_data()
