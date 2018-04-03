
# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from dateutil import parser

def get_dated_filename(filename):
    return '{}.{}_{}'.format(filename, time.strftime("%d-%m-%Y"), time.strftime("%X"))


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pickle

print('test log 16')
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

use_sample = False
persist_intermediate = False

gen_test_input = True

path = '../input/' 
path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

categorical = ['app', 'device', 'os', 'channel', 'hour']

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


# In[2]:

def prepare_data(data, training_day, profile_days, sample_count=1,
                 with_hist_profile=True, only_for_ip_with_hist = False, for_test = False,
                 start_time=None, end_time=None,
                 start_hist_time=None):
    if sample_count != 1:
        #sample 1/4 of the data:
        data = data.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        len_train = len(data)
        print('len after sample:', len_train)

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
                          with_hist, counting_col='channel', cast_type=True, qcut_count=0.98, discretization=0):
    features_added = []
    feature_name_added = '_'.join(group_by_cols) + 'count'
    print('count ip with group by:', group_by_cols)
    n_chans = training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]] \
        .count().reset_index().rename(columns={counting_col: feature_name_added})
    training = training.merge(n_chans, on=group_by_cols, how='left')
    del n_chans
    gc.collect()
    training[feature_name_added] = training[feature_name_added].astype('uint16')
    if qcut_count != 0:
        print('before qcut', feature_name_added, training[feature_name_added].describe())
        quantile_cut = training[feature_name_added].quantile(qcut_count)
        training[feature_name_added] = training[feature_name_added].apply(
            lambda x: x if x < quantile_cut else 65535).astype('uint16')
        print('after qcut', feature_name_added, training[feature_name_added].describe())
    if discretization != 0:
        print('before qcut', feature_name_added, training[feature_name_added].describe())
        training[feature_name_added] = pd.qcut(training[feature_name_added], discretization, labels=False,
                                               duplicates='drop').fillna(0).astype('uint16')
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

    return training, features_added

def generate_counting_history_features(data, history, history_attribution,
                                       with_hist_profile = True, remove_hist_profile_count=0):
        
    new_features = []

    # Count by IP,DAY,HOUR
    print('a given IP address within each hour...')
    data, features_added = add_statistic_feature(['ip','day','hour'], data, history, history_attribution, False)
    new_features = new_features + features_added
    gc.collect()

    # Count by IP and APP
    data, features_added = add_statistic_feature(['ip','app'], data, history, history_attribution, with_hist_profile)
    new_features = new_features + features_added
    
    # Count by IP and channel
    data, features_added = add_statistic_feature(['ip','channel'], data, history, history_attribution, with_hist_profile, counting_col='os')
    new_features = new_features + features_added
    
    # Count by IP and channel app
    data, features_added = add_statistic_feature(['ip','channel', 'app'], data, history, history_attribution, with_hist_profile, counting_col='os')
    new_features = new_features + features_added
    
    data, features_added  = add_statistic_feature(['ip','app','os'], data, history, history_attribution, with_hist_profile)
    new_features = new_features + features_added

    #######
    # Count by IP
    data, features_added  = add_statistic_feature(['ip'], data, history, history_attribution, with_hist_profile)
    new_features = new_features + features_added

    #######
    #tested channle, app, os count feature, worse in test 8.
    #######
    # Count by Channel
    #data, features_added  = add_statistic_feature(['channel'], data, history, history_attribution, with_hist_profile, counting_col='os')
    #new_features = new_features + features_added
    #######
    # Count by APP
    #data, features_added  = add_statistic_feature(['app'], data, history, history_attribution, with_hist_profile)
    #new_features = new_features + features_added
    #######
    # Count by OS
    #data, features_added  = add_statistic_feature(['os'], data, history, history_attribution, with_hist_profile)
    #new_features = new_features + features_added


    # Count by IP HOUR CHANNEL                                               
    data, features_added  = add_statistic_feature(['ip','hour','channel'],
                                                  data, history, history_attribution, with_hist_profile, counting_col='os')
    new_features = new_features + features_added

    # Count by IP HOUR Device
    data, features_added  = add_statistic_feature(['ip','hour','os'],
                                                  data, history, history_attribution, with_hist_profile)
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['ip','hour','app'],
                                                  data, history, history_attribution, with_hist_profile, counting_col='os')
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['channel','app'],
                                                  data, history, history_attribution, with_hist_profile, counting_col='os')
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['channel','os'],
                                                  data, history, history_attribution, with_hist_profile, counting_col='app')
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['channel','app','os'],
                                                  data, history, history_attribution, with_hist_profile, counting_col='device')
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['app','os'],
                                                  data, history, history_attribution, with_hist_profile)
    new_features = new_features + features_added


    if remove_hist_profile_count != 0:
        data = data.query('ipcount_in_hist > {}'.format(remove_hist_profile_count))

    return data, new_features

#test['hour'] = test["click_time"].dt.hour.astype('uint8')
#test['day'] = test["click_time"].dt.day.astype('uint8')



def gen_train_df(with_hist_profile = True):
    train = pd.read_csv(path_train_sample if use_sample else path_train, dtype=dtypes,
            header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)


    len_train = len(train)
    print('The initial size of the train set is', len_train)
    print('Binding the training and test set together...')


    print("Creating new time features in train: 'hour' and 'day'...")
    train['hour'] = train["click_time"].dt.hour.astype('uint8')
    train['day'] = train["click_time"].dt.day.astype('uint8')

    train_data, train_ip_contains_training_day, train_ip_contains_training_day_attributed =  \
        prepare_data(train, 8, 2, 6, with_hist_profile, start_time='2017-11-06 00:00:00',
                     end_time='2017-11-08 15:00:00', start_hist_time='2017-11-06 0:00:00')

    train_data, new_features = generate_counting_history_features(train_data, train_ip_contains_training_day,
                                                                  train_ip_contains_training_day_attributed,
                                                                  with_hist_profile)

    train_data = train_data.set_index('click_time').ix['2017-11-08 04:00:00':'2017-11-08 15:00:00'].reset_index()

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
        prepare_data(train, 9, 2, 6, with_hist_profile, start_time='2017-11-07 00:00:00',
                     end_time='2017-11-09 15:00:00', start_hist_time='2017-11-07 0:00:00')

    print('len val:', len(val))
    val, new_features1 = generate_counting_history_features(val, train_ip_contains_training_day,
                                                           train_ip_contains_training_day_attributed,
                                                           with_hist_profile)

    val = val.set_index('click_time').ix['2017-11-09 04:00:00':'2017-11-09 15:00:00'].reset_index()
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

    if persist_intermediate:
        if use_sample:
            train.to_csv(get_dated_filename('training_sample.csv'), index=False)
            val.to_csv(get_dated_filename('val_sample.csv'), index=False)
        else:
            train.to_csv(get_dated_filename('training.csv'), index=False)
            val.to_csv(get_dated_filename('val.csv'), index=False)

        print('save dtypes')

        y = {k: str(v) for k, v in train.dtypes.to_dict().items()}
        print(y)
        del y['click_time']
        #del y['Unnamed: 0']
        pickle.dump(y,open('output_dtypes.pickle','wb'))

    #sys.exit(0)
    return train, val, new_features


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

    params = {
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
        'early_stopping_round':20,
        #'is_unbalance': True,
        'scale_pos_weight':99.7
        }

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

        lgb_model = lgb.train(params,
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
        lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
        plt.title("Light GBM Feature Importance")
        plt.savefig('feature_import.png')

        # Feature names:
        print('Feature names:', lgb_model.feature_name())
        # Feature importances:
        print('Feature importances:', list(lgb_model.feature_importance()))

        feature_imp = pd.DataFrame(lgb_model.feature_name(),list(lgb_model.feature_importance()))

        if persist_intermediate:
            print('dumping model')
            lgb_model.save_model(get_dated_filename('model.txt'))

        print("Writing the val_prediction into a csv file...")
        if persist_intermediate:

            print('gen val prediction')
            val_prediction = lgb_model.predict(val[predictors1], num_iteration=lgb_model.best_iteration)
            pd.Series(val_prediction).to_csv(get_dated_filename("val_prediction.csv"), index=False)

    return lgb_model


# In[ ]:

for_test = True

def gen_test_df(with_hist_profile = True):
    #del train
    #del test
    #gc.collect()

    #prepare test data:
    if with_hist_profile:
        train = pd.read_csv(path_train if not use_sample else path_train_sample, dtype=dtypes,
                header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)
    test = pd.read_csv(path_test if not use_sample else path_test_sample, dtype=dtypes, header=0,
            usecols=test_cols,parse_dates=["click_time"])#.sample(1000)
    if with_hist_profile:
        train=train.append(test)
    else:
        train = test
    del test
    gc.collect()
    print("Creating new time features in train: 'hour' and 'day'...")
    train['hour'] = train["click_time"].dt.hour.astype('uint8')
    train['day'] = train["click_time"].dt.day.astype('uint8')
    
    train, train_ip_contains_training_day, train_ip_contains_training_day_attributed = \
        prepare_data(train, 10, 2, 1, with_hist_profile, for_test=True,
                     start_time = '2017-11-10 00:00:00',
                     end_time = '2017-11-10 23:59:59', start_hist_time = '2017-11-07 0:00:00'
    )

    train, new_features = generate_counting_history_features(train, train_ip_contains_training_day, 
                                                             train_ip_contains_training_day_attributed,
                                                             with_hist_profile)

    train['is_attributed'] = 0

    if persist_intermediate:
        train.to_csv(get_dated_filename('to_submit.csv' + '.sample' if use_sample else 'to_submit.csv'), index=False)

    return train, new_features

# In[ ]:



def gen_ffm_data():
    train, val, new_features = gen_train_df(False)
    train_len = len(train)
    val_len = len(val)
    gc.collect()
    test, _ = gen_test_df(False)
    test_len= len(test)
    gc.collect()

    print('train({}) val({}) test({}) generated'.format(train_len, val_len,test_len))


    train = train.append(val)
    test = train.append(test)

    del train
    del val

    gc.collect()

    print(test)


train_model = True

if train_model:

    train, val, new_features = gen_train_df(False)

    lgb_model = train_lgbm(train, val, new_features)
    # In[ ]:
    del train
    del val
    gc.collect()

to_submit = False

if to_submit:
    print('test data:', train)

    print('new features:', new_features)
    print("Preparing data for submission...")

    submit = pd.read_csv(path_test, dtype='int', usecols=['click_id'])
    print('submit test len:', len(submit))
    print("Predicting the submission data...")
    submit['is_attributed'] = lgb_model.predict(train[predictors1], num_iteration=lgb_model.best_iteration)

    print("Writing the submission data into a csv file...")

    submit.to_csv(get_dated_filename("submission_notebook.csv"),index=False)

    print("All done...")

#gen_ffm_data()

predict_from_saved_model = False

to_predict = False

if to_predict:
    test, new_test_features = gen_test_df(False)

    #print(test['ipcount_in_hist'].describe())

    print(test)

    if predict_from_saved_model:
        lgb_model = lgb.Booster(model_file='model.txt.01-04-2018_10:59:15')

    submit = pd.read_csv(path_test if not use_sample else path_test_sample, dtype='int', usecols=['click_id'])

    predictors1 = categorical + new_test_features
    submit['is_attributed'] = lgb_model.predict(test[predictors1], num_iteration=lgb_model.best_iteration)

    print("Writing the submission data into a csv file...")

    submit.to_csv(get_dated_filename("submission_notebook.csv"), index=False)

    print("All done...")
