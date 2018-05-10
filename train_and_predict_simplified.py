# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
import copy
from scipy import special as sp
from multiprocessing import Process

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib
from pprint import pprint
from pathlib import Path
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
import threading
from sklearn.metrics import roc_auc_score
import mmh3
import pickle
from contextlib import contextmanager
import os, psutil
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import sys
import gc
import csv
from pympler import muppy
from pympler import summary
import warnings
import hashlib
from multiprocessing import Pool, TimeoutError
from train_utils.constants import *
import train_utils.model_params
import train_utils.features_def
import random
from train_utils.config_schema import *


matplotlib.use('Agg')


def get_dated_filename(filename):
    #print('got file name: {}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S")))
    #return '{}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))
    if filename.rfind('.') != -1:
        id = filename.rfind('.')
        filename = filename[:id] + '_' + config_scheme_to_use.config_name + filename[id:]
    else:
        filename = filename + '_' + config_scheme_to_use.config_name
    print('got file name: {}'.format(filename))
    return filename



def cpuStats(pp = False):
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    gc.collect()
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)(Pdb) import objgraph

    return memoryUse

@contextmanager
def timer(name):
    t0 = time.time()
    print('start ',name)
    yield
    print('[{}] done in {} s'.format(name, time.time() - t0))
    print('mem after {}: {}'.format(name, cpuStats()))




dump_train_data = False

use_sample = False


if len(sys.argv) > 1 and sys.argv[1] == 'use_sample':
    use_sample = True

if use_sample:
    neg_sample_rate = 2
persist_intermediate = False
print_verbose = False

gen_test_input = True

read_path_with_hist = False

TRAIN_SAMPLE_DATA_LEN = 100001

try:
    os.mkdir( ft_cache_path )
except:
    None

try:
    os.mkdir('./new_libffm_output')
except:
    None



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
    if not config_scheme_to_use.use_hour_group is None:
        data['hour'] = data['hour'] // config_scheme_to_use.use_hour_group

    if not config_scheme_to_use.add_n_min_as_hour is None:
        data['hour'] = data.click_time.astype(int) // (10 ** 9 * 60* config_scheme_to_use.add_n_min_as_hour)

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
            data[interactive_features_name] = (data[interactive_features_name].apply(mmh3.hash) % 1000000).astype('uint32')
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


def get_recursive_alpha_beta(sumi, count, global_alpha = 20.0, global_beta = 10000.0):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        if global_alpha == 0:
            print("!!!! zero global alpha")
            exit(-1)

        def getalpha(sumi, count, alpha0, beta0):
            try:
                return alpha0 * (sp.psi(sumi + alpha0) - sp.psi(alpha0)) / \
                    (sp.psi(count + alpha0 + beta0) - sp.psi(alpha0 + beta0))
            except:
                print('warining: {} {} {} {}'.format(sumi, count, alpha0, beta0))
                return global_alpha

        def getbeta(sumi, count, alpha0, beta0):
            try:
                return beta0 * (sp.psi(count - sumi + beta0) - sp.psi(beta0)) / \
                    (sp.psi(count + alpha0 + beta0) - sp.psi(alpha0 + beta0))
            except:
                print('warining: {} {} {} {}'.format(sumi, count, alpha0, beta0))
                return global_beta

        alpha = global_alpha
        beta = global_beta

        for i in range(1000):
            alpha0 = alpha
            beta0 = beta
            alpha = getalpha(sumi, count, alpha0, beta0)
            beta = getbeta(sumi, count, alpha0, beta0)

            if alpha == 0:
                break
            #print('alpha:', alpha)


        if (alpha is None) or (beta is None):
            print("!!!!!!!!!!!Alpha = 0, NAN in calculating alpha!!!!!!!!!!!!!!!!!: alpha:{}, beta:{}".format(alpha, beta))
            exit(-1)
            return global_alpha, global_beta
        else:
            return alpha, beta


def add_statistic_feature(group_by_cols, training, qcut_count=config_scheme_to_use.qcut, #0, #0.98,
                          discretization=0, discretization_bins=None,
                          log_discretization=False,
                          op='count',
                          use_ft_cache = False,
                          ft_cache_prefix = '',
                          only_ft_cache = False,
                          astype=None,
                          sample_indice = None,
                          df_before_sample = None):
    #print('\n\n------running add_statistic_feature in pid [{}]-------\nonly_ft_cache:{}\n'.format(
    #    os.getpid(), only_ft_cache))
    print('[PID {}] adding: {}, {}'.format(os.getpid(), str(group_by_cols), op))


    input_len = len(training)
    feature_name_added = '_'.join(group_by_cols) + op

    ft_cache_file_name = config_scheme_to_use.use_ft_cache_from + "_" + ft_cache_prefix + '_' + feature_name_added
    ft_cache_file_name = ft_cache_file_name + '_sample' if use_sample else ft_cache_file_name
    ft_cache_file_name = ft_cache_file_name + '.pickle.bz2'

    loaded_from_cache = False

    print('checking {}, exist:{}'.format(ft_cache_path + ft_cache_file_name, os.path.exists((ft_cache_path + ft_cache_file_name))))
    if use_ft_cache and os.path.exists ((ft_cache_path + ft_cache_file_name)):
        if only_ft_cache:
            print('cache only, cache exists, return.')
            return
        print('[PID {}] cache file exist, loading: {}'.format(os.getpid(), ft_cache_path + ft_cache_file_name))

        try:
            with timer('read pickle ft cache file:' + ft_cache_path + ft_cache_file_name):
                ft_cache_data = pd.read_pickle(ft_cache_path + ft_cache_file_name,
                                        #dtype='float32',
                                        #header=0, engine='c',
                                        compression='bz2')
            if sample_indice is not None:
                #print(ft_cache_data)
                #print(sample_indice)
                ft_cache_data = ft_cache_data.loc[sample_indice]
                print('sample indice applied, len after sample of ft cache:',len(ft_cache_data))
            #print('SAMPLE:{}-{}'.format(feature_name_added, ft_cache_data.sample(5, random_state=88)))

        except:
            print('[PID {}] err loading: {}'.format(os.getpid(), ft_cache_path + ft_cache_file_name))
            raise ValueError('[PID {}] err loading: {}'.format(os.getpid(), ft_cache_path + ft_cache_file_name))

        #print('before merge', training.columns)
        #training = training.join(ft_cache_data)#training.merge(ft_cache_data, how='left', left_index=True, right_index=True)
        #print(training.head())
        training[feature_name_added] = ft_cache_data
        print('[PID {}] loaded {} from file {}, count:({})'.format(
            os.getpid(), feature_name_added, ft_cache_path + ft_cache_file_name, training[feature_name_added].count()))
        loaded_from_cache=True
        del ft_cache_data
        gc.collect()
        return training, [feature_name_added], None
    if use_ft_cache and only_ft_cache:
        #print('only gen cache, use df_before_sample..., use ',df_before_sample)
        training = df_before_sample


    counting_col = group_by_cols[len(group_by_cols) - 1]
    group_by_cols = group_by_cols[0:len(group_by_cols) - 1]
    features_added = []
    discretization_bins_used = {}
    #print('[PID {}] count with group by: {}'.format(os.getpid(), str(group_by_cols)))

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
            #print('next click added:', training[feature_name_added].describe())
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
            #print('next click added:', training[feature_name_added].describe())
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
        #print(training[feature_name_added].describe())
    else:
        tempstr = 'training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]]'
        if len(op) > 2 and op[0:2] == 'qt':
            temp1 = eval('{}.quantile({})'.format(tempstr, float(op[2:])) )
            #temp1 = eval('{}.apply(lambda x: np.percentile(x.sample(min(len(x), 100)), q={}))'.format(tempstr, float(op[2:])) )
        else:
            temp1 = eval(tempstr + '.' + op + '()')

        n_chans = temp1.reset_index().rename(columns={counting_col: feature_name_added})
        #training.sort_index(inplace=True)

        #print('nan count: ', n_chans[n_chans[feature_name_added].isnull()])
        print('[PID {}] nan count: {}'.format(os.getpid(), n_chans[feature_name_added].isnull().sum()))
        training = training.merge(n_chans, on=group_by_cols,# if len(group_by_cols) >1 else group_by_cols[0],
                                  how='left')
        del n_chans
        #training.sort_index(inplace=True)



        if config_scheme_to_use.normalization and op == 'count':
            if hasattr(add_statistic_feature, first_df_count):
                rate = add_statistic_feature.first_df_count / input_len
            else:
                add_statistic_feature.first_df_count = input_len
                rate = 1.0
            training[feature_name_added] = (training[feature_name_added] * rate).astype('uint32')


    gc.collect()

    # auto type cast:
    auto_type_cast_ops_list = ['count', 'nunique', 'cumcount']
    if config_scheme_to_use.auto_type_cast and op in auto_type_cast_ops_list:
        if training[feature_name_added].max() <= 65535:
            training[feature_name_added] = training[feature_name_added].astype('uint16')
        elif  training[feature_name_added].max() <= 2 ** 32 -1:
            training[feature_name_added] = training[feature_name_added].astype('uint32')
        elif  training[feature_name_added].max() <= 255:
            training[feature_name_added] = training[feature_name_added].astype('uint8')

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

    print('[PID {}] added features:'.format(os.getpid(), features_added))
    if print_verbose:
        print(training[feature_name_added].describe())
    #print('nan count: ', training[training[feature_name_added].isnull()])

    print('[PID {}] columns after added: {}'.format(os.getpid(), training.columns.values))

    #for test:
    #print('SAMPLE:{}-{}'.format(feature_name_added, training[feature_name_added].sample(5, random_state=88)))

    if use_ft_cache and not loaded_from_cache:

        try:
            os.mkdir(ft_cache_path)
            print('created dir', ft_cache_path)
        except:
            #print(ft_cache_path + ' already exist.')
            None


        pd.DataFrame(training[feature_name_added]).to_pickle(
            ft_cache_path + ft_cache_file_name,compression='bz2')
        #print('[PID {}] saved {} to file {} sum:({})'.format(
        #    os.getpid(), feature_name_added, ft_cache_path + ft_cache_file_name, training[feature_name_added].sample(6, random_state=98)))
        print('[PID {}] saved {} to file {} count:({})'.format(
            os.getpid(), feature_name_added, ft_cache_path + ft_cache_file_name, training[feature_name_added].count()))

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
            elif not config_scheme_to_use.use_hourly_alpha_beta or 'hour' not in group_by_cols:
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
                add_statistic_feature.train_cvr_cache[feature_name_added] = temp_sum.copy(True)
                del temp_sum
                gc.collect()
            else:
                if not hasattr(add_statistic_feature, 'hourly_alpha_beta'):
                    add_statistic_feature.hourly_alpha_beta = train[['hour', 'is_attributed']].\
                        groupby(by=['hour'])[['is_attributed']].sum()
                    add_statistic_feature.hourly_alpha_beta['count'] = train[['hour', 'is_attributed']].\
                        groupby(by=['hour'])[['is_attributed']].count()['is_attributed']
                    add_statistic_feature.hourly_alpha_beta['alpha'] = add_statistic_feature.hourly_alpha_beta.apply(
                        lambda x: get_recursive_alpha_beta(x['is_attributed'], x['count'],
                                                           add_statistic_feature.alpha,
                                                           add_statistic_feature.beta)[0], axis=1
                    )
                    add_statistic_feature.hourly_alpha_beta['beta'] = add_statistic_feature.hourly_alpha_beta.apply(
                        lambda x: get_recursive_alpha_beta(x['is_attributed'], x['count'],
                                                           add_statistic_feature.alpha,
                                                           add_statistic_feature.beta)[1], axis=1
                    )
                    del add_statistic_feature.hourly_alpha_beta['is_attributed']
                    del add_statistic_feature.hourly_alpha_beta['count']
                    add_statistic_feature.hourly_alpha_beta = add_statistic_feature.hourly_alpha_beta.reset_index()
                    print('debug: hourly alapha beta:')
                    print(add_statistic_feature.hourly_alpha_beta.to_string())
                    print(add_statistic_feature.hourly_alpha_beta.\
                          apply(lambda x: x['alpha']/(x['alpha']+x['beta']), axis=1).to_string())
                    print('hourly mean:')
                    print(train[['hour', 'is_attributed']].\
                        groupby(by=['hour'])[['is_attributed']].mean().to_string())
                    print('hourly sum:')
                    print(train[['hour', 'is_attributed']].\
                        groupby(by=['hour'])[['is_attributed']].sum().to_string())
                    print('hourly count:')
                    print(train[['hour', 'is_attributed']].\
                        groupby(by=['hour'])[['is_attributed']].count().to_string())

                temp_count = train[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[
                    ['is_attributed']].count().rename(columns={'is_attributed':'count'})
                temp_count['sum'] = train[group_by_cols + ['is_attributed']].\
                    groupby(by=group_by_cols)[['is_attributed']].sum()['is_attributed']

                temp_count = temp_count.reset_index().merge(add_statistic_feature.hourly_alpha_beta,
                                                            on='hour',how='left')
                temp_count[feature_name_added] = temp_count.apply(lambda x: (x['sum'] + x['alpha'])/ \
                                                                            (x['count'] + x['alpha'] + x['beta']), axis=1)
                del temp_count['alpha']
                del temp_count['beta']
                del temp_count['sum']
                del temp_count['count']
                #del temp_count['is_attributed']

                gc.collect()
                add_statistic_feature.train_cvr_cache[feature_name_added] = temp_count.copy(True)
                del temp_count
                gc.collect()

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
                                       only_ft_cache = False,
                                       val_start = None,
                                       val_end = None,
                                       only_scvr_ft = 3,
                                       checksum = 'checksum',
                                       sample_indice = None,
                                       df_before_sample = None):
    #print('tail all before:',data.tail())
    #print('DEBUG:', data.query('ip == 123517 & day==7 &channel ==328'))
    #print('DEBUG:', data.loc[6,:])

    if val_start is not None:
        print('clear val(data[{}:{}]) is_attributed before gen sta fts and restore after'.format(val_start, val_end))
        print('sum of val target col before clear:',
              data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')].sum())

        #print(data.iloc[val_start:val_end,data.columns.values.tolist().index('is_attributed')].head())
        data['is_attributed_backup'] = data['is_attributed']

        data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')] = 0
        print('sum of val target col:', data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')].sum())
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
            #print('add ad_val fts:', data['ad_val'].describe())
            new_features.append('ad_val')

            del ad_val_bst
            gc.collect()

    discretization_bins_used = None
    i = -1
    to_run_in_pool = []
    for add_feature in add_features_list:
        i += 1
        if only_scvr_ft == 1 and add_feature['op'] != 'smoothcvr':
            continue
        elif only_scvr_ft == 2 and add_feature['op'] == 'smoothcvr':
            continue

        to_run_in_pool.append(add_feature)

    print('TORUN:',to_run_in_pool)

    with timer('adding feature caches first round: {}/{}, {}'.format(i, len(add_features_list), str(add_feature))):
        p_list = []
        #data.sort_index(inplace=True)
        if use_ft_cache:
            for add_feature in to_run_in_pool:

                p = Process(target=add_statistic_feature, args = (
                                        add_feature['group'], #group_by_cols
                                        data, #training
                                        0,#qcut
                                        discretization, #discretization
                                        discretization_bins,#discretization_bins
                                        config_scheme_to_use.log_discretization, #log_discretization
                                        add_feature['op'],
                                        use_ft_cache,
                                        checksum,#ft_cache_prefix
                                        True, #only_ft_cache,
                                        add_feature['astype'] if 'astype' in add_feature else None, #astype
                                        None,
                                        df_before_sample
                                    ))
                p.start()
                p_list.append(p)

                while len(p_list) >= process_poll_size:
                    p_list[0].join(1)
                    for p in p_list:
                        if not p.is_alive():
                            p_list.remove(p)
                            break

        #data.sort_index(inplace=True)

        print('second round load cache files:')
        for add_feature in to_run_in_pool:
            data, features_added,discretization_bins_used_current_feature = \
                add_statistic_feature(
                                    list(add_feature['group']), #group_by_cols
                                    data, #training
                                    0,#qcut
                                    discretization, #discretization
                                    discretization_bins,#discretization_bins
                                    config_scheme_to_use.log_discretization, #log_discretization
                                    add_feature['op'],
                                    use_ft_cache,
                                    checksum,#ft_cache_prefix
                                    only_ft_cache,
                                    add_feature['astype'] if 'astype' in add_feature else None, #astype,
                                    sample_indice
                                )
            #print('returned from pool: data-{} features_added-{} discretization_bins_used_current_feature-{}'.format(
            #    res[0], features_added, discretization_bins_used
            #))

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

    if val_start is not None:
        print('restore val is_attributed')
        data['is_attributed']= data['is_attributed_backup']
        del data['is_attributed_backup']
        print('sum of val target col:',
              data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')].sum())


    return data, new_features, discretization_bins_used




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
def get_stacking_val_df():
    val = pd.read_csv(path_train_sample if use_sample else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.lgbm_stacking_val_from) \
                            if not use_sample and config_scheme_to_use.lgbm_stacking_val_from is not None else None,
                        nrows=config_scheme_to_use.lgbm_stacking_val_to - config_scheme_to_use.lgbm_stacking_val_from \
                            if not use_sample and config_scheme_to_use.lgbm_stacking_val_from is not None else None,
                        parse_dates=["click_time"])

    print('mem after loaded stacking val data:', cpuStats())

    gc.collect()
    return val

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

def get_checksum_from_df(df):
    #m = hashlib.md5()
    #df.apply(lambda x: m.update(x.to_string().encode('utf8')), axis = 1)
    #m.update(df.to_string().encode('utf8'))
    #ret = mmh3.hash_from_buffer(df['click_time'].astype(str).get_values().copy(order='C'), signed = False)
    ret = mmh3.hash_from_buffer(df['click_time'].sample(frac=0.1, random_state=88).to_string(), signed = False)
    gc.collect()
    return str(ret)

def train_and_predict(com_fts_list, use_ft_cache = False, only_cache=False,
                                         use_base_data_cache=False, gen_fts = False, load_test_supplement = False):
    with timer('load combined data df'):
        combined_df, train_len, val_len, test_len = get_combined_df(config_scheme_to_use.new_predict or gen_fts,
                                                                    load_test_supplement = load_test_supplement)
        print('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
        combined_df.reset_index(drop=True,inplace=True)

    with timer('checksum data'):
        checksum = get_checksum_from_df(combined_df)
        print('md5 checksum of whole data set:', checksum)



    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)

    neg_sample_indice = None
    if config_scheme_to_use.use_neg_sample:
        print('neg sample 1/200(1:2 pos:neg) after checksum...')
        np.random.seed(888)
        #neg_sample_indice = random.sample(range(len(combined_df)),len(combined_df) // 200)
        neg_sample_indice = (np.random.randint(0, neg_sample_rate, len(combined_df), np.uint8) == 0) \
                            | (combined_df['is_attributed'] == 1) \
                            | np.concatenate((np.zeros(train_len, np.bool_) ,np.ones(val_len + test_len,np.bool_)))
        #print('neg sample indice: len:{}, tail:{}'.format(len(neg_sample_indice), neg_sample_indice[-10:]))
        #print('neg sample indice: len:{}, head:{}'.format(len(neg_sample_indice), neg_sample_indice[:10]))

        combined_df_before_sample = combined_df.copy(True)
        #print('len before sampel:',len(combined_df_before_sample))
        #print('len before sampel:',len(combined_df))

        combined_df = combined_df[neg_sample_indice]
        #print('len after sampel:',len(combined_df_before_sample))
        #print('len after sampel:',len(combined_df))

    with timer('gen statistical hist features'):
        combined_df, new_features, discretization_bins_used = \
        generate_counting_history_features(combined_df,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list,
                                           val_start = train_len,
                                           val_end = train_len + val_len,
                                           only_scvr_ft = 2,
                                           checksum = checksum,
                                           sample_indice = neg_sample_indice,
                                           df_before_sample = combined_df_before_sample)

    #test
    #print('sample:',combined_df.sample(10,random_state=888).to_string())
    #print('poslen: {}, neglen:{}'.format(len(combined_df.query('is_attributed == 1')),
    #                                     len(combined_df.query('is_attributed == 0'))))


    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]


    if config_scheme_to_use.train_smoothcvr_cache_from is not None:
        gen_smoothcvr_cache(config_scheme_to_use.train_smoothcvr_cache_from, config_scheme_to_use.train_smoothcvr_cache_to)

        train, new_features_cvr, _ = \
            generate_counting_history_features(train,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache=use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list,
                                           only_scvr_ft=1)
        new_features += new_features_cvr

        clear_smoothcvr_cache()
        gen_smoothcvr_cache(config_scheme_to_use.val_smoothcvr_cache_from, config_scheme_to_use.val_smoothcvr_cache_to)
        val, _, _ = \
            generate_counting_history_features(val,
                                           discretization=config_scheme_to_use.discretization,
                                           use_ft_cache=use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list,
                                           only_scvr_ft=1)

    if config_scheme_to_use.dump_train_data:
        train.to_csv("train_ft_dump.csv.bz2", compression='bz2',index=False)
        val.to_csv("val_ft_dump.csv.bz2", compression='bz2',index=False)

    if dump_train_data and config_scheme_to_use.new_predict:
        test = combined_df[train_len + val_len:]
        test[categorical + new_features].to_csv("test_ft_dump.csv.bz2", compression='bz2',index=False)

    with timer('train lgbm model...'):
        lgb_model, val_prediction, predictors, importances, val_auc = train_lgbm(train, val, new_features, False)

    if config_scheme_to_use.new_predict:
        with timer('predict test data:'):
            if not dump_train_data: # because for dump case, it'll be set above
                test = combined_df[train_len + val_len: train_len+val_len+test_len]

            if config_scheme_to_use.train_smoothcvr_cache_from is not None:
                clear_smoothcvr_cache()
                gen_smoothcvr_cache(config_scheme_to_use.test_smoothcvr_cache_from,
                                    config_scheme_to_use.test_smoothcvr_cache_to)
                test, _, _ = \
                    generate_counting_history_features(test,
                                                       discretization=config_scheme_to_use.discretization,
                                                       use_ft_cache=use_ft_cache,
                                                       ft_cache_prefix='joint',
                                                       add_features_list=com_fts_list,
                                                       only_scvr_ft=1)

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
                random_state=666,
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

    if config_scheme_to_use.run_theme == 'train_and_predict':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache)
    elif config_scheme_to_use.run_theme == 'train_and_predict_with_test_supplement':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list,
                          use_ft_cache=config_scheme_to_use.use_ft_cache,
                          load_test_supplement=True)
    elif config_scheme_to_use.run_theme == 'lgbm_params_search':
        print('add features list: ')
        pprint(config_scheme_to_use.add_features_list)
        lgbm_params_search(config_scheme_to_use.add_features_list)
    else:
        print("nothing to run... exit")


with timer('run_model...'):
    run_model()

print('run_model done')
