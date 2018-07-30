# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sys
import copy
import lightgbm as lgb

from scipy import special as sp
from multiprocessing import Process
import multiprocessing as mp
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import matplotlib
from pprint import pprint
from pprint import pformat
from pathlib import Path
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
import threading
from sklearn.metrics import roc_auc_score
import mmh3
import pickle
import os, psutil
from sklearn.model_selection import train_test_split
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
import itertools
from train_utils.config_schema import *
from train_utils.utils import *
import logging
from optparse import OptionParser
logger = getLogger()


parser = OptionParser()
parser.add_option("-c", "--config", dest="config_theme_name", action="store",
                  help="config theme name to use", metavar="CONFIGTHEME", default=None)
parser.add_option("-u", "--unittest",
                  action="store_true", dest="unittest",
                  help="enable unit test mode", default=False)
(options, args) = parser.parse_args()

if options.config_theme_name is None:
    parser.error("-c/--config required, see -h or --help")

config_scheme_to_use = use_config_scheme(options.config_theme_name)

if options.unittest:
    logger.info('running unit test mode....')

matplotlib.use('Agg')




def get_dated_filename(filename):
    #logger.debug('got file name: {}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S")))
    #return '{}_{}.csv'.format(filename, time.strftime("%d-%m-%Y_%H-%M-%S"))
    if filename.rfind('.') != -1:
        id = filename.rfind('.')
        filename = filename[:id] + '_' + config_scheme_to_use.config_name + filename[id:]
    else:
        filename = filename + '_' + config_scheme_to_use.config_name
    logger.debug('got file name: {}'.format(filename))
    return filename




dump_train_data = False

if options.unittest:
    neg_sample_rate = 2
    logger.setLevel(logging.DEBUG)

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
    logger.debug("Creating new time features in train: 'hour' and 'day'...")
    data['hour'] = data["click_time"].dt.hour.astype('uint8')
    data['day'] = data["click_time"].dt.day.astype('uint8')
    if config_scheme_to_use.add_second_ft:
        data['second'] = data["click_time"].dt.second.astype('int8')
        categorical.append('second')

    add_hh_feature = False
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
                logger.debug('added iterative fts: %s',interactive_features_name )
    return data


def post_statistics_features(data):
    logger.debug('ip_in_test_frequent_channel_is_attributed_count describe:')
    if config_scheme_to_use.add_in_test_frequent_dimensions is not None and \
        'channel' in config_scheme_to_use.add_in_test_frequent_dimensions:
        logger.debug(data['ip_in_test_frequent_channel_is_attributedcount'].describe())
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
            logger.warn("!!!! zero global alpha")
            exit(-1)

        def getalpha(sumi, count, alpha0, beta0):
            try:
                return alpha0 * (sp.psi(sumi + alpha0) - sp.psi(alpha0)) / \
                    (sp.psi(count + alpha0 + beta0) - sp.psi(alpha0 + beta0))
            except:
                logger.debug('warining: {} {} {} {}'.format(sumi, count, alpha0, beta0))
                return global_alpha

        def getbeta(sumi, count, alpha0, beta0):
            try:
                return beta0 * (sp.psi(count - sumi + beta0) - sp.psi(beta0)) / \
                    (sp.psi(count + alpha0 + beta0) - sp.psi(alpha0 + beta0))
            except:
                logger.debug('warining: {} {} {} {}'.format(sumi, count, alpha0, beta0))
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
            #logger.debug('alpha:', alpha)


        if (alpha is None) or (beta is None):
            logger.warn("!!!!!!!!!!!Alpha = 0, NAN in calculating alpha!!!!!!!!!!!!!!!!!: alpha:{}, beta:{}".format(alpha, beta))
            exit(-1)
            return global_alpha, global_beta
        else:
            return alpha, beta

def load_ft_cache_file(group_by_cols, op, ft_cache_prefix, sample_indice):
    feature_name_added = '_'.join(group_by_cols) + op

    with timer('preloading ft file for:' + feature_name_added, logging.DEBUG):

        ft_cache_file_name = config_scheme_to_use.use_ft_cache_from + "_" + ft_cache_prefix + '_' + feature_name_added
        ft_cache_file_name = ft_cache_file_name + '_sample' if options.unittest else ft_cache_file_name
        ft_cache_file_name = ft_cache_file_name + '.pickle.bz2'
        try:
            ft_cache_data = pd.read_pickle(ft_cache_path + ft_cache_file_name,
                                            #dtype='float32',
                                            #header=0, engine='c',
                                            compression='bz2')
            if sample_indice is not None:
                ft_cache_data = ft_cache_data.loc[sample_indice]
            #logger.debug('tail of preload cache file(%s): %s', ft_cache_path + ft_cache_file_name,
            #             ft_cache_data.tail().to_string())

        except:
            logger.info("Unexpected error:", sys.exc_info()[0])
            logger.info('removing %s.', ft_cache_path + ft_cache_file_name)
            os.remove(ft_cache_path + ft_cache_file_name)
            ft_cache_data = None
            raise ValueError('%s loading error'.format(ft_cache_path + ft_cache_file_name))

    return ft_cache_data

def add_statistic_feature(group_by_cols,
                          training,
                          log_discretization=False,
                          op='count',
                          use_ft_cache = False,
                          ft_cache_prefix = '',
                          only_ft_cache = False,
                          astype=None,
                          sample_indice = None,
                          df_before_sample = None,
                          preload_df = None):
    #logger.debug('\n\n------running add_statistic_feature in pid [{}]-------\nonly_ft_cache:{}\n'.format(
    #    os.getpid(), only_ft_cache))
    logger.debug('[PID {}] adding: {}, {}'.format(os.getpid(), str(group_by_cols), op))

    debug = False
    input_len = len(training)
    feature_name_added = '_'.join(group_by_cols) + op

    ft_cache_file_name = config_scheme_to_use.use_ft_cache_from + "_" + ft_cache_prefix + '_' + feature_name_added
    ft_cache_file_name = ft_cache_file_name + '_sample' if options.unittest else ft_cache_file_name
    ft_cache_file_name = ft_cache_file_name + '.pickle.bz2'

    loaded_from_cache = False

    #logger.debug('checking {}, exist:{}'.format(ft_cache_path + ft_cache_file_name, os.path.exists((ft_cache_path + ft_cache_file_name))))

    if use_ft_cache and os.path.exists ((ft_cache_path + ft_cache_file_name)):
        if only_ft_cache:
            logger.debug('cache only, cache exists, return.')
            return
        logger.debug('[PID {}] cache file exist, loading: {}'.format(os.getpid(), ft_cache_path + ft_cache_file_name))


        if preload_df is not None:
            try:
                with timer('read pickle ft cache file:' + ft_cache_path + ft_cache_file_name):
                    if preload_df is not None:
                        ft_cache_data = preload_df
                    else:
                        ft_cache_data = pd.read_pickle(ft_cache_path + ft_cache_file_name,
                                            #dtype='float32',
                                            #header=0, engine='c',
                                            compression='bz2')
                        if sample_indice is not None:
                            #logger.debug(ft_cache_data)
                            #logger.debug(sample_indice)
                            ft_cache_data = ft_cache_data.loc[sample_indice]
                            logger.debug('sample indice applied, len after sample of ft cache: %d',len(ft_cache_data))
                        #logger.debug('SAMPLE:{}-{}'.format(feature_name_added, ft_cache_data.sample(5, random_state=88)))

            except:
                logger.debug('[PID {}] err loading: {}'.format(os.getpid(), ft_cache_path + ft_cache_file_name))
                raise ValueError('[PID {}] err loading: {}'.format(os.getpid(), ft_cache_path + ft_cache_file_name))

            #logger.debug('before merge', training.columns)
            #training = training.join(ft_cache_data)#training.merge(ft_cache_data, how='left', left_index=True, right_index=True)
            training[feature_name_added] = ft_cache_data
            #logger.debug('ft loaded: %s' + training[feature_name_added].tail().to_string())

            logger.debug('[PID {}] loaded {} from file {}, count:({})'.format(
                os.getpid(), feature_name_added, ft_cache_path + ft_cache_file_name, training[feature_name_added].count()))
            loaded_from_cache=True
            del ft_cache_data
            gc.collect()
            return training, [feature_name_added]
    if use_ft_cache and only_ft_cache:
        #logger.debug('only gen cache, use df_before_sample..., use ',df_before_sample)
        training = df_before_sample


    counting_col = group_by_cols[len(group_by_cols) - 1]
    group_by_cols = group_by_cols[0:len(group_by_cols) - 1]
    features_added = []
    discretization_bins_used = {}
    #logger.debug('[PID {}] count with group by: {}'.format(os.getpid(), str(group_by_cols)))

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
                logger.debug('data: %s',training[0:10])
                logger.debug('debug str %s',joint_col[0:10])
                logger.debug('debug str %s', (training['ip'].astype(str) + "_" + training['app'].astype(str) + "_" + training['device'].astype(str) \
            + "_" + training['os'].astype(str)+ "_" + training['channel'].astype(str))[0:10])

            training['category'] = joint_col.apply(mmh3.hash) % D
            if debug:
                logger.debug('debug category %s',training['category'][0:10])

            del joint_col
            gc.collect()

            n = 3
            click_buffers = []
            for i in range(n):
                click_buffers.append(np.full(D, 3000000000, dtype=np.uint32))
            training['epochtime'] = training['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, echtime in zip(reversed(training['category'].values),
                                      reversed(training['epochtime'].values)):
                # shift values in buffers queue and append new value from the tail
                for i in range(n - 1):
                    click_buffers[i][category] = click_buffers[i+1][category]
                next_clicks.append(click_buffers[0][category] - echtime)
                click_buffers[n-1][category] = echtime
            del (click_buffers)
            training[feature_name_added] = list(reversed(next_clicks))

            #training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            #features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            #logger.debug('next click added:', training[feature_name_added].describe())
    elif op=='previousnclick':
        with timer("Adding previous n click times"):
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
                logger.debug('data: %s',training[0:10])
                logger.debug('debug str %s',joint_col[0:10])
                logger.debug('debug str %s', (training['ip'].astype(str) + "_" + training['app'].astype(str) + "_" + training['device'].astype(str) \
            + "_" + training['os'].astype(str)+ "_" + training['channel'].astype(str))[0:10])

            training['category'] = joint_col.apply(mmh3.hash) % D
            if debug:
                logger.debug('debug category %s',training['category'][0:10])

            del joint_col
            gc.collect()

            n = 3
            click_buffers = []
            for i in range(n):
                click_buffers.append(np.full(D, 0, dtype=np.uint32))
            training['epochtime'] = training['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, echtime in zip(training['category'].values, training['epochtime'].values):
                # shift values in buffers queue and append new value from the tail
                for i in range(n - 1):
                    click_buffers[i][category] = click_buffers[i+1][category]
                next_clicks.append(click_buffers[0][category] - echtime)
                click_buffers[n-1][category] = echtime
            del (click_buffers)
            training[feature_name_added] = list(reversed(next_clicks))

            #training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            #features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            #logger.debug('next click added:', training[feature_name_added].describe())
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

            debug = True
            if debug:
                logger.debug('data: %s',training[0:10])
                logger.debug('debug str %s',joint_col[0:10])
                logger.debug('debug str %s', (training['ip'].astype(str) + "_" + training['app'].astype(str) + "_" + training['device'].astype(str) \
            + "_" + training['os'].astype(str)+ "_" + training['channel'].astype(str))[0:10])

            try:
                training['category'] = joint_col.apply(mmh3.hash) % D
            except:
                logger.debug(joint_col.describe())
                raise AttributeError('None type')

            if debug:
                logger.debug('debug category %s',training['category'][0:10])

            del joint_col
            gc.collect()
            click_buffer = np.full(D, 3000000000, dtype=np.uint32) #3000000000
            training['epochtime'] = training['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, echtime in zip(reversed(training['category'].values),
                                      reversed(training['epochtime'].values)):
                next_clicks.append(click_buffer[category] - echtime)
                click_buffer[category] = echtime
            del (click_buffer)
            training[feature_name_added] = list(reversed(next_clicks))

            #training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            #features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            #logger.debug('next click added:', training[feature_name_added].describe())
    elif op=='previousclick':
        with timer("Adding previous click times"):
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
                logger.debug('data: %s',training[0:10])
                logger.debug('debug str %s',joint_col[0:10])
                logger.debug('debug str %s', (training['ip'].astype(str) + "_" + training['app'].astype(str) + "_" + training['device'].astype(str) \
            + "_" + training['os'].astype(str)+ "_" + training['channel'].astype(str))[0:10])

            try:
                training['category'] = joint_col.apply(mmh3.hash) % D
            except:
                logger.debug(joint_col.describe())
                raise AttributeError('None type')

            if debug:
                logger.debug('debug category %s',training['category'][0:10])

            del joint_col
            gc.collect()
            click_buffer = np.full(D, 0, dtype=np.uint32)
            training['epochtime'] = training['click_time'].astype(np.int64) // 10 ** 9
            next_clicks = []
            for category, echtime in zip(training['category'].values,
                                      training['epochtime'].values):
                next_clicks.append(click_buffer[category] - echtime)
                click_buffer[category] = echtime
            del (click_buffer)
            training[feature_name_added] = list(reversed(next_clicks))

            #training[feature_name_added+'_shift'] = pd.DataFrame(list(reversed(next_clicks))).shift(+1).values
            #features_added.append(feature_name_added+'_shift')

            training.drop('epochtime', inplace=True, axis=1)
            training.drop('category', inplace=True, axis=1)

            #if print_verbose:
            #logger.debug('next click added:', training[feature_name_added].describe())
    elif op=='smoothcvr':
        with timer('gen cvr grouping cache:'):
            if 'day' not in group_by_cols:
                group_by_cols.append('day')

            if not hasattr(add_statistic_feature, 'alpha'):
                add_statistic_feature.alpha, add_statistic_feature.beta = \
                    get_recursive_alpha_beta(training['is_attributed'].sum(), training['is_attributed'].count())
                logger.debug('total alpha/beta: {}/{}'.format(add_statistic_feature.alpha, add_statistic_feature.beta))
                logger.debug('total cvr:{}, alpha/(alpha+beta):{}'.format(training['is_attributed'].mean(),
                    add_statistic_feature.alpha/ (add_statistic_feature.alpha + add_statistic_feature.beta)))

            # shiftting day forward 1 day
            training['day'] = training['day'] + 1

            temp_count = training[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[
                ['is_attributed']].count()
            temp_sum = training[group_by_cols + ['is_attributed']].groupby(by=group_by_cols)[['is_attributed']].sum()
            temp_sum[feature_name_added] = (temp_sum['is_attributed'] + add_statistic_feature.alpha) / \
                                           (temp_count[
                                                'is_attributed'] + add_statistic_feature.alpha + add_statistic_feature.beta).astype(
                                               'float16')
            del temp_count
            del temp_sum['is_attributed']
            training['day'] = training['day'] - 1
            gc.collect()
        training = training.merge(temp_sum.reset_index(), on=group_by_cols if len(group_by_cols) >1 else group_by_cols[0],
                                  how='left')
        if not hasattr(add_statistic_feature, 'global_cvr'):
            add_statistic_feature.global_cvr = training['is_attributed'].mean()
        training[feature_name_added] = training[feature_name_added].fillna(add_statistic_feature.global_cvr)
        logger.debug('smooth cvr: {} gened'.format(feature_name_added))
        #logger.debug(training[feature_name_added].describe())
    else:
        tempstr = 'training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]]'
        if len(op) > 2 and op[0:2] == 'qt':
            temp1 = eval('{}.quantile({})'.format(tempstr, float(op[2:])) )
            #temp1 = eval('{}.apply(lambda x: np.percentile(x.sample(min(len(x), 100)), q={}))'.format(tempstr, float(op[2:])) )
        else:
            temp1 = eval(tempstr + '.' + op + '()')

        n_chans = temp1.reset_index().rename(columns={counting_col: feature_name_added})

        if feature_name_added == 'ip_app_os_hourvar':
            logger.debug('dump inter var {}: {}'.format(feature_name_added,n_chans.query('ip_app_os_hourvar != 0').tail(500).to_string()))
        #training.sort_index(inplace=True)

        #logger.debug('nan count: ', n_chans[n_chans[feature_name_added].isnull()])
        #logger.debug('[PID {}] nan count: {}'.format(os.getpid(), n_chans[feature_name_added].isnull().sum()))
        training = training.merge(n_chans, on=group_by_cols,# if len(group_by_cols) >1 else group_by_cols[0],
                                  how='left', copy=False)
        del n_chans
        #training.sort_index(inplace=True)


    gc.collect()

    # auto type cast:
    auto_type_cast_ops_list = ['count', 'nunique', 'cumcount']
    if config_scheme_to_use.auto_type_cast and op in auto_type_cast_ops_list:
        if training[feature_name_added].max() <= 65535:
            training[feature_name_added] = training[feature_name_added].astype('uint16', copy=False)
        elif  training[feature_name_added].max() <= 2 ** 32 -1:
            training[feature_name_added] = training[feature_name_added].astype('uint32', copy=False)
        elif  training[feature_name_added].max() <= 255:
            training[feature_name_added] = training[feature_name_added].astype('uint8', copy=False)



    if log_discretization:
        if training[feature_name_added].min() < 0:
            logger.debug('!!!! invalid time in {}, fix it.....'.format(feature_name_added))
            training[feature_name_added] = training[feature_name_added].apply(lambda x: np.max([0, x]))
        training[feature_name_added] = np.log2(1 + training[feature_name_added].values).astype(int)
        logger.debug('log dicretizing feature: %s', feature_name_added)

    features_added.append(feature_name_added)
    for ft in features_added:
        if astype is not None:
            training[ft] = training[ft].astype(astype)

    logger.debug('[PID {}] added features:'.format(os.getpid(), features_added))
    if print_verbose:
        logger.debug(training[feature_name_added].describe())
    #logger.debug('nan count: ', training[training[feature_name_added].isnull()])

    logger.debug('[PID {}] columns after added: {}'.format(os.getpid(), training.columns.values))

    #for test:
    #if op == 'smoothcvr':
    #    logger.debug('SAMPLE:{}-{}'.format(feature_name_added, training[feature_name_added].sample(5, random_state=88)))

    if use_ft_cache and not loaded_from_cache:

        try:
            os.mkdir(ft_cache_path)
            logger.debug('created dir %s', ft_cache_path)
        except:
            #logger.debug(ft_cache_path + ' already exist.')
            None


        pd.DataFrame(training[feature_name_added]).to_pickle(
            ft_cache_path + ft_cache_file_name,compression='bz2')
        #logger.debug('[PID {}] saved {} to file {} sum:({})'.format(
        #    os.getpid(), feature_name_added, ft_cache_path + ft_cache_file_name, training[feature_name_added].sample(6, random_state=98)))
        logger.debug('[PID {}] saved {} to file {} count:({})'.format(
            os.getpid(), feature_name_added, ft_cache_path + ft_cache_file_name, training[feature_name_added].count()))

        if only_ft_cache:
            del training[feature_name_added]
            del features_added[-1]
            gc.collect()

    return training, features_added



def generate_counting_history_features(data,
                                       add_features_list=None,
                                       use_ft_cache = False,
                                       ft_cache_prefix = '',
                                       only_ft_cache = False,
                                       val_start = None,
                                       val_end = None,
                                       checksum = 'checksum',
                                       sample_indice = None,
                                       df_before_sample = None):

    if val_start is not None:
        logger.debug('clear val(data[{}:{}]) is_attributed before gen sta fts and restore after'.format(val_start, val_end))
        logger.debug('sum of val target col before clear: %s',
              data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')].sum())

        #logger.debug(data.iloc[val_start:val_end,data.columns.values.tolist().index('is_attributed')].head())
        data['is_attributed_backup'] = data['is_attributed']

        data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')] = 0
        logger.debug('sum of val target col: %d', data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')].sum())

    new_features = set()

    if config_scheme_to_use.add_10min_ft:
        with timer('adding 10min feature:'):
            data['id_10min'] = data.click_time.astype(int) // (10 ** 9 * 600 *5)
            categorical.append('id_10min')

    i = -1
    to_run_in_pool = []
    for add_feature in add_features_list:
        feature_name_added = '_'.join(add_feature['group']) + add_feature['op']

        if feature_name_added in data.columns:
            logger.debug('{} already in data, skip...'.format(add_feature))
            new_features = new_features.union([feature_name_added])

            continue

        i += 1

        to_run_in_pool.append(add_feature)

    logger.debug('TORUN: %s',pformat(to_run_in_pool))

    with timer('adding feature caches first round: {}/{}, {}'.format(i, len(add_features_list), str(add_feature)),
               logging.DEBUG):
        p_list = []
        #data.sort_index(inplace=True)
        if use_ft_cache:
            for add_feature in to_run_in_pool:

                p = Process(target=add_statistic_feature, args = (
                                        add_feature['group'], #group_by_cols
                                        data, #training
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

                alive_count = 100
                while alive_count >= process_poll_size:
                    time.sleep(1)
                    alive_count = 0
                    for p in p_list:
                        alive_count += p.is_alive()

            for p in p_list:
                p.join()
        #data.sort_index(inplace=True)

        logger.debug('second round load cache files:')
        logger.debug('TORUN: %s', pformat(to_run_in_pool))

        multithread_load_ft = True
        preload_dfs = {}
        pdict = {}
        if multithread_load_ft and use_ft_cache:
            for add_feature in to_run_in_pool:
                p = ThreadWithReturnValue(target = load_ft_cache_file,
                                          args=(list(add_feature['group']),
                                                add_feature['op'],
                                                checksum,
                                                sample_indice))
                pdict[str(add_feature)] = p
                p.start()

                alive_count = 100
                while alive_count >= process_poll_size:
                    time.sleep(1)
                    alive_count = 0
                    for pp in pdict:
                        alive_count += pdict[pp].is_alive()

            logger.debug('TORUN: %s', pformat(to_run_in_pool))

            for add_feature in pdict:
                preload_dfs[str(add_feature)] = pdict[str(add_feature)].join()

                if preload_dfs[str(add_feature)] is not None and len(preload_dfs[str(add_feature)]) != len(data):
                    del preload_dfs[str(add_feature)]
                #logger.info('preload_dfs keys: %s', pformat(preload_dfs.keys()))
                #assert (preload_dfs[str(add_feature)] is not None), 'cache len loaded should be the same as sampled df'
                #assert (len(preload_dfs[str(add_feature)]) == len(data)), 'cache len loaded should be the same as sampled df'

        logger.debug('TORUN: %s', pformat(to_run_in_pool))

        for add_feature in to_run_in_pool:
            data, features_added = \
                add_statistic_feature(
                                    list(add_feature['group']), #group_by_cols
                                    data, #training
                                    config_scheme_to_use.log_discretization, #log_discretization
                                    add_feature['op'],
                                    use_ft_cache,
                                    checksum,#ft_cache_prefix
                                    only_ft_cache,
                                    add_feature['astype'] if 'astype' in add_feature else None, #astype,
                                    sample_indice,
                                    preload_df= preload_dfs[str(add_feature)] if str(add_feature) in preload_dfs else None
                                )
            if str(add_feature) in preload_dfs:
                del preload_dfs[str(add_feature)]
            gc.collect()
            #logger.debug('returned from pool: data-{} features_added-{} discretization_bins_used_current_feature-{}'.format(
            #    res[0], features_added, discretization_bins_used
            #))

            new_features = new_features.union(features_added)
            gc.collect()


    logger.debug('\n\n\n-------------\n{} DEBUG CVR:\n-------------\n\n\n'.format(ft_cache_prefix))
    if 'hour_is_attributedsmoothcvr' in data.columns and \
            'hour_is_attributedmean' in data.columns and \
            'hour_is_attributedcount' in data.columns:
        logger.debug('gened hour_is_attributedsmoothcvr:')
        logger.debug(data[['hour_is_attributedsmoothcvr', 'hour_is_attributedmean',
                    'hour_is_attributedcount','hour']].sample(20).to_string())


    if 'app_device_os_ip_is_attributedsmoothcvr' in data.columns and \
        'app_device_os_ip_is_attributedmean' in data.columns and \
        'app_device_os_ip_is_attributedcount' in  data.columns:
        logger.debug('gened ip_app_device_os_is_attributedsmoothcvr:')
        logger.debug(data[['app_device_os_ip_is_attributedsmoothcvr',
              'app_device_os_ip_is_attributedmean','app_device_os_ip_is_attributedcount']].sample(20).to_string())


    logger.debug('\n\n\n-------------\nDEBUG CVR DONE\n-------------\n\n\n')


    data = post_statistics_features(data)


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

    logger.debug('data dtypes: %s',data.dtypes)

    if val_start is not None:
        logger.debug('restore val is_attributed')
        data['is_attributed']= data['is_attributed_backup']
        del data['is_attributed_backup']
        logger.debug('sum of val target col: %s',
              data.iloc[val_start:val_end, data.columns.values.tolist().index('is_attributed')].sum())


    return data, new_features




def train_lgbm(train, val, new_features, do_val_prediction=False):

    target = 'is_attributed'

    iter_num_set =  -1 if config_scheme_to_use.lgbm_params['early_stopping_round'] is not None \
        else config_scheme_to_use.lgbm_params['num_boost_round']

    predictors1 =  list(new_features.union(categorical))

    if config_scheme_to_use.add_hist_statis_fts:
        predictors1 = predictors1 + hist_st

    gc.collect()

    logger.debug("Preparing the datasets for training...")

    predictors_to_train = [predictors1]

    for predictors in predictors_to_train:
        # important to make sure the order is the same to ensure same training result
        # when feature extraction in multi-thread
        predictors.sort()
        categorical.sort()

        if dump_train_data:
            train[predictors].to_csv("train_ft_dump.csv.bz2", compression='bz2',index=False)
            val[predictors].to_csv("val_ft_dump.csv.bz2", compression='bz2',index=False)
            logger.info('dumping done')
            exit(0)
        logger.debug('training with : %s', predictors)
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

                    logger.debug('train weights: %s', pd.DataFrame({'weights':train_weights}).describe())
                    logger.debug('val weights: %s', pd.DataFrame({'weights:':val_weights}).describe())



        dtrain = lgb.Dataset(train[predictors].values.astype(np.float32),
                             label=train[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical,
                             weight=train_weights,
                             free_raw_data=(config_scheme_to_use.lgbm_seed_test_list is None) and \
                                           (not isinstance(config_scheme_to_use.lgbm_params, list)) and \
                                            (not config_scheme_to_use.test_important_fts)
                             )
        dvalid = lgb.Dataset(val[predictors].values.astype(np.float32),
                             label=val[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical,
                             weight=val_weights,
                             free_raw_data=(config_scheme_to_use.lgbm_seed_test_list is None) and \
                                           (not isinstance(config_scheme_to_use.lgbm_params, list)) and \
                                            (not config_scheme_to_use.test_important_fts)
                             )

        evals_results = {}
        logger.debug("Training the model...")


        #to_dump = train[predictors]
        #to_dump.append(val[predictors])
        #to_dump.to_csv('/tmp/jjj', index=False)
        #exit(0)

        val_non_empty = (len(val) > 0)

        valid_sets_to_use = [dtrain, dvalid] if val_non_empty else [dtrain]
        valid_names_to_use = ['train', 'valid'] if val_non_empty else ['train']

        if config_scheme_to_use.lgbm_seed_test_list is not None:

            for lgbm_seed_test in config_scheme_to_use.lgbm_seed_test_list:
                lgb_model = lgb.train({**config_scheme_to_use.lgbm_params,
                                       **{
                                           'drop_seed':lgbm_seed_test,
                                           'feature_fraction_seed': lgbm_seed_test,
                                           'bagging_seed': lgbm_seed_test,
                                           'data_random_seed': lgbm_seed_test,
                                       }},
                                  dtrain,
                                  valid_sets=valid_sets_to_use,
                                  valid_names=valid_names_to_use,
                                  #valid_sets=[dvalid] if len(val) > 0 else None,
                                  #valid_names=['valid'] if len(val) > 0 else None,
                                  evals_result=evals_results,
                                  num_boost_round=1000,
                                  early_stopping_rounds=config_scheme_to_use.lgbm_params['early_stopping_round'],
                                  verbose_eval=10,
                                  feval=None)
                logger.info(
                    'trainning@seed of %d done, best iter num: %d, best train auc: %f, val auc: %f',
                    lgbm_seed_test,
                    # 'trainning done, best iter num: %d, best train auc: %f, val auc: %f',
                    lgb_model.best_iteration,
                    lgb_model.best_score['train']['auc'],
                    lgb_model.best_score['valid']['auc'] if len(val) > 0 else 0
                )

        elif isinstance(config_scheme_to_use.lgbm_params,list):

            for params in config_scheme_to_use.lgbm_params:
                lgb_model = lgb.train(params,
                                  dtrain,
                                  valid_sets=valid_sets_to_use,
                                  valid_names=valid_names_to_use,
                                  #valid_sets=[dvalid] if len(val) > 0 else None,
                                  #valid_names=['valid'] if len(val) > 0 else None,
                                  evals_result=evals_results,
                                  num_boost_round=1000,
                                  early_stopping_rounds=params['early_stopping_round'],
                                  verbose_eval=10,
                                  feval=None)
                logger.info(
                    'trainning@params of %s done, best iter num: %d, best train auc: %f, val auc: %f',
                    pformat(params),
                    # 'trainning done, best iter num: %d, best train auc: %f, val auc: %f',
                    lgb_model.best_iteration,
                    lgb_model.best_score['train']['auc'],
                    lgb_model.best_score['valid']['auc'] if len(val) > 0 else 0
                )
        elif config_scheme_to_use.test_important_fts:
            early_stopping = config_scheme_to_use.lgbm_params['early_stopping_round']
            lgb_model = lgb.train(config_scheme_to_use.lgbm_params,
                                  dtrain,
                                  valid_sets=valid_sets_to_use,
                                  valid_names=valid_names_to_use,
                                  #valid_sets=[dvalid] if len(val) > 0 else None,
                                  #valid_names=['valid'] if len(val) > 0 else None,
                                  evals_result=evals_results,
                                  num_boost_round=1000,
                                  early_stopping_rounds=early_stopping,
                                  verbose_eval=10,
                                  feval=None)
            logger.info(
                'trainning all fts done, best iter num: %d, best train auc: %f, val auc: %f',
                # 'trainning done, best iter num: %d, best train auc: %f, val auc: %f',
                lgb_model.best_iteration,
                lgb_model.best_score['train']['auc'],
                lgb_model.best_score['valid']['auc']
            )

            sorted_fts = sorted(zip(lgb_model.feature_name(), list(lgb_model.feature_importance())),
                                key=lambda x: x[1])
            importance_ordered_fts = [x[0] for x in sorted_fts]

            steps = 5

            for start_pos in range(len(predictors) // steps, len(predictors), len(predictors) // steps):
                del dtrain, dvalid
                gc.collect()
                fts_to_use = importance_ordered_fts[start_pos:]
                fts_to_use = list(set(fts_to_use).union(set(categorical)))

                dtrain = lgb.Dataset(train[fts_to_use].values.astype(np.float32),
                                     label=train[target].values,
                                     feature_name=fts_to_use,
                                     categorical_feature=categorical,
                                     weight=train_weights,
                                     free_raw_data=(config_scheme_to_use.lgbm_seed_test_list is None) and \
                                                   (not isinstance(config_scheme_to_use.lgbm_params, list)) and \
                                                   (not config_scheme_to_use.test_important_fts)
                                     )
                dvalid = lgb.Dataset(val[fts_to_use].values.astype(np.float32),
                                     label=val[target].values,
                                     feature_name=fts_to_use,
                                     categorical_feature=categorical,
                                     weight=val_weights,
                                     free_raw_data=(config_scheme_to_use.lgbm_seed_test_list is None) and \
                                                   (not isinstance(config_scheme_to_use.lgbm_params, list)) and \
                                                   (not config_scheme_to_use.test_important_fts)
                                     )
                lgb_model = lgb.train(config_scheme_to_use.lgbm_params,
                                      dtrain,
                                      valid_sets=[dtrain, dvalid] if len(val) > 0 else [dtrain],
                                      valid_names=['train', 'valid'] if len(val) > 0 else ['train'],
                                      #valid_sets=[dvalid] if len(val) > 0 else None,
                                      #valid_names=['valid'] if len(val) > 0 else None,
                                      evals_result=evals_results,
                                      num_boost_round=1000,
                                      early_stopping_rounds=early_stopping,
                                      verbose_eval=10,
                                      feval=None)
                logger.info(
                    'trainning from %d done, best iter num: %d, best train auc: %f, val auc: %f',
                    start_pos,
                    # 'trainning done, best iter num: %d, best train auc: %f, val auc: %f',
                    lgb_model.best_iteration,
                    lgb_model.best_score['train']['auc'],
                    lgb_model.best_score['valid']['auc'] if len(val) > 0 else 0
                )


        else:
            lgb_model = lgb.train(config_scheme_to_use.lgbm_params,
                              dtrain,
                              valid_sets=valid_sets_to_use,
                              valid_names=valid_names_to_use,
                              #valid_sets=[dvalid] if len(val) > 0 else None,
                              #valid_names=['valid'] if len(val) > 0 else None,
                              evals_result=evals_results,
                              num_boost_round=1000,
                              early_stopping_rounds=config_scheme_to_use.lgbm_params['early_stopping_round'],
                              verbose_eval=10,
                              feval=None)

        # Nick's Feature Importance Plot
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=[7, 10])
        # lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
        # plt.title("Light GBM Feature Importance")
        # plt.savefig('feature_import.png')

        # Feature names:
        logger.debug('Feature names: %s', lgb_model.feature_name())
        # Feature importances:
        logger.debug('Feature importances: %s', list(lgb_model.feature_importance()))


        logger.info(
                    'trainning done, best iter num: %d, best train auc: %f, val auc: %f',
                    #'trainning done, best iter num: %d, best train auc: %f, val auc: %f',
                    lgb_model.best_iteration,
                    lgb_model.best_score['train']['auc'],
                    lgb_model.best_score['valid']['auc'] if len(val) > 0 else 0
                    )
        try:
            logger.info('split importance:')
            logger.info(pformat(sorted(zip(lgb_model.feature_name(),list(lgb_model.feature_importance())),
                          key=lambda x: x[1])))

            logger.info('gain importance:')
            logger.info(pformat(sorted(zip(lgb_model.feature_name(),list(lgb_model.feature_importance(importance_type='gain'))),
                          key=lambda x: x[1])))

        except:
            logger.debug('error sorting and zipping fts')

        importance_dict = dict(zip(lgb_model.feature_name(), list(lgb_model.feature_importance())))

        feature_imp = pd.DataFrame(lgb_model.feature_name(), list(lgb_model.feature_importance()))

        persist_model = True
        if persist_model:
            logger.debug('dumping model')
            lgb_model.save_model(get_dated_filename('model.txt'))
            logger.debug('dumping predictors of this model')
            pickle.dump(predictors1, open(get_dated_filename('predictors.pickle'), 'wb'))

        val_prediction = None
        if do_val_prediction:
            logger.debug("Writing the val_prediction into a csv file...")
            # if persist_intermediate:

            logger.debug('gen val prediction')
            val_prediction = lgb_model.predict(val[predictors1], num_iteration=lgb_model.best_iteration)
            # pd.Series(val_prediction).to_csv(get_dated_filename("val_prediction.csv"), index=False)
            val_auc = roc_auc_score(val['is_attributed'], val_prediction)
            if persist_intermediate:
                val['predict'] = val_prediction
                val.to_csv(get_dated_filename("val_prediction.csv"), index=False)

    if iter_num_set != -1:
        lgb_model.best_iteration = iter_num_set

    if do_val_prediction:
        return lgb_model, val_prediction, predictors1, importance_dict, val_auc
    else:
        return lgb_model, val_prediction, predictors1, importance_dict, lgb_model.best_score['valid']['auc']  if len(val) > 0 else 0


def get_train_df():
    train = None
    if config_scheme_to_use.train_from is not None and isinstance(config_scheme_to_use.train_from, list):
        for data_from, data_to in \
                zip(sample_from_list, sample_to_list) if options.unittest else \
                zip(config_scheme_to_use.train_from, config_scheme_to_use.train_to):
            train0 = pd.read_csv(path_train_sample if options.unittest else path_train,
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
        train = pd.read_csv(path_train_sample if options.unittest else path_train,
                            dtype=dtypes,
                            header=0,
                            usecols=train_cols,
                            skiprows=range(1, config_scheme_to_use.train_from) \
                                if not options.unittest and config_scheme_to_use.train_from is not None else None,
                            nrows=config_scheme_to_use.train_to - config_scheme_to_use.train_from \
                                if not options.unittest and config_scheme_to_use.train_from is not None else None,
                            parse_dates=["click_time"])

    logger.debug('mem after loaded train data: %s', cpuStats())

    if config_scheme_to_use.train_filter and \
                    config_scheme_to_use.train_filter['filter_type'] == 'sample':
        sample_count = config_scheme_to_use.train_filter['sample_count']
        train = train.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        gc.collect()
        logger.debug('mem after filtered train data: %s', cpuStats())
    elif config_scheme_to_use.train_filter and \
                    config_scheme_to_use.train_filter['filter_type'] == 'random_sample':
        train = train.sample(frac = config_scheme_to_use.train_filter['frac'])

    gc.collect()
    return train

def get_val_df():
    val = pd.read_csv(path_train_sample if options.unittest else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.val_from) \
                            if not options.unittest and config_scheme_to_use.val_from is not None else None,
                        nrows=config_scheme_to_use.val_to - config_scheme_to_use.val_from \
                            if not options.unittest and config_scheme_to_use.val_from is not None else None,
                        parse_dates=["click_time"])

    logger.debug('mem after loaded val data: %s', cpuStats())

    if config_scheme_to_use.val_filter and \
        config_scheme_to_use.val_filter['filter_type'] == 'sample':
        sample_count = config_scheme_to_use.val_filter['sample_count']
        val = val.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        gc.collect()
        logger.debug('mem after filtered val data: %s', cpuStats())
    elif config_scheme_to_use.val_filter and \
                    config_scheme_to_use.val_filter['filter_type'] == 'random_sample':
        val = val.sample(frac = config_scheme_to_use.val_filter['frac'])



    gc.collect()
    test_null_val = False

    if test_null_val:
        val = val.query('is_attributed != 0')
        val = val.query('is_attributed != 1')

    return val

def get_test_df():
    test = pd.read_csv(path_test_sample if options.unittest else path_test,
                       dtype=dtypes,
                       header=0,
                       usecols=test_cols,
                       parse_dates=["click_time"])
    test['is_attributed'] = 0
    return test
def get_stacking_val_df():
    val = pd.read_csv(path_train_sample if options.unittest else path_train,
                        dtype=dtypes,
                        header=0,
                        usecols=train_cols,
                        skiprows=range(1, config_scheme_to_use.lgbm_stacking_val_from) \
                            if not options.unittest and config_scheme_to_use.lgbm_stacking_val_from is not None else None,
                        nrows=config_scheme_to_use.lgbm_stacking_val_to - config_scheme_to_use.lgbm_stacking_val_from \
                            if not options.unittest and config_scheme_to_use.lgbm_stacking_val_from is not None else None,
                        parse_dates=["click_time"])

    logger.debug('mem after loaded stacking val data: %s', cpuStats())

    gc.collect()
    return val

def get_test_supplement_df():
    test_supplement = pd.read_csv(path_test_supplement,
                       dtype=dtypes,
                       header=0,
                       skiprows=range(1,54583762),
                       usecols=test_cols,
                       nrows=1000 if options.unittest else None,
                       parse_dates=["click_time"])
    test_supplement['is_attributed'] = 0
    return test_supplement

def get_combined_df(gen_test_data, load_test_supplement=False):

    test_len = 0
    logger.debug('loading train data...')
    train = get_train_df()
    train_len = len(train)

    logger.debug('loading val data')
    val = get_val_df()
    val_len = len(val)

    train = train.append(val)

    del val
    gc.collect()
    logger.debug('mem after appended val data: %s', cpuStats())

    if gen_test_data and load_test_supplement:
        test_supplement = get_test_supplement_df()
        train = train.append(test_supplement)
        test_len += len(test_supplement)
        del test_supplement
        gc.collect()
    elif gen_test_data:
        test = get_test_df()
        test_len = len(test)
        train = train.append(test)
        del test
        gc.collect()


    return train, train_len, val_len, test_len

def get_checksum_from_df(df):
    #m = hashlib.md5()
    #df.apply(lambda x: m.update(x.to_string().encode('utf8')), axis = 1)
    #m.update(df.to_string().encode('utf8'))
    #ret = mmh3.hash_from_buffer(df['click_time'].astype(str).get_values().copy(order='C'), signed = False)
    if options.unittest:
        ret = mmh3.hash_from_buffer(df['click_time'].to_string(), signed = False)
    else:
        ret = mmh3.hash_from_buffer(df['click_time'].sample(frac=0.1, random_state=88).to_string(), signed = False)
    gc.collect()
    return str(ret)

def do_data_validation(df, df0, sample_indice):
    df = df.copy(True)
    df.reset_index(drop=True, inplace=True)
    df0 = df0.copy(True)
    df0.reset_index(drop=True, inplace=True)

    logger.debug('df columns to val: %s', df.columns)

    if 'ip_app_device_os_channel_is_attributednextclick' in df.columns:

        df_for_val = do_next_Click( df0, agg_type='float64')[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)

        logger.debug('var gap: %d',(df_for_val['ip_app_device_os_channel_nextClick'] - df['ip_app_device_os_channel_is_attributednextclick']).sum())
        logger.debug('var diff: %s',df['ip_app_device_os_channel_is_attributednextclick'][(df_for_val['ip_app_device_os_channel_nextClick'] - df['ip_app_device_os_channel_is_attributednextclick']) != 0].head().to_string())
        logger.debug('var diff: %s',df_for_val['ip_app_device_os_channel_nextClick'][(df_for_val['ip_app_device_os_channel_nextClick'] - df['ip_app_device_os_channel_is_attributednextclick']) != 0].head().to_string())
        logger.debug('var diff: %s',df[:][(df_for_val['ip_app_device_os_channel_nextClick'] - df['ip_app_device_os_channel_is_attributednextclick']) != 0].head().to_string())
        logger.debug('var diff: %s',df_for_val[:][(df_for_val['ip_app_device_os_channel_nextClick'] - df['ip_app_device_os_channel_is_attributednextclick']) != 0].head().to_string())

        #logger.debug('var diff: %s', df.query('ip == 64 and app == 5348 and device == 29 and os == 1 and channel == 19'))
        logger.debug('df dump: %s', df.loc[df['ip'] == 5314].loc[df['app'] == 18].loc[df['device']==1].loc[df['os']==19].loc[df['channel']==107]['click_time'])
        logger.debug('df dump: %s', df_for_val.loc[df_for_val['ip'] == 5314].loc[df_for_val['app'] == 18].loc[df_for_val['device']==1].loc[df_for_val['os']==19].loc[df_for_val['channel']==107]['click_time'])

        logger.debug('df dump: %s', df.loc[df['ip'] == 5314].loc[df['app'] == 18].loc[df['device']==1].loc[df['os']==19].loc[df['channel']==107])
        logger.debug('df dump: %s', df_for_val.loc[df_for_val['ip'] == 5314].loc[df_for_val['app'] == 18].loc[df_for_val['device']==1].loc[df_for_val['os']==19].loc[df_for_val['channel']==107])


    if 'ip_hour_is_attributedcount' in df.columns:

        df_for_val = do_count( df0, ['ip', 'hour'], 'A0', show_max=False )[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)

        logger.debug('var gap: %d',(df_for_val['A0'] - df['ip_hour_is_attributedcount']).sum())
        logger.debug('var diff: %s',df['ip_hour_is_attributedcount'][(df_for_val['A0'] - df['ip_hour_is_attributedcount']) != 0].head().to_string())
        logger.debug('var diff: %s',df_for_val['A0'][(df_for_val['A0'] - df['ip_hour_is_attributedcount']) != 0].head().to_string())


    if 'ip_device_os_app_is_attributedcount' in df.columns:

        df_for_val = do_count( df0, ['ip', 'device', 'os', 'app'], 'A0', show_max=False )[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)

        logger.debug('var gap: %d',(df_for_val['A0'] - df['ip_device_os_app_is_attributedcount']).sum())
        logger.debug('var diff: %s',df['ip_device_os_app_is_attributedcount'][(df_for_val['A0'] - df['ip_device_os_app_is_attributedcount']) != 0].head().to_string())
        logger.debug('var diff: %s',df_for_val['A0'][(df_for_val['A0'] - df['ip_device_os_app_is_attributedcount']) != 0].head().to_string())



    if 'ip_device_os_appnunique' in df.columns:
        df_for_val = do_countuniq( df0, ['ip', 'device', 'os'], 'app', 'A0', show_max=False )[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)

        logger.debug('var gap: %d',(df_for_val['A0'] - df['ip_device_os_appnunique']).sum())
        logger.debug('var diff: %s',df['ip_device_os_appnunique'][(df_for_val['A0'] - df['ip_device_os_appnunique']) != 0].head().to_string())
        logger.debug('var diff: %s',df_for_val['A0'][(df_for_val['A0'] - df['ip_device_os_appnunique']) != 0].head().to_string())


        assert (df_for_val['A0'] - df['ip_device_os_appnunique']).sum() == 0

    if 'ip_app_os_hourvar' in df.columns:
        df_for_val = do_var( df0, ['ip', 'app', 'os'], 'hour', 'A0', show_max=False ,agg_type='float64')[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)


        logger.debug('var gap: %d',(df_for_val['A0'] - df['ip_app_os_hourvar']).sum())
        logger.debug('var diff:%s',df['ip_app_os_hourvar'][(df_for_val['A0'] - df['ip_app_os_hourvar']) != 0].head().to_string())
        logger.debug('var diff:%s',df_for_val['A0'][(df_for_val['A0'] - df['ip_app_os_hourvar']) != 0].head().to_string())


        assert (df_for_val['A0'] - df['ip_app_os_hourvar']).sum() == 0

    if 'ip_app_channel_hourmean' in df.columns:
        df_for_val = do_mean( df0, ['ip', 'app', 'channel'], 'hour', 'A0', show_max=False ,agg_type='float64' )[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)


        logger.debug('var gap:%d',(df_for_val['A0'] - df['ip_app_channel_hourmean']).sum())
        logger.debug('var diff:%s',df['ip_app_channel_hourmean'][(df_for_val['A0'] - df['ip_app_channel_hourmean']) != 0].head().to_string())
        logger.debug('var diff:%s',df_for_val['A0'][(df_for_val['A0'] - df['ip_app_channel_hourmean']) != 0].head().to_string())

        assert (df_for_val['A0'] - df['ip_app_channel_hourmean']).sum() == 0

    if 'ip_device_os_channelcumcount' in df.columns:

        df_for_val = do_cumcount( df0, ['ip', 'device', 'os'], 'channel', 'A0', show_max=False )[sample_indice]; gc.collect()
        df_for_val.reset_index(drop=True, inplace=True)

        assert (df_for_val['A0'] - df['ip_device_os_channelcumcount']).sum() == 0

    #logger.debug()
def neg_sample_df(combined_df, train_len, val_len, test_len):
    neg_sample_indice = None
    if config_scheme_to_use.use_neg_sample:
        logger.debug('neg sample 1/200(1:2 pos:neg) after checksum... with seed {}'.format(config_scheme_to_use.neg_sample_seed))
        np.random.seed(config_scheme_to_use.neg_sample_seed)

        pos_in_train_count = (combined_df[:train_len]['is_attributed'] == 1).sum()
        neg_in_train_count = (combined_df[:train_len]['is_attributed'] != 1).sum()

        neg_sample_rate = neg_in_train_count // pos_in_train_count

        logger.info('pos count: %d, neg count: %d, total len: %d, sample rate: %d',
                    pos_in_train_count,
                    neg_in_train_count,
                    train_len,
                    neg_sample_rate)


        #neg_sample_indice = random.sample(range(len(combined_df)),len(combined_df) // 200)
        neg_sample_indice = (np.random.randint(0, neg_sample_rate, len(combined_df), np.uint16) == 0) \
                            | (combined_df['is_attributed'] == 1) \
                            | np.concatenate((np.zeros(train_len, np.bool_) ,np.ones(val_len + test_len,np.bool_)))
        #logger.debug('neg sample indice: len:{}, tail:{}'.format(len(neg_sample_indice), neg_sample_indice[-10:]))
        #logger.debug('neg sample indice: len:{}, head:{}'.format(len(neg_sample_indice), neg_sample_indice[:10]))

        combined_df_before_sample = combined_df.copy(True)
        #logger.debug('len before sampel:',len(combined_df_before_sample))
        logger.info('len before sampel: %d',len(combined_df))

        combined_df = combined_df[neg_sample_indice]
        #logger.debug('len after sampel:',len(combined_df_before_sample))
        logger.info('len after sampel: %d',len(combined_df))
        train_len = len(combined_df) - test_len - val_len
    else:
        combined_df_before_sample = combined_df.copy(True)

    return neg_sample_indice, combined_df, combined_df_before_sample, train_len, val_len, test_len

def get_input_data(load_test_supplement):
    with timer('load combined data df'):
        combined_df, train_len, val_len, test_len = get_combined_df(config_scheme_to_use.use_test_data or config_scheme_to_use.new_predict,
                                                                    config_scheme_to_use.use_test_supplyment)#load_test_supplement = load_test_supplement)
        logger.debug('total len: {}, train len: {}, val len: {}.'.format(len(combined_df), train_len, val_len))
        combined_df.reset_index(drop=True,inplace=True)

    with timer('checksum data'):
        checksum = get_checksum_from_df(combined_df)
        logger.debug('md5 checksum of whole data set: %s', checksum)

    with timer('gen categorical features'):
        combined_df = gen_categorical_features(combined_df)

    neg_sample_indice, combined_df, combined_df_before_sample, train_len, val_len, test_len = \
        neg_sample_df(combined_df, train_len, val_len, test_len)
    return neg_sample_indice, combined_df, combined_df_before_sample, train_len, val_len, test_len, checksum

def train_and_predict(com_fts_list, use_ft_cache = False, load_test_supplement = False,
                      return_dict = None):

    with timer('load input data from files', logging.INFO):
        neg_sample_indice, combined_df, combined_df_before_sample, train_len, val_len, test_len, checksum = \
            get_input_data(load_test_supplement)

    with timer('gen statistical hist features', logging.INFO):
        combined_df, new_features = \
        generate_counting_history_features(combined_df,
                                           use_ft_cache = use_ft_cache,
                                           ft_cache_prefix='joint',
                                           add_features_list=com_fts_list,
                                           val_start = train_len,
                                           val_end = train_len + val_len,
                                           checksum = checksum,
                                           sample_indice = neg_sample_indice,
                                           df_before_sample = combined_df_before_sample)

    data_validation = True
    if data_validation and options.unittest:
        do_data_validation(combined_df, combined_df_before_sample, neg_sample_indice)

    train = combined_df[:train_len]
    val = combined_df[train_len:train_len + val_len]

    # do val filter after feature extraction
    if config_scheme_to_use.val_filter_test_hours:
        with timer('filter val according to test hours', logging.INFO):
            logger.debug('len before val test hour filer: %d', len(val))
            logger.debug('hours before val test hour filer: %s', val.groupby(by=val["click_time"].dt.hour)['app'].count().to_string())

            val = val[val["click_time"].dt.hour.isin(most_freq_hours_in_test_data)]
            logger.debug('len after val test hour filer: %d', len(val))
            logger.debug('hours after val test hour filer: %s', val.groupby(by=val["click_time"].dt.hour)['app'].count().to_string())
            logger.debug('test')
            logger.debug('val\'s start time: %s', val['click_time'].min())
            logger.debug('val\'s end time: %s', val['click_time'].max())


    if config_scheme_to_use.dump_train_data:
        train.to_csv("train_ft_dump.csv.bz2", compression='bz2',index=False)
        val.to_csv("val_ft_dump.csv.bz2", compression='bz2',index=False)

    if dump_train_data and config_scheme_to_use.new_predict:
        test = combined_df[train_len + val_len:]
        test[categorical + new_features].to_csv("test_ft_dump.csv.bz2", compression='bz2',index=False)

    dump_data_for_validation = False
    if dump_data_for_validation:
        to_dump = train.sample(100, random_state=666)
        to_dump.append(val.sample(100, random_state=666))
        to_dump.to_csv('/tmp/dump_for_validation.csv', index=False)
        logger.info('/tmp/dump_for_validation.csv gened')
        #logger.info('train dump_data_for_validation:\n %s', train.sample(100, random_state=888).to_string())
        #logger.info('val dump_data_for_validation:\n %s', val.sample(100, random_state=888).to_string())
        exit(0)

    with timer('train lgbm model...', logging.INFO):
        lgb_model, val_prediction, predictors, importances, val_auc = train_lgbm(train, val, new_features, False)

    logger.info('trained model: %s, num best iter: %d', pformat(vars(lgb_model)), lgb_model.best_iteration)

    if config_scheme_to_use.new_predict:
        with timer('predict test data:', logging.INFO):
            test = combined_df[train_len + val_len: train_len+val_len+test_len]


            logger.debug('NAN next click count in test: %s', len(test.query('ip_app_device_os_is_attributednextclick > 1489000000')))

            predict_result = lgb_model.predict(test[predictors],
                num_iteration=lgb_model.best_iteration)
            submission = pd.DataFrame({'is_attributed':predict_result,
                                       'click_id':test['click_id'].astype('uint32').values})


            # re-read the test without supplyment and merge with id
            if config_scheme_to_use.use_test_supplyment:
                test_without_supplyment = pd.read_csv(path_test_sample if options.unittest else path_test,
                                   dtype=dtypes,
                                   header=0,
                                   usecols=['click_id'])
                submission = test_without_supplyment.merge(submission, how='left',
                                   on=['click_id'])

            logger.debug("Writing the submission data into a csv file...")

            predict_filename = get_dated_filename("submission_notebook")

            with timer('generating predict file ' + predict_filename, logging.INFO):
                submission.to_csv(predict_filename, index=False)

            if config_scheme_to_use.submit_prediction:
                with timer('submitting '+ predict_filename, logging.INFO):
                    cmd = 'kaggle competitions submit -c talkingdata-adtracking-fraud-detection -f ./%s -m \"%s\"' % \
                                           (predict_filename, config_scheme_to_use.config_name)
                    logger.info('running %s', cmd)
                    submit_ret = os.system(cmd)
                    if submit_ret != 0:
                        logger.info('submitting error, return value: %d', submit_ret)
                    else:
                        logger.info('submitting succeeded.')
            logger.debug("All done...")


    if return_dict is not None:
        return_dict['val_auc'] = val_auc
        return_dict['importances'] = importances
        return_dict['best_iter'] = lgb_model.best_iteration
        return_dict['train_auc'] = lgb_model.best_score['train']['auc']
    return importances, val_auc



def run_model():
    logger.debug('run theme: %s', config_scheme_to_use.run_theme)

    if config_scheme_to_use.run_theme == 'train_and_predict':
        logger.debug('add features list: ')
        logger.debug(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list, use_ft_cache=config_scheme_to_use.use_ft_cache)
    elif config_scheme_to_use.run_theme == 'train_and_predict_with_test_supplement':
        logger.debug('add features list: ')
        logger.debug(config_scheme_to_use.add_features_list)
        train_and_predict(config_scheme_to_use.add_features_list,
                          use_ft_cache=config_scheme_to_use.use_ft_cache,
                          load_test_supplement=True)
    else:
        logger.info("nothing to run... exit")


with timer('run_model...'):
    run_model()

logger.debug('run_model done')
