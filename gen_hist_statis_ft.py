
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import gc
import math
import time


import os, psutil
def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    gc.collect()
    return memoryUse

from contextlib import contextmanager
@contextmanager

def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {} s, mem GB: {}'.format(name, time.time() - t0, cpuStats()))


train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

chunk_size = 10000
sample_count = 1

use_sample = True

count_without_attribution = True

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
for_train = True


def get_next_batch_for_train(iter, chunk_size):
    with timer("load training data chunk and gen hour/day"):
        print('loading data')
        if iter is None:
            iter = pd.read_csv(
                '../input/talkingdata-adtracking-fraud-detection/train_sample.csv' if use_sample \
                    else '../input/talkingdata-adtracking-fraud-detection/train.csv',
                chunksize = chunk_size,
                header=0,
                usecols=train_cols,
                parse_dates=["click_time"], dtype=dtypes)
        data = next(iter, None)

        if data is None:
            return iter, None

        print('added hour and day')

        data['hour'] = data["click_time"].dt.hour.astype('uint8')
        data['day'] = data["click_time"].dt.day.astype('uint8')

        if sample_count != 0:
            print('sampling data')
            data = data.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        data.drop('click_time', inplace=True, axis=1)
        print('len read:', len(data))

        gc.collect()

        return iter, data

def next_batch_test():
    with timer("load training data and gen hour/day"):
        print('loading data')
        data1 = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train_sample.csv' if use_sample
            else '../input/talkingdata-adtracking-fraud-detection/train.csv',
            chunksize=chunk_size, header=0,usecols=train_cols,
            parse_dates=["click_time"], dtype=dtypes)
        #if chunk_read is not None:
        #    data = next(data)
        print('added hour and day')

        data1['hour'] = data1["click_time"].dt.hour.astype('uint8')
        data1['day'] = data1["click_time"].dt.day.astype('uint8')

         #data1.drop('click_time', inplace=True, axis=1)

        if sample_count != 0:
            print('sampling data')
            data1 = data1.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()

        with timer("filter to only keep 9th data"):
            data1 = data1.query('day == 9')
            gc.collect()

            print('loaded train data @9:',len(data1))

        with timer("load test data and gen hour/day"):
            data = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv',
                               chunksize=chunk_size, header=0,usecols=test_cols,
                               parse_dates=["click_time"], dtype=dtypes)
            data = next(data)
            print('loaded test data :',len(data))

            data['hour'] = data["click_time"].dt.hour.astype('uint8')
            data['day'] = data["click_time"].dt.day.astype('uint8')

            #data.drop('click_time', inplace=True, axis=1)
            if sample_count != 0:
                print('sampling data')
                data = data.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()

        with timer("concat test and train"):
            data = pd.concat([data1, data])

    print('len read:',len(data))

    gc.collect()
log_group = 100000  # 1000 views -> 60% confidence, 100 views -> 40% confidence


def rate_calculation(x):
    """Calculate the attributed rate. Scale by confidence"""
    rate = x.sum() / float(x.count())
    conf = np.min([1, np.log(x.count()) / log_group])
    return rate * conf

cvr_columns_lists = [

    #['ip', 'app', 'device', 'os', 'channel']

    # best cvr tested:
    ['ip','device'],
    ['ip'], ['os'], ['channel']




    #['app','channel'],
    #['app'], ['device']

    #['ip', 'device', 'hour'],
    #['app', 'channel', 'hour'],
    #['ip', 'hour'], ['app', 'hour'], ['device', 'hour'], ['os', 'hour'], ['channel', 'hour'],
    #['ip', 'app', 'device', 'os', 'channel', 'hour'],

    # V2 Features #
    ###############
    #['app', 'os', 'hour'],
    #['app', 'device', 'hour'],
]
def persist(data, name):
    if for_train:
        with timer("store data:"+ name):

            #data.drop('hour', inplace=True, axis=1)
            #data.drop('day', inplace=True, axis=1)
            data.to_csv(name, index=False)

    else:
        with timer("store data:"+ name):

            #data.drop('hour', inplace=True, axis=1)
            #data.drop('day', inplace=True, axis=1)
            gc.collect()
            data.to_csv(name, index=False)

    # In[6]:

    print(data.describe())

print('done')

log_group = 100000

agg_types = ['non_attr_count', 'cvr']

# Aggregation function
def rate_calculation(sum, count, type):
    if type == 'non_attr_count':
        if sum == 0:
            return count
        else:
            return 0
    else:
        if count ==0:
            return 0
        """Calculate the attributed rate. Scale by confidence"""
        rate = sum / float(count)
        conf = min(1, math.log(count) / log_group)
        return rate * conf


def add_category_columns(data):
    x = None
    for col in cvr_columns:
        if x is None:
            x = data[col].astype(str)
        else:
            x = x + "_" + data[col].astype(str)

    if previous_day:
        y = (data['day'] - 1).astype(str)

    gc.collect()

    print('gen category and previous_category in data...')
    with timer("gen category and previous_category in data... " + new_col_name):
        if previous_day:
            data['previous_category'] = (x + "_" + y).apply(hash) % D
            del y
            gc.collect()
        data['category'] = (x + "_" + data['day'].astype(str)).apply(hash) % D
        del x
        gc.collect()
    return data
D = 2 ** 26

previous_day = False
for cvr_columns in cvr_columns_lists:
    new_col_name = '_'.join(cvr_columns)  + ['_']
    iter = None
    rates = {type: [] for type in agg_types}

    with timer("gen cvr for " + new_col_name):

        click_buffer = np.full(D, 0, dtype=np.uint32)
        attribution_buffer = np.full(D, 0, dtype=np.uint32)

        print('1st round of chunk iteration...')
        iter, data = get_next_batch_for_train(None, chunk_size)

        while(data is not None):
            print('processing chunk:', len(data))
            data = add_category_columns(data)

            i=0
            for category, is_attributed in zip(data['category'].values, data['is_attributed'].values):
                if i %10000 == 0:
                    print("processing {} line in first round of loop".format(i))
                click_buffer[category] += 1
                attribution_buffer[category] += is_attributed
                i+=1

            iter, data = get_next_batch_for_train(iter, chunk_size)
            gc.collect()

        del data
        del iter
        gc.collect()

        iter, data = get_next_batch_for_train(None, chunk_size)

        print('2nd chunk iteration:')
        while (data is not None):
            print('processing chunk:', len(data))

            data = add_category_columns(data)

            for type in agg_types:
                print('itering type:',type)
                i=0
                for category in data['previous_category'].values if previous_day else data['category'].values:
                    if i %10000 == 0:
                        print("processing {} line in 2nd round of loop".format(i))
                    rates[type].append(rate_calculation(attribution_buffer[category], click_buffer[category], type))
                    i+=1

                #gc.collect()
                #data[new_col_name + type] = list(rates)
                #data[new_col_name + type] = data[new_col_name + type].astype('float16')
                #del rates
                gc.collect()

                print('describe of the new col:', new_col_name)
                #print(data[new_col_name + type].describe())
            iter, data = get_next_batch_for_train(iter, chunk_size)
            gc.collect()
            print('rate len so far %d for type %s.' %(len(rates[type]), type))

        del (click_buffer)
        del (attribution_buffer)
        gc.collect()

    print('persisting ', new_col_name)

    for type in agg_types:
        to_persist = pd.DataFrame(rates[type], dtype='float16')
        persist(to_persist,
                new_col_name + type + '.train.csv' if for_train else new_col_name + type + '.test.csv')

    with timer("dropping  " + new_col_name + type):
        del rates
        gc.collect()



