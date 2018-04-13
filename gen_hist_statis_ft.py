
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

chunk_size = 1000000
sample_count = 1

use_sample = False

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
for_test = True

def get_next_batch_for_train(iter, chunk_size, get_train_data=True):
    with timer("load training data chunk and gen hour/day"):
        print('loading data')
        if iter is None:
            file_path = ''
            if get_train_data:
                file_path = '../input/talkingdata-adtracking-fraud-detection/train_sample.csv' if use_sample \
                    else '../input/talkingdata-adtracking-fraud-detection/train.csv'
            else:
                file_path = '../input/talkingdata-adtracking-fraud-detection/test_sample.csv' if use_sample \
                    else '../input/talkingdata-adtracking-fraud-detection/test.csv'
            iter = pd.read_csv(
                file_path,
                chunksize = chunk_size,
                header=0,
                usecols=train_cols if get_train_data else test_cols,
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


log_group = 100000  # 1000 views -> 60% confidence, 100 views -> 40% confidence


def rate_calculation(x):
    """Calculate the attributed rate. Scale by confidence"""
    rate = x.sum() / float(x.count())
    conf = np.min([1, np.log(x.count()) / log_group])
    return rate * conf

cvr_columns_lists = [

    ['ip', 'app', 'device', 'os', 'channel'],

    # best cvr tested:
    ['ip','device'],
    ['ip'], ['os'], ['channel'],




    ['app', 'os'],
    ['app','channel'],
    ['app', 'device']
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
    with timer("store data:"+ name):

        #data.drop('hour', inplace=True, axis=1)
        #data.drop('day', inplace=True, axis=1)
        gc.collect()
        data.to_csv(name+".bz2", index=False,compression='bz2')

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

test_os = False
D = 2 ** 26
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
        data['category'] = x.apply(hash) % D
        del x
        gc.collect()
    return data

previous_day = False
for cvr_columns in cvr_columns_lists:
    new_col_name = '_'.join(cvr_columns)  + '_'
    iter = None
    rates = {type: np.full(184903892, -float('inf'), dtype=np.float16) for type in agg_types}
    rates_idx = {type: 0 for type in agg_types}

    with timer("gen cvr for " + new_col_name):

        click_buffer = np.full(D, 0, dtype=np.uint32)
        attribution_buffer = np.full(D, 0, dtype=np.uint32)

        print('1st round of chunk iteration...')
        iter, data = get_next_batch_for_train(None, chunk_size)

        while(data is not None):
            print('processing chunk:', len(data))
            if test_os:
                data['os'] = 10
            data = add_category_columns(data)

            i=0
            for category, is_attributed in zip(data['category'].values, data['is_attributed'].values):
                if i %100000 == 0:
                    print("processing {} line in first round of loop".format(i))
                click_buffer[category] += 1
                attribution_buffer[category] += is_attributed
                i+=1

            iter, data = get_next_batch_for_train(iter, chunk_size)
            gc.collect()

        #print('count buffer ', click_buffer[click_buffer>0])
        #print('sum buffer ', attribution_buffer[attribution_buffer>0])
        del data
        del iter
        gc.collect()

        marks = ['for_train', 'for_test']
        for mark in marks:
            if not eval(mark):
                continue

            print('for {} chunk iteration:'.format(mark))

            iter, data = get_next_batch_for_train(None, chunk_size, mark == 'for_train')
            while (data is not None):
                print('processing chunk:', len(data))
                if test_os:
                    data['os'] = 10

                data = add_category_columns(data)

                for type in agg_types:
                    print('itering type:',type)
                    i=0
                    for category in data['previous_category'].values if previous_day else data['category'].values:
                        if i %100000 == 0:
                            print("processing {} line in 2nd round of loop".format(i))
                            #print('cat:{}, attr buffer: {}, click buff: {}'.format(category,
                            #                                                       attribution_buffer[category],
                            #                                      click_buffer[category]))
                        rates[type][rates_idx[type]] = rate_calculation(attribution_buffer[category],
                                                                  click_buffer[category], type)
                        rates_idx[type] +=1
                        i+=1

                    #print('non zero rates:', list(filter(lambda x:x>0, rates[type])))

                    #gc.collect()
                    #data[new_col_name + type] = list(rates)
                    #data[new_col_name + type] = data[new_col_name + type].astype('float16')
                    #del rates
                    gc.collect()

                    print('describe of the new col:', new_col_name)
                    #print(data[new_col_name + type].describe())
                iter, data = get_next_batch_for_train(iter, chunk_size, for_train)
                gc.collect()
                print('rate len so far %d for type %s.' %(rates_idx[type], type))

            for type in agg_types:
                to_persist = pd.DataFrame(rates[type][:rates_idx[type]], dtype='float16')
                persist(to_persist,
                        new_col_name + type + '.train.csv' if mark == 'for_train' \
                            else new_col_name + type + '.test.csv')
                rates_idx[type] = 0

        del (click_buffer)
        del (attribution_buffer)
        gc.collect()

    print('persisting ', new_col_name)


    with timer("dropping  " + new_col_name + type):
        del rates
        del rates_idx
        gc.collect()



