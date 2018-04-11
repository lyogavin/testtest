
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

chunk_read = None
use_sample = True
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

data = None

if for_train:
    with timer("load training data and gen hour/day"):
        print('loading data')
        data = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv',
                           chunksize = chunk_read, header=0,usecols=train_cols,
                           parse_dates=["click_time"], dtype=dtypes)
        if chunk_read is not None:
            data = next(data)
        print('added hour and day')

        data['hour'] = data["click_time"].dt.hour.astype('uint8')
        data['day'] = data["click_time"].dt.day.astype('uint8')

        if chunk_read is None and use_sample:
            print('sampling data')
            data = data.set_index('ip').loc[lambda x: (x.index + 401) % 6 == 0].reset_index()
        #data.drop('click_time', inplace=True, axis=1)

else:
    with timer("load training data and gen hour/day"):
        print('loading data')
        data1 = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv',
                            chunksize=chunk_read, header=0,usecols=train_cols,
                            parse_dates=["click_time"], dtype=dtypes)
        if chunk_read is not None:
            data = next(data)
        print('added hour and day')

        data1['hour'] = data1["click_time"].dt.hour.astype('uint8')
        data1['day'] = data1["click_time"].dt.day.astype('uint8')

         #data1.drop('click_time', inplace=True, axis=1)

        if chunk_read is None and use_sample:
            print('sampling data')
            data1 = data1.set_index('ip').loc[lambda x: (x.index + 401) % 6 == 0].reset_index()

    with timer("filter to only keep 9th data"):
        data1 = data1.query('day == 9')
        gc.collect()

        print('loaded train data @9:',len(data1))

    with timer("load test data and gen hour/day"):
        data = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv',
                           chunksize=chunk_read, header=0,usecols=test_cols,
                           parse_dates=["click_time"], dtype=dtypes)
        if chunk_read is not None:
            data = next(data)
        print('loaded test data :',len(data))

        data['hour'] = data["click_time"].dt.hour.astype('uint8')
        data['day'] = data["click_time"].dt.day.astype('uint8')

        #data.drop('click_time', inplace=True, axis=1)
        if chunk_read is None and use_sample:
            print('sampling data')
            data = data.set_index('ip').loc[lambda x: (x.index + 401) % 6 == 0].reset_index()

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
    ['ip','device'],
    ['ip', 'app', 'device', 'os', 'channel'],
    ['app','channel'],
    ['ip'], ['app'], ['device'], ['os'], ['channel'],

    # V2 Features #
    ###############
    ['app', 'os'],
    ['app', 'device'],

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
# Aggregation function
def rate_calculation(sum, count):
    if count ==0:
        return 0
    """Calculate the attributed rate. Scale by confidence"""
    rate = sum / float(count)
    conf = min(1, math.log(count) / log_group)
    return rate * conf

for cvr_columns in cvr_columns_lists:
    new_col_name = '_'.join(cvr_columns + ['cvr'])

    with timer("gen cvr for " + new_col_name):
        old_way = False
        if old_way:
            sta_ft = data[cvr_columns + ['day','is_attributed']].\
                groupby(cvr_columns + ['day'])[['is_attributed']].apply(rate_calculation).reset_index()

            sta_ft['day'] = sta_ft['day'] + (1 if chunk_read == 0 else 0)

            sta_ft = sta_ft.rename(columns={'is_attributed':new_col_name})
            data= data.merge(sta_ft, on=cvr_columns + ['day'], how='left')

            data[new_col_name] = data[new_col_name].astype('float32')

            import gc
            del sta_ft
            #gc.collect()

            #print(data)

        D = 2 ** 26
        x = None
        for col in cvr_columns:
            if x is None:
                x = data[col].astype(str)
            else:
                x = x + "_" + data[col].astype(str)

        y = (data['day'] - 1).astype(str)

        print('gen category and previous_category in data...')
        with timer("gen category and previous_category in data... " + new_col_name):
            data['category'] = (x + "_" + data['day'].astype(str)).apply(hash) % D
            data['previous_category'] = (x + "_" + y).apply(hash) % D


        click_buffer = np.full(D, 0, dtype=np.uint32)
        attribution_buffer = np.full(D, 0, dtype=np.uint32)

        #data['epochtime'] = data['click_time'].astype(np.int64) // 10 ** 9
        rates = []
        i=0
        for category, is_attributed in zip(data['category'].values, data['is_attributed'].values):
            if i %10000 == 0:
                print("processing {} line in first round of loop".format(i))
            click_buffer[category] += 1
            attribution_buffer[category] += is_attributed
            i+=1

        i=0
        for category in data['previous_category'].values:
            if i %10000 == 0:
                print("processing {} line in 2nd round of loop".format(i))
            rates.append(rate_calculation(attribution_buffer[category], click_buffer[category]))
            i+=1

        del (click_buffer)
        del (attribution_buffer)
        data.drop('category', inplace=True, axis=1)
        data.drop('previous_category', inplace=True, axis=1)

        gc.collect()
        data[new_col_name] = list(rates)
        data[new_col_name] = data[new_col_name].astype('float32')
        del rates
        gc.collect()

        print('describe of the new col:', new_col_name)
        print(data[new_col_name].describe())




        #data.info()

    print('persisting ', new_col_name)

    persist(data[new_col_name] if for_train else data.query('day == 10')[new_col_name],
            new_col_name + '.train.csv' if for_train else new_col_name + '.test.csv')

    with timer("dropping  " + new_col_name):
        data.drop(new_col_name, axis=1, inplace=True)
        gc.collect()



