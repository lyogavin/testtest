# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import gc
import math
from contextlib import contextmanager
@contextmanager
def timer(name):
	t0 = time.time()
	yield
	print('[{}] done in {} s'.format(name, time.time() - t0))

import os, psutil
def cpuStats():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0] / 2. ** 30
	print('memory GB:', memoryUse)

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}

use_sample = False


path = '../input/talkingdata-adtracking-fraud-detection/'

path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'


batchsize = 10000000

ATTRIBUTION_CATEGORIES = [
    # V1 Features #
    ###############
    ['ip'], ['app'], ['device'], ['os'], ['channel'],

    # V2 Features #
    ###############
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
]

log_group = 100000
# Aggregation function
def rate_calculation(sum, count):
    """Calculate the attributed rate. Scale by confidence"""
    rate = sum / float(count)
    conf = math.min(1, math.log(count) / log_group)
    return rate * conf

rcount = 0
for data in pd.read_csv(path_train_sample if use_sample else path_train, engine='c', chunksize=batchsize,
                        sep=",", dtype=dtypes):
    rcount += batchsize


    data['hour'] = data["click_time"].dt.hour.astype('uint8')
    data['day'] = data["click_time"].dt.day.astype('uint8')

    for cvr_category in ATTRIBUTION_CATEGORIES:
        buffer_click_count = np.full(D, 0, dtype=np.uint32)
        buffer_attribution_count = np.full(D, 0, dtype=np.uint32)
        buffer_rate = np.full(D, 0, dtype=np.float32)

        # build current day profile features:
        with timer("building current day profile features"):
            D = 2 ** 26

            for index, row in data.iterrows():
                category_id_list = [row[field].astype(str) for field in cvr_category]
                category_id = hash('_'.join(category_id_list)) % D
                buffer_click_count[category_id] += 1
                if row['is_attributed'] == 1:
                    buffer_attribution_count[category_id] += 1

            for index, row in data.iterrows():
                category_id_list = [row[field].astype(str) for field in cvr_category]
                category_id = hash('_'.join(category_id_list)) % D
                buffer_rate[category_id] = rate_calculation(buffer_attribution_count[category_id],
                                                            buffer_click_count[category_id])


            data['category'] = (data['ip'].astype(str) + "_" + data['app'].astype(str) + "_" + \
                                data['device'].astype(str) \
                                + "_" + data['os'].astype(str) + "_" + data['channel'].astype(str)).apply(hash) % D
            click_count_later = []
            for category in reversed(data['category'].values):
                click_count_later.append(click_count_buffer[category])
                click_count_buffer[category] += 1
            del (click_count_buffer)
            data['click_count_later'] = list(reversed(click_count_later))
        gc.collect()
        new_features = new_features + ['click_count_later']

gc.collect()

cvr_columns_lists = [['ip', 'device'], ['app', 'channel']]

for cvr_columns in cvr_columns_lists:
    sta_ft = data[cvr_columns + ['hour', 'day', 'is_attributed']].groupby(cvr_columns + ['day', 'hour'])[
        ['is_attributed']].mean().reset_index()
    print(sta_ft.describe())
    sta_ft.info()

    sta_ft['day'] = sta_ft['day'] + 1

    new_col_name = '_'.join(cvr_columns + ['cvr'])
    sta_ft = sta_ft.rename(columns={'is_attributed': new_col_name})
    data = data.merge(sta_ft, on=cvr_columns + ['day', 'hour'], how='left')

    data[new_col_name] = data[new_col_name].astype('float32')

    import gc

    del sta_ft
    gc.collect()

    print(data)
    print(data.describe())
    data.info()

if for_train:
    data.drop('hour', inplace=True, axis=1)
    data.drop('day', inplace=True, axis=1)
    data.to_csv('train_with_cvr.csv.gzip', index=False, compression='gzip')

else:
    data = data.query('day == 10')
    data.drop('hour', inplace=True, axis=1)
    data.drop('day', inplace=True, axis=1)
    gc.collect()
    data.to_csv('test_with_cvr.csv.gzip', index=False, compression='gzip')

# In[6]:

print(data.describe())

print('done')

