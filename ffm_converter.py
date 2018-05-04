#!/usr/bin/env python3

# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 




####!/usr/bin/env python3 -m cProfile

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse,sys
from collections import OrderedDict
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
import hashlib
import csv
import mmh3
from contextlib import contextmanager

import os, psutil, time


def cpuStats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    gc.collect()
    # all_objects = muppy.get_objects()
    # sum1 = summary.summarize(all_objects)
    # summary.print_(sum1)

    return memoryUse

@contextmanager
def timer(name):
    t0 = time.time()
    print('start ',name)
    yield
    print('[{}] done in {} s'.format(name, time.time() - t0))
    print('mem after {}: {}'.format(name, cpuStats()))

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

NR_BINS = 1000000

def hashstr(input):
    return str(mmh3.hash(input) %(NR_BINS-1)+1)
    #return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16)%(NR_BINS-1)+1)

print(os.listdir("../input"))

import logging
logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('./ffm_converter.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)



# Any results you write to the current directory are saved as output.

'''
Another CTR comp and so i suspect libffm will play its part, after all it is an atomic bomb for this kind of stuff.
A sci-kit learn inspired script to convert pandas dataframes into libFFM style data.

The script is fairly hacky (hey thats Kaggle) and takes a little while to run a huge dataset.
The key to using this class is setting up the features dtypes correctly for output (ammend transform to suit your needs)

Example below


'''


class FFMFormatPandas:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None
        self.y = None

    def fit(self, df, y=None):
        self.y = y
        logger.info('fitting')
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            logger.info('fit procssing %s', col)
            vals = df[col].unique()
            for val in vals:
                if pd.isnull(val):
                    continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, idx, row, t):
        if idx % 10000 == 0:
            logger.info('transforming idx: %d, %s, %s',idx, row, t)
        ffm = []
        ffm.append(str(idx))
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))
            
        items = row.loc[row.index != self.y].to_dict(into=OrderedDict).items()

        for col, val in items:
            #col_type = t[col]
            name = '_'.join([str(col), str(val)])
            ffm.append(hashstr(name))
            #if col_type.kind ==  'O':
            #    ffm.append(':'.join[str(self.field_index_[col]), str(self.feature_index_[name]),'1'])
            #elif col_type.kind == 'u' or col_type.kind == 'i':
            #    ffm.append(':'.join([str(self.field_index_[col]), str(self.feature_index_[col]), str(val)]))

        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        logger.info('transforming')
        if 'click_id' in df.columns:
            print('use click_id as idx')
            df = df.set_index('click_id')
        return pd.Series({idx: self.transform_row_(idx, row, t) for idx, row in df.iterrows()})


def convert_features_to_text(data, predictors, hash = False):
    NR_BINS = 1000000

    def hashstr(input):
        #return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16) % (NR_BINS - 1) + 1)
        return str(mmh3.hash_from_buffer(input, signed=False) % (NR_BINS - 1) + 1)

    with timer('convert_features_to_text'):
        i = 0
        str_array = None
        assign_name_id = 0
        for feature in predictors:
            if feature == 'click_id':
                continue
            acro_name_to_dump = ''
            if not feature in acro_names:
                print('{} missing acronym, assign name AN{}'.format(feature, assign_name_id))
                acro_name_to_dump = 'AN' + str(assign_name_id)
                assign_name_id += 1
            else:
                acro_name_to_dump =  acro_names[feature]
            if str_array is None:
                if 'click_id' in data.columns:
                    str_array = data['click_id'].astype(str) + ' '
                else:
                    str_array = data.index.astype(str) + ' '

                if 'is_attributed' in data.columns:
                    str_array = str_array + data['is_attributed'].astype(str)
                else:
                    str_array = str_array + '0'
            else:
                if not hash:
                    str_array = str_array + " " + acro_name_to_dump + "_" + data[feature].astype(str)
                else:
                    temp = (acro_name_to_dump + "_" + data[feature].astype(str)). \
                        apply(hashstr)
                    str_array = str_array + " " + temp

            gc.collect()
            print('mem after gc:', cpuStats())
            i += 1

        str_array = str_array.values
        return str_array



if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path')
parser.add_argument('va_src_path')
parser.add_argument('tr_dst_path')
parser.add_argument('va_dst_path')
args = vars(parser.parse_args())

import pickle
dtypes = pickle.load(open("output_dtypes.pickle",'rb'))
print('use dtypes:',dtypes)
#data = pd.read_csv(args['tr_src_path'], parse_dates=["click_time"],dtype=dtypes,header=0)#.sample(1000)
#data.drop('click_time', axis=1, inplace=True)
#data.drop('ip', axis=1, inplace=True)
#print('data:',data)

#data = pd.read_csv(args['va_src_path'], parse_dates=["click_time"],dtype=dtypes,header=0)#.sample(1000)
#data.drop('click_time', axis=1, inplace=True)
#data.drop('ip', axis=1, inplace=True)

with timer("convert data:"):
    old_way = False
    if old_way:
        with timer('load train:'):
            data = pd.read_csv(args['tr_src_path'], dtype=dtypes, header=0, engine='c')  # .sample(1000)
        with timer('convert train:'):
            ffm_train = FFMFormatPandas()
            ffm_train_data = ffm_train.fit_transform(data, y='is_attributed')
            logger.info('converted data: %s', ffm_train_data)

            ffm_train_data.to_csv(args['tr_dst_path'], index=False)

        with timer('load val:'):
            data = pd.read_csv(args['va_src_path'], dtype=dtypes, header=0, engine='c')  # .sample(1000)
        with timer('convert val:'):
            ffm_train = FFMFormatPandas()
            ffm_train_data = ffm_train.fit_transform(data, y='is_attributed')
            logger.info('converted data: %s', ffm_train_data)

            ffm_train_data.to_csv(args['va_dst_path'], index=False)
    else:
        with timer('load train:'):
            data = pd.read_csv(args['tr_src_path'], dtype=dtypes, header=0, engine='c')  # .sample(1000)
        with timer('convert train:'):
            str_array = convert_features_to_text(data, data.columns, True)
            np.savetxt(open(args['tr_dst_path'], 'w'), str_array, '%s')

        with timer('load val:'):
            data = pd.read_csv(args['va_src_path'], dtype=dtypes, header=0, engine='c')  # .sample(1000)
        with timer('convert val:'):
            str_array = convert_features_to_text(data, data.columns, True)
            np.savetxt(open(args['va_dst_path'], 'w'), str_array, '%s')