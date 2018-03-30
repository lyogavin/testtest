#!/usr/bin/env python3

# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse,sys

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))



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
        print('fitting')
        df_ffm = df[df.columns.difference([self.y])]
        if self.field_index_ is None:
            self.field_index_ = {col: i for i, col in enumerate(df_ffm)}

        if self.feature_index_ is not None:
            last_idx = max(list(self.feature_index_.values()))

        if self.feature_index_ is None:
            self.feature_index_ = dict()
            last_idx = 0

        for col in df.columns:
            print('fit procssing ', col)
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
        if idx % 1000 == 0:
            print('transforming idx: {}'.format(idx))
        ffm = []
        if self.y != None:
            ffm.append(str(row.loc[row.index == self.y][0]))
        if self.y is None:
            ffm.append(str(0))
            
        items = row.loc[row.index != self.y].to_dict().items()

        for col, val in items:
            col_type = t[col]
            name = '_'.join([str(col), str(val)])
            if col_type.kind ==  'O':
                ffm.append(':'.join[str(self.field_index_[col]), str(self.feature_index_[name]),'1'])
            elif col_type.kind == 'i':
                ffm.append(':'.join([str(self.field_index_[col]), str(self.feature_index_[col]), str(val)]))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        print('transforming')
        return pd.Series({idx: self.transform_row_(idx, row, t) for idx, row in df.iterrows()})





if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('tr_src_path')
parser.add_argument('va_src_path')
parser.add_argument('tr_dst_path')
parser.add_argument('va_dst_path')
args = vars(parser.parse_args())


data = pd.read_csv(args['tr_src_path'], parse_dates=["click_time"])#.sample(1000)
ffm_train = FFMFormatPandas()
ffm_train_data = ffm_train.fit_transform(data, y='is_attributed')
print('converted data:', ffm_train_data)

ffm_train_data.to_csv(args['tr_dst_path'])

data = pd.read_csv(args['va_src_path'], parse_dates=["click_time"])#.sample(1000)
ffm_train = FFMFormatPandas()
ffm_train_data = ffm_train.fit_transform(data, y='is_attributed')
print('converted data:', ffm_train_data)

ffm_train_data.to_csv(args['va_dst_path'])
