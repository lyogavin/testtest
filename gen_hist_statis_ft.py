
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import gc

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']


dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
for_train = False

data = None

if for_train:
    data = pd.read_csv('../input/train.csv', header=0,usecols=train_cols,parse_dates=["click_time"], dtype=dtypes)
    print('added hour and day')

    data['hour'] = data["click_time"].dt.hour.astype('uint8')
    data['day'] = data["click_time"].dt.day.astype('uint8')

    data.drop('click_time', inplace=True, axis=1)

else:
    data1 = pd.read_csv('../input/train.csv', header=0,usecols=train_cols,parse_dates=["click_time"], dtype=dtypes)
    print('added hour and day')

    data1['hour'] = data1["click_time"].dt.hour.astype('uint8')
    data1['day'] = data1["click_time"].dt.day.astype('uint8')

    data1.drop('click_time', inplace=True, axis=1)

    #print('sampling data')
    #data1 = data1.set_index('ip').loc[lambda x: (x.index + 401) % 10 == 0].reset_index()

    data1 = data1.query('day == 9')
    gc.collect()

    print('loaded train data @9:',len(data1))

    data = pd.read_csv('../input/test.csv', header=0,usecols=test_cols,parse_dates=["click_time"], dtype=dtypes)
    print('loaded test data :',len(data))

    data['hour'] = data["click_time"].dt.hour.astype('uint8')
    data['day'] = data["click_time"].dt.day.astype('uint8')

    data.drop('click_time', inplace=True, axis=1)

    #print('sampling data')
    #data = data.set_index('ip').loc[lambda x: (x.index + 401) % 10 == 0].reset_index()

    data = pd.concat([data1, data])

print('len read:',len(data))

gc.collect()


cvr_columns_lists = [['ip','device'],['app','channel']]

for cvr_columns in cvr_columns_lists:
    sta_ft = data[cvr_columns + ['hour','day','is_attributed']].        groupby(cvr_columns + ['day','hour'])[['is_attributed']].mean().reset_index()
    print(sta_ft.describe())
    sta_ft.info()

    sta_ft['day'] = sta_ft['day'] +1
    
    new_col_name = '_'.join(cvr_columns + ['cvr'])
    sta_ft = sta_ft.rename(columns={'is_attributed':new_col_name})
    data= data.merge(sta_ft, on=cvr_columns + ['day','hour'], how='left')

    data[new_col_name] = data[new_col_name].astype('float32')

    import gc
    del sta_ft
    gc.collect()

    print(data)
    print(data.describe())
    data.info()

if for_train:
    data.to_csv('train_with_cvr.csv.gzip', index=False, compression='gzip')

else:
    data = data.query('day == 10')
    gc.collect()
    data.to_csv('test_with_cvr.csv.gzip', index=False, compression='gzip')

# In[6]:

print(data.describe())

print('done')

