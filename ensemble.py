# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit, logit
import sys
import gc
import time

def get_dated_filename(filename):
    if filename.rfind('.') != -1:
        id = filename.rfind('.')
        filename = filename[:id] + '_' + ensembel_theme_to_use_name + filename[id:]
    else:
        filename = filename + '_' + ensembel_theme_to_use_name
    print('got file name: {}'.format(filename))
    return filename



debug = False

ensemble_models = {
    '124_3':{
        'file':'/mnt/ensemble_predictions_lgbm/124_3.csv',
        'scale':1.0,
        'LB': 0.9793,
        'note':'best single'
    },
    '124_9':{
        'file':'/mnt/ensemble_predictions_lgbm/124_9.csv',
        'scale':1.0,
        'LB': 0.9785,
        'note': 'lgbm+pub ftrl fts'
    },
    '124_11':{
        'file':'/mnt/ensemble_predictions_lgbm/submission_notebook_train_config_124_11.csv',
        'scale':1.0,
        'LB': 0.9787,
        'note': 'best nunique fts'
    },
    '124_20':{
        'file':'/mnt/ensemble_predictions_lgbm/124_20.csv',
        'scale':1.0,
        'LB': 0.9795,
        'note':  '07/08/09 each train a model then merge(5:3:2)'
    },
    '124_28':{
        'file':'/mnt/ensemble_predictions_lgbm/_IGNORE_submission_notebook_train_config_124_28.csv',
        'scale':1.0,
        'LB': 0.9779,
        'note': 'best nunique and count from coms search'
    },
    '124_37':{
        'file':'/mnt/ensemble_predictions_lgbm/submission_notebook_train_config_124_37.csv',
        'scale':1.0,
        'LB': 0.9790,
        'note': 'hourly alpha 1day before smooth cvr ft, best coms search'
    },
    '124_31':{
        'file':'/mnt/ensemble_predictions_lgbm/_IGNORE_submission_notebook_train_config_124_31.csv',
        'scale':1.0,
        'LB': 0.9786,
        'note': 'use new params from search'
    },
    '124_41':{
        'file':'/mnt/ensemble_predictions_lgbm/124_41.csv',
        'scale':1.0,
        'LB': 0.9786,
        'note': 'use 3 hours counting features to simulate streaming kernel'
    },
    '124_42':{
        'file':'/mnt/ensemble_predictions_lgbm/124_42.csv',
        'scale':1.0,
        'LB': 0.9780,
        'note': 'use 15 min as hour counting features as multi scale '
    },
    '117_8':{
        'file':'/mnt/ensemble_predictions_l2/117_8.csv_SCALE_500',
        'scale':1.0,
        'LB': 0.9780,
        'note': 'best FFM model'
    },
    '117_12':{
        'file':'/mnt/ensemble_predictions_ffm/117_12.csv',
        'scale':1.0,
        'LB': 0.9777,
        'note': 'new cvr fts + interactive fts + params tunning'
    },
    '123_0':{
        'file':'/mnt/ensemble_predictions_streaming/123_0.csv',
        'scale':1.0,
        'LB': 0.9769,
        'note': 'pub fm ftrl kernel'
    },
    '123_10':{
        'file':'/mnt/ensemble_predictions_streaming/123_0.csv',
        'scale':1.0,
        'LB': 0.9710,
        'note': 'pub fm ftrl kernel for DNN model'
    },
    '130_20':{
        'file':'/mnt/ensemble_predictions_lgbm/130_20.csv',
        'scale':1.0,
        'LB': 0.9798,
        'note': '124_3*0.375 + 124_20*0.375 + 124_37*0.25'
    },
    # from pub kernels:
    '140_1' : {
        'file':'/mnt/ensemble_predictions_pub/sub_it6.csv',
        'LB': 0.9798,
        'note': 'from wenjie bai pub kernel trained full data'
    },
    '140_2' : {
        'file':'/mnt/ensemble_predictions_pub/submission_geo.csv',
        'LB': 0.0,
        'note': 'from log-and-harmonic-mean-lets-go'
    },
    '140_3' : {
        'file':'/mnt/ensemble_predictions_pub/submission_avg.csv',
        'LB': 0.0,
        'note': 'from log-and-harmonic-mean-lets-go'
    },
    '140_4' : {
        'file':'/mnt/ensemble_predictions_pub/submission_final4.csv',
        'LB': 0.9812,
        'note': 'simple merge and avg'
    },
    '140_5' : {
        'file':'/mnt/ensemble_predictions_pub/sub_it7.csv',
        'LB': 0.9811,
        'note': 'sub it 7'
    },
    #ensembled models:
    '132_3' : {
        'file':'/mnt/ensemble_submission_ensemble_theme_132_3.csv',
        'LB': 0.9799,
        'note': 'from ensemble 132_3'
    },
    '132_2' : {
        'file':'/mnt/ensemble_submission_ensemble_theme_132_2.csv',
        'LB': 0.9800,
        'note': 'from ensemble 132_2'
    },
    '132_5' : {
        'file':'/mnt/ensemble_submission_ensemble_theme_132_5.csv',
        'LB': 0.9804,
        'note': 'from ensemble 132_5'
    },

}

if debug:
    ensemble_models = {
        '100_1':{
            'file':'./ensemble_predictions/1.csv',
            'LB': 1.0,
            'note': 'test 1'
        },
        '100_2':{
            'file':'./ensemble_predictions/2.csv',
            'LB': 0.9798,
            'note': 'test 2'
        }
    }

test_ensemble_theme_100 = {
    '100_1':0.3,
    '100_2':0.7
}

ensemble_theme_132_1 = {
    '130_20': 0.7,
    '124_41': 0.05,
    '117_8': 0.05,
    '117_12': 0.05,
    '123_0': 0.13,
    '123_10': 0.02,
}

ensemble_theme_132_2 = {
    '130_20': 0.68,
    '124_41': 0.05,
    '117_8': 0.05,
    '117_12': 0.05,
    '123_0': 0.10,
    '123_10': 0.02,
    '124_42': 0.05
}
ensemble_theme_132_3 = {
    '130_20': 0.50,
    '124_41': 0.09,
    '117_8': 0.08,
    '117_12': 0.08,
    '123_0': 0.11,
    '123_10': 0.04,
    '124_42': 0.10
}

ensemble_theme_132_4 = {
    '132_2': 0.65,
    '140_1': 0.35
}


ensemble_theme_132_5 = {
    '132_2': 0.60,
    '140_1': 0.40
}
ensemble_theme_132_6 = {
    '132_5': 0.20,
    '140_4': 0.80
}
ensemble_theme_132_7 = {
    '132_5': 0.20,
    '140_5': 0.80
}
ensemble_theme_132_8 = {
    '132_5': 0.25,
    '140_5': 0.75
}
ensemble_theme_132_9 = {
    '132_5': 0.25,
    '140_4': 0.75
}

ensemble_theme_132_10 = {
    '132_5': 0.30,
    '140_4': 0.70
}

ensembel_theme_to_use_name = '!!!!!!!!!WRONG!!!!!!!!!!!!!'

def use_ensemble_theme(str):
    global ensembel_theme_to_use
    global ensembel_theme_to_use_name
    ensembel_theme_to_use = eval(str)
    ensembel_theme_to_use_name = str
    print('using ensemble theme:', str)

if len(sys.argv) > 1:
    use_ensemble_theme(sys.argv[1])
else:
    use_ensemble_theme('ensemble_theme_132_10')


if debug:
    print('debuging')
    use_ensemble_theme('test_ensemble_theme_100')

gen_test_input = True

path = '../input/'
path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'
path_val = 'val.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

categorical = ['app', 'device', 'os', 'channel', 'hour']

dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'float64',
    'click_id': 'uint32'
}

skip = range(1, 140000000)
print("Loading Data")
# skiprows=skip,

import pickle

# gen lgbm val prediction

#lgb_model = lgb.Booster(model_file='model.txt')
#submit['is_attributed'] = lgb_model.predict(train[predictors1], num_iteration=lgb_model.best_iteration)
import math

almost_zero = 1e-10
almost_one = 1 - almost_zero


vlogit = np.vectorize(logit)
vlogistic = np.vectorize(expit)


sum = None


for item in ensembel_theme_to_use:
    to_ensemble = ensemble_models[item]
    scale = ensembel_theme_to_use[item]
    print('processing ensemble id {} with scale {}: {}'.format(item, scale, to_ensemble))

    lgbm_submission = pd.read_csv(to_ensemble['file'], header = 0,dtype=dtypes)
    values = vlogit(lgbm_submission['is_attributed'].clip(almost_zero, almost_one))

    normalization = False
    if normalization:
        std = values.std()
        std = 1 if std == 0 else std
        values = (values - values.mean()) / std
    values = values * scale

    if sum is None:
        sum = lgbm_submission#pd.DataFrame({'is_attributed': values, 'click_id': lgbm_submission['click_id']})
        sum['is_attributed'] = values
    else:
        sum['is_attributed'] = sum['is_attributed'] + values

    del lgbm_submission
    del values
    gc.collect()

sum['is_attributed'] = vlogistic(sum['is_attributed'])

sum.to_csv(get_dated_filename('ensemble_submission.csv'), index=False)

print('ensemble done')
