import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from os import walk
import gc
import time

from contextlib import contextmanager
import sys
import os, psutil
import pickle


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




print(os.listdir("../input"))

almost_zero = 1e-10
almost_one = 1 - almost_zero
id_8_4am = 82259195
id_8_3pm = 118735619
id_9_4am = 144708152
id_9_3pm = 181878211
id_7_4am = 22536989
id_7_3pm = 56845833
id_9_4pm = 184903891 -1
id_7_0am = 9308570
id_9_0am = 131886955
id_8_0am = 68941880

id_9_3pm_reserve_last_250w = id_9_3pm - 250*10000

use_sample = False

path = '../input/talkingdata-adtracking-fraud-detection/'

lbscores = {}

path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'

def get_stacking_val_df(lgbm_stacking_val_from, lgbm_stacking_val_to):
    val = pd.read_csv(path_train_sample if use_sample else path_train,
                        dtype={'is_attributed': 'uint8'},
                        header=0,
                        usecols=['is_attributed'],
                        skiprows=range(1, lgbm_stacking_val_from) \
                            if not use_sample and lgbm_stacking_val_from is not None else None,
                        nrows=lgbm_stacking_val_to - lgbm_stacking_val_from \
                            if not use_sample and lgbm_stacking_val_from is not None else None)

    print('mem after loaded stacking val data:', cpuStats())

    gc.collect()
    return val


val_dir = '/mnt/stacking/'

if len(sys.argv) > 1:
    print('using input dir as ensemble prediction dir: ', sys.argv[1])
    val_dir = sys.argv[1]

val_score_files = []
prediction_files = []

val_score_prefix = 'stacking_val_train_config_'

for (dirpath, dirnames, filenames) in walk(val_dir):
    for f in filenames:
        if f[0:len(val_score_prefix)] != val_score_prefix:
            continue
        elif not f[len(val_score_prefix):] in filenames:
            print('skipping {} due to corresponding prediction {} not found.'.format(f, f[len(val_score_prefix):]))
        else:
            print('adding ', f)
            val_score_files.append(f)
            prediction_files.append(f[len(val_score_prefix):])

    break

if len(prediction_files) <= 1:
    print('exiting not enough files found')
    exit(-1)

with timer('preparing training data.'):

    cvdata = pd.DataFrame( {
        f:pd.read_csv(val_dir + val_score_prefix + f)['is_attributed'].clip(almost_zero,almost_one).apply(logit)
        for f in prediction_files
        } )
    X_train = np.array(cvdata[prediction_files])


    print('loading validation labels:')
    lgbm_stacking_val_from = id_9_3pm_reserve_last_250w
    lgbm_stacking_val_to = id_9_3pm

    print('the stacking val score time ranges has to stay the same with training: {}~{}'.
          format(lgbm_stacking_val_from, lgbm_stacking_val_to))


    y_train = get_stacking_val_df(lgbm_stacking_val_from, lgbm_stacking_val_to)['is_attributed']

    print(cvdata.head())

    print('cvdata corr:')
    print(cvdata.corr())

with timer('fit the stacking model.'):

    stack_model = LogisticRegression()
    stack_model.fit(X_train, y_train)

    print('stack_model coef:')
    print(stack_model.coef_)


    weights = stack_model.coef_/stack_model.coef_.sum()
    columns = cvdata[prediction_files].columns
    scores = [ roc_auc_score( y_train, expit(cvdata[c]) )  for c in columns ]
    names = [ c for c in columns ]
    lb = [ lbscores[c] if c in lbscores else 0 for c in columns ]
    print(pd.DataFrame( data={'LB score': lb, 'CV score':scores, 'weight':weights.reshape(-1)}, index=names ))

    print(  'Stacker score: ', roc_auc_score( y_train, stack_model.predict_proba(X_train)[:,1] )  )


with timer('generating final sub....'):
    final_sub = pd.DataFrame()
    subs = {m:pd.read_csv(val_dir+m).rename({'is_attributed':m},axis=1) for m in prediction_files}
    first_model = prediction_files[0]
    final_sub['click_id'] = subs[first_model]['click_id']

    df = subs[first_model]
    for m in prediction_files:
        if m != first_model:
            df = df.merge(subs[m], on='click_id')  # being careful in case clicks are in different order
    df.head()

    X_test = np.array( df.drop(['click_id'],axis=1)[prediction_files].clip(almost_zero,almost_one).apply(logit) )
    final_sub['is_attributed'] = stack_model.predict_proba(X_test)[:,1]
    print(final_sub.head(10))

with timer('saving final sub file'):
    final_sub_file = val_dir + 'sub_stacked.csv'
    final_sub.to_csv(final_sub_file, index=False, float_format='%.9f')

    print(final_sub_file + ' generated.')

with timer('saving model...'):
    with open(val_dir + 'stacking_model.pickle', 'wb') as file:
        pickle.dump(stack_model, file)


