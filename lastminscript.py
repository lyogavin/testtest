
# coding: utf-8

# In[ ]:

'''
First of all, thanks to the work of: 

Md Asraful Kabir
https://www.kaggle.com/asraful70/talkingdata-added-new-features-in-lightgbm

This kernel is made for running NOT on this environment. If you run it locally, you will get a result of 0.9811

At the moment I'm lucky, to use the server of TU Berlin. So if you like to download and submit it, please go ahead.

Any ideas on improvement are welcome. 

'''


# In[ ]:

FILENO= 200102 #To distinguish the output file name.
debug=0  #Whethere or not in debuging mode

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
#import matplotlib.pyplot as plt
import os

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation as LDA
path = '../input/talkingdata-adtracking-fraud-detection/'


# In[ ]:

###### Feature extraction ######

#### Extracting next click feature 
    ### Taken help from https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
    ###Did some Cosmetic changes 

predictors=[]


def do_LDA(df, agg_suffix='LDA', agg_type='float32'):
    print(">> \nExtracting {agg_suffix} LDA features...\n")

    GROUP_BY_LDA = []

    for col1 in ['ip', 'app', 'device', 'os', 'channel']:
        for col2 in [ 'ip', 'app', 'device', 'os', 'channel']:
            if col1 != col2:
                GROUP_BY_LDA.append({'groupby': [col1, col2]})

    # Calculate the time to next click for each group
    for spec in GROUP_BY_LDA:
        # Name of new feature
        feature_name_added = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)
        sentence_dict = {}

        for groupby, joint in zip(df[spec['groupby'][0]].values,
                                  df[spec['groupby'][1]].values):
            sentence_dict.setdefault(joint, []).append(str(groupby))
        keys = list(sentence_dict.keys())
        sentences = [' '.join(sentence_dict[key]) for key in keys]
        sentences_as_matrix = CountVectorizer(analyzer = "word",
                                              token_pattern = r"(?u)\b\w+\b").\
            fit_transform(sentences)
        topics = LDA(n_components=5, learning_method='batch').fit_transform(sentences_as_matrix)
        to_merge = pd.DataFrame({spec['groupby'][1]: keys,
                                 feature_name_added + '_0':topics[:, 0],
                                 feature_name_added + '_1':topics[:, 1],
                                 feature_name_added + '_2':topics[:, 2],
                                 feature_name_added + '_3':topics[:, 3],
                                 feature_name_added + '_4':topics[:, 4]})
        df = pd.merge(df, to_merge,
                            on = spec['groupby'][1], how = 'left')
        del keys, sentences, sentences_as_matrix, topics, to_merge
        gc.collect()
        #print(df[feature_name_added + '_0'])
        predictors.append(feature_name_added + '_0')
        predictors.append(feature_name_added + '_1')
        predictors.append(feature_name_added + '_2')
        predictors.append(feature_name_added + '_3')
        predictors.append(feature_name_added + '_4')

        print('added 5 ' + feature_name_added + ' features')

    return (df)

def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    
    print(">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']},
    {'groupby': ['ip', 'os', 'device', 'channel']},
    
    {'groupby': ['ip', 'os', 'device', 'app', 'hour']},
    {'groupby': ['ip', 'os', 'device', 'channel', 'hour']},
    {'groupby': ['device']},
    {'groupby': ['device', 'channel']},     
    {'groupby': ['app', 'device', 'channel']},
    {'groupby': ['device', 'hour']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)


# In[ ]:

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    print(">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)    


# In[ ]:

## Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


# In[ ]:

##  Below a function is written to extract unique count feature from different cols
def do_countuniq( df, group_cols, counted, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


# In[ ]:

### Below a function is written to extract cumulative count feature  from different cols    
def do_cumcount( df, group_cols, counted,agg_type='uint32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


# In[ ]:

### Below a function is written to extract mean feature  from different cols
def do_mean( df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


# In[ ]:

def do_var( df, group_cols, counted, agg_type='float32', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )


# In[ ]:

###  A function is written to train the lightGBM model with different given parameters
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')


# In[ ]:


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=50, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.05,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
    }
    
    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)
    print("\nModel Report")
    print("bst1.best_iteration: ", bst1.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst1.best_iteration-1])

    return (bst1,bst1.best_iteration)


# In[ ]:

## Running the full calculation.

#### A function is written here to run the full calculation with defined parameters.

def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'click_time'    : 'str',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32'
            }

    print('loading train data...',frm,to)
    train_df = pd.read_csv(path + "train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

    print('loading test data...')
    if debug:
        test_df = pd.read_csv(path +  "test_supplement.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(path + "test_supplement.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

        
    len_train = len(train_df)
    train_df = train_df.append(test_df)
    
    del test_df
    
    train_df['click_time'] = pd.to_datetime(train_df['click_time'])
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8')     
    
    gc.collect()

    train_df = do_LDA( train_df,agg_suffix='LDA', agg_type='float32'  ); gc.collect()
    train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
    train_df = do_prev_Click( train_df,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  
    train_df = do_countuniq( train_df, ['ip'], 'channel' ); gc.collect()
    
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    train_df = do_mean( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour' ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app'); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os'); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device'); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel'); gc.collect()
    train_df = do_cumcount( train_df, ['ip'], 'os'); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
    train_df = do_count( train_df, ['ip', 'day', 'hour'] ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app']); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'os']); gc.collect()
    train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour'); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour'); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day'); gc.collect()
    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour' ); gc.collect()
    
    train_df = do_mean( train_df, ['os', 'device', 'app', 'channel'], 'hour' ); gc.collect()
    train_df = do_countuniq( train_df, ['os', 'device', 'app', 'channel'], 'hour' ); gc.collect()
    #train_df = do_var( train_df, ['os', 'device', 'app', 'channel'], 'hour' ); gc.collect()

    
    
    print(train_df.head(5))
    gc.collect()
    
    
    print('\n\nBefore appending predictors...\n\n',sorted(predictors))
    target = 'is_attributed'
    word= ['app','device','os', 'channel', 'hour']
    for feature in word:
        if feature not in predictors:
            predictors.append(feature)
    #categorical = ['app', 'device', 'os', 'channel', 'hour', 'os_device_app_channel_by_hour_countuniq']
    categorical = ['app', 'device', 'os', 'channel', 'hour']

    print('\n\nAfter appending predictors...\n\n',sorted(predictors))

    test_df = train_df[len_train:]
    val_df = train_df[(len_train-val_size):len_train]
    train_df = train_df[:(len_train-val_size)]

    print("\ntrain size: ", len(train_df))
    print("\nvalid size: ", len(val_df))
    print("\ntest size : ", len(test_df))

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id'].astype('int')

    gc.collect()

    print('neg sampling...')
    np.random.seed(999)

    pos_in_train_count = (train_df['is_attributed'] == 1).sum()
    neg_in_train_count = (train_df['is_attributed'] != 1).sum()
    neg_sample_rate = neg_in_train_count // pos_in_train_count

    print('pos count: {}, neg count: {}, total len: {}, sample rate: {}'.format(
                pos_in_train_count,
                neg_in_train_count,
                len(train_df),
                neg_sample_rate))

    neg_sample_indice = (np.random.randint(0, neg_sample_rate, len(train_df), np.uint16) == 0) \
                        | (train_df['is_attributed'] == 1)

    train_df = train_df[neg_sample_indice]

    print("Training...")
    start_time = time.time()
    new_lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.02,
        'num_leaves': 9,
        'max_depth': 5,
        'min_child_samples': 100,
        'max_bin': 150,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'min_child_weight': 0,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'nthread': 16,
        'verbose': 9,
        'early_stopping_round': 500,
        # 'is_unbalance': True,
        'scale_pos_weight': 1.0#99.0
    }
    params = {
        'learning_rate': 0.05,
        'nthread': 16,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.9,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 

    }

    for jj in range(1):
        print('round ' + str(jj) + ' trainning')
        (bst,best_iteration) = lgb_modelfit_nocv(new_lgbm_params, #params,
                                train_df,
                                val_df,
                                predictors,
                                target,
                                objective='binary',
                                metrics='auc',
                                early_stopping_rounds=800,
                                verbose_eval=True,
                                num_boost_round=2000,
                                categorical_features=categorical)

    print('[{}]: model training time'.format(time.time() - start_time))
    del train_df
    del val_df
    gc.collect()


    #ax = lgb.plot_importance(bst, max_num_features=300)
    
    #plt.savefig('test%d.png'%(fileno), dpi=600,bbox_inches="tight")
    #plt.show()


    print("Features importance...")
    gain = bst.feature_importance('gain')
    ft = pd.DataFrame({'feature':bst.feature_name(), 'split':bst.feature_importance('split'), 'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    print(ft)    
    

    ##############################################
    print('loading test data...')
    if debug:
        dfTest = pd.read_csv(path +  "test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        dfTest = pd.read_csv(path + "test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    
    ip_dfTest_supplement = pd.DataFrame()
    ip_dfTest_supplement['ip'] = test_df['ip']
    ip_dfTest_supplement['app'] = test_df['app']
    ip_dfTest_supplement['device'] = test_df['device']
    ip_dfTest_supplement['os'] = test_df['os']
    ip_dfTest_supplement['channel'] = test_df['channel']
    ip_dfTest_supplement['click_time'] = test_df['click_time']

    print("Predicting...")
    ip_dfTest_supplement['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
    
    del test_df
    
    ip_dfTest_supplement = ip_dfTest_supplement.groupby(['ip',
                                                    'app',
                                                    'device',
                                                    'os',
                                                    'channel',
                                                    'click_time'])['is_attributed'].agg(['mean']).reset_index().rename(index=str, 
                                                        columns=({'ip': 'ip',
                                                                  'mean': 'is_attributed'}))
    print(ip_dfTest_supplement.columns)
    
    sub = pd.DataFrame()
    sub = dfTest.merge(ip_dfTest_supplement, how='left',on=['ip','app', 'device', 'os', 'channel','click_time'])
    
    print(len(sub))
    print(sub.columns)
    del dfTest
    del ip_dfTest_supplement
    
    
    print("del ip,app,device,os,channel,click_weekday")
    del sub['ip']
    del sub['app']
    del sub['device']
    del sub['os']
    del sub['channel']
    del sub['click_time']

    print(len(sub))
    print(sub.head(5))
    
    ##############################################

    sub.to_csv('sub_it%d.csv'%(fileno),index=False,float_format='%.9f')
    print("done...")
    return sub
    


# In[ ]:

####### Chunk size defining and final run  ############

nrows=184903891-1
nchunk=25000000
#val_size=2500000
#val_size=5000000
val_size=40000000


#frm=nrows-65000000
frm=0
to=nrows
if debug:
    frm=0
    nchunk=100000
    val_size=10000
    to=frm+nchunk


sub=DO(frm,to,FILENO)



# In[6]:
'''
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df1 = pd.read_csv('../input/sub-it200102csv/sub_it200102.csv')
print(len(df1))
df1.to_csv('sub-it200102csv', index=False, float_format='%.9f')
'''

# In[ ]:



