
# coding: utf-8

# In[ ]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
            name = '_'.join([col, str(val)])
            if col_type.kind ==  'O':
                ffm.append(':'.join[self.field_index_[col], self.feature_index_[name],'1'])
            elif col_type.kind == 'i':
                ffm.append(':'.join([self.field_index_[col], self.feature_index_[col], str(val)]))
        return ' '.join(ffm)

    def transform(self, df):
        t = df.dtypes.to_dict()
        print('transforming')
        return pd.Series({idx: self.transform_row_(idx, row, t) for idx, row in df.iterrows()})





from sklearn.model_selection import train_test_split
import lightgbm as lgb
import sys
import gc


path = '../input/' 
path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
        
skip = range(1, 140000000)
print("Loading Data")
#skiprows=skip, 
train = pd.read_csv(path_train, dtype=dtypes,
        header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)


FFM = False
if FFM:
    ffm_train = FFMFormatPandas()
    ffm_train_data = ffm_train.fit_transform(train, y='is_attributed')

    print('FFM data:',ffm_train_data)





test = pd.read_csv(path_test, dtype=dtypes, header=0,
        usecols=test_cols,parse_dates=["click_time"])#.sample(1000)
#test['is_attributed'] = -1

len_train = len(train)
print('The initial size of the train set is', len_train)
print('The initial size of the test set is', len(test))
print('Binding the training and test set together...')



print("Creating new time features in train: 'hour' and 'day'...")
train['hour'] = train["click_time"].dt.hour.astype('uint8')
train['day'] = train["click_time"].dt.day.astype('uint8')


print("Creating new time features in test: 'hour' and 'day'...")
test['hour'] = test["click_time"].dt.hour.astype('uint8')
test['day'] = test["click_time"].dt.day.astype('uint8')


# In[2]:

def prepare_data(data, training_day, profile_days, sample_count=1):
    if sample_count != 1:
        #sample 1/4 of the data:
        data = data.set_index('ip').loc[lambda x: (x.index + 401) % sample_count == 0].reset_index()
        len_train = len(data)
        print('len after sample:', len_train)

    train_ip_contains_training_day = data.groupby('ip').filter(lambda x: x['day'].max() == training_day)

    print('train_ip_contains_training_day', train_ip_contains_training_day)
    print('train_ip_contains_training_day unique ips:', len(train_ip_contains_training_day['ip'].unique()))

    train_ip_contains_training_day = train_ip_contains_training_day         .query('day < {0} & day > {1}'.format(training_day, training_day - 1 - profile_days) )
    print('train_ip_contains_training_day unique ips:', len(train_ip_contains_training_day['ip'].unique()))

    print('split attributed data:')
    train_ip_contains_training_day_attributed = train_ip_contains_training_day.query('is_attributed == 1')
    print('len:',len(train_ip_contains_training_day_attributed))

    #only use data on 9 to train, but data before 9 as features
    train = data.query('day == {}'.format(training_day))
    print('training data len:', len(train))
    
    return train, train_ip_contains_training_day, train_ip_contains_training_day_attributed

def add_statistic_feature(group_by_cols, training, training_hist, training_hist_attribution, 
                          with_hist, counting_col='channel'):
    features_added = []
    feature_name_added = '_'.join(group_by_cols) + 'count'
    print('count ip with group by:', group_by_cols)
    n_chans = training[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]]         .count().reset_index().rename(columns={counting_col: feature_name_added})
    training = training.merge(n_chans, on=group_by_cols, how='left')
    del n_chans
    gc.collect()
    training[feature_name_added] = training[feature_name_added].astype('uint16')
    features_added.append(feature_name_added)
    
    if with_hist:
        print('count ip with group by in hist data:', group_by_cols)
        feature_name_added = '_'.join(group_by_cols) + "count_in_hist"
        n_chans = training_hist[group_by_cols + [counting_col]].groupby(by=group_by_cols)[[counting_col]]             .count().reset_index().rename(columns={counting_col: feature_name_added})
        training = training.merge(n_chans, on=group_by_cols, how='left')
        del n_chans
        gc.collect()
        #training[feature_name_added] = training[feature_name_added].astype('uint16')
        print('count ip attribution with group by in hist data:', group_by_cols)
        feature_name_added1 = '_'.join(group_by_cols) + "count_attribution_in_hist"
        n_chans = training_hist_attribution[group_by_cols + [counting_col]]             .groupby(by=group_by_cols)[[counting_col]]             .count().reset_index().rename(columns={counting_col: feature_name_added1 })
        training = training.merge(n_chans, on=group_by_cols, how='left')
        del n_chans
        gc.collect()
        #training[feature_name_added1] = training[feature_name_added1].astype('uint16')
                                               
        training['_'.join(group_by_cols) + "count_attribution_rate_in_hist"] =             training[feature_name_added] / training[feature_name_added1]
            
        features_added.append(feature_name_added)
        features_added.append(feature_name_added1)
        features_added.append('_'.join(group_by_cols) + "count_attribution_rate_in_hist")
        
    print('added features:', features_added)
                                               
    return training, features_added

def generate_counting_history_features(data, history, history_attribution):
        
    new_features = []

    # Count by IP,DAY,HOUR
    print('a given IP address within each hour...')
    data, features_added = add_statistic_feature(['ip','day','hour'], data, history, history_attribution, False)
    new_features = new_features + features_added
    gc.collect()

    # Count by IP and APP
    data, features_added = add_statistic_feature(['ip','app'], data, history, history_attribution, True)
    new_features = new_features + features_added
    
    # Count by IP and channel
    data, features_added = add_statistic_feature(['ip','channel'], data, history, history_attribution, True, counting_col='os')
    new_features = new_features + features_added
    
    # Count by IP and channel app
    data, features_added = add_statistic_feature(['ip','channel', 'app'], data, history, history_attribution, True, counting_col='os')
    new_features = new_features + features_added
    
    data, features_added  = add_statistic_feature(['ip','app','os'], data, history, history_attribution, True)
    new_features = new_features + features_added

    #######
    # Count by IP
    data, features_added  = add_statistic_feature(['ip'], data, history, history_attribution, True)
    new_features = new_features + features_added

    # Count by IP HOUR CHANNEL                                               
    data, features_added  = add_statistic_feature(['ip','hour','channel'],         data, history, history_attribution, True, counting_col='os')
    new_features = new_features + features_added

    # Count by IP HOUR Device
    data, features_added  = add_statistic_feature(['ip','hour','os'],         data, history, history_attribution, True)
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['ip','hour','app'],         data, history, history_attribution, True)
    new_features = new_features + features_added

    data, features_added  = add_statistic_feature(['ip','hour','device'],         data, history, history_attribution, True)
    new_features = new_features + features_added
    
    return data, new_features

#test['hour'] = test["click_time"].dt.hour.astype('uint8')
#test['day'] = test["click_time"].dt.day.astype('uint8')

train, train_ip_contains_training_day, train_ip_contains_training_day_attributed =     prepare_data(train, 9, 3, 4)

train, new_features = generate_counting_history_features(train, train_ip_contains_training_day, 
                                                         train_ip_contains_training_day_attributed)

print('train data:', train)
print('new features:', new_features)

val = train.set_index('ip').loc[lambda x: (x.index) % 17 == 0].reset_index()
print(val)
print('The size of the validation set is ', len(val))

gc.collect()

train = train.set_index('ip').loc[lambda x: (x.index) % 17 != 0].reset_index()
print('The size of the train set is ', len(train))

target = 'is_attributed'
train[target] = train[target].astype('uint8')
train.info()

train.to_csv('training.csv')
val.to_csv('val.csv')

sys.exit(0)


# In[6]:

print(train[['ip_channelcount','ip_channel_appcount']])


# In[7]:

predictors0 = ['device', 'app', 'os', 'channel', 'hour', # Starter Vars, Then new features below
              'ip_day_hourcount','ipcount','ip_appcount', 'ip_app_oscount',
              "ip_hour_channelcount", "ip_hour_oscount", "ip_hour_appcount","ip_hour_devicecount"]

categorical = ['app', 'device', 'os', 'channel', 'hour']

predictors1 = categorical + new_features
#for ii in new_features:
#    predictors1 = predictors1 + ii
#print(predictors1)
gc.collect()

#train.fillna(value={x:-1 for x in new_features})

print("Preparing the datasets for training...")

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves': 7,  
    'max_depth': 4,  
    'min_child_samples': 100,  
    'max_bin': 150,  
    'subsample': 0.7,  
    'subsample_freq': 1,  
    'colsample_bytree': 0.7,  
    'min_child_weight': 0,  
    'subsample_for_bin': 200000,  
    'min_split_gain': 0,  
    'reg_alpha': 0,  
    'reg_lambda': 0,  
    'nthread': 5,
    'verbose': 9,
    #'is_unbalance': True,
    'scale_pos_weight':99 
    }
    
predictors_to_train = [predictors1]

for predictors in predictors_to_train:
    print('training with :', predictors)
    #print('training data: ', train[predictors].values)
    #print('validation data: ', val[predictors].values)
    dtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    dvalid = lgb.Dataset(val[predictors].values, label=val[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )

    evals_results = {}
    print("Training the model...")

    lgb_model = lgb.train(params, 
                     dtrain, 
                     valid_sets=[dtrain, dvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=1000,
                     early_stopping_rounds=30,
                     verbose_eval=50, 
                     feval=None)

    #del train
    #del val
    #gc.collect()

    # Nick's Feature Importance Plot
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=[7,10])
    lgb.plot_importance(lgb_model, ax=ax, max_num_features=len(predictors))
    plt.title("Light GBM Feature Importance")
    plt.savefig('feature_import.png')

    # Feature names:
    print('Feature names:', lgb_model.feature_name())
    # Feature importances:
    print('Feature importances:', list(lgb_model.feature_importance()))

    feature_imp = pd.DataFrame(lgb_model.feature_name(),list(lgb_model.feature_importance()))
    
    
    


# In[ ]:

for_test = True

if for_test:
    del train
    del test
    gc.collect()

    #prepare test data:
    train = pd.read_csv(path_train, dtype=dtypes,
            header=0,usecols=train_cols,parse_dates=["click_time"])#.sample(1000)
    test = pd.read_csv(path_test, dtype=dtypes, header=0,
            usecols=test_cols,parse_dates=["click_time"])#.sample(1000)
    train=train.append(test)
    del test
    gc.collect()
    print("Creating new time features in train: 'hour' and 'day'...")
    train['hour'] = train["click_time"].dt.hour.astype('uint8')
    train['day'] = train["click_time"].dt.day.astype('uint8')
    
    train, train_ip_contains_training_day, train_ip_contains_training_day_attributed =         prepare_data(train, 10, 3, 1)

    train, new_features = generate_counting_history_features(train, train_ip_contains_training_day, 
                                                             train_ip_contains_training_day_attributed)


# In[ ]:

print('test data:', train)

print('new features:', new_features)
print("Preparing data for submission...")

submit = pd.read_csv(path_test, dtype='int', usecols=['click_id'])
print('submit test len:', len(submit))
print("Predicting the submission data...")
submit['is_attributed'] = lgb_model.predict(train[predictors1], num_iteration=lgb_model.best_iteration)

print("Writing the submission data into a csv file...")

submit.to_csv("submission_notebook.csv",index=False)

print("All done...")






# In[ ]:


