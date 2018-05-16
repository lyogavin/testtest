import train_utils.constants


default_lgbm_params = {
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
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}

new_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}

lgbm_params_search_128_114 = dict(new_lgbm_params)
lgbm_params_search_128_114.update({
    'colsample_bytree': 0.8793460386326015, 'learning_rate': 0.19814501809928017, 'max_depth': 9,
    'min_child_samples': 188, 'min_child_weight': 4, 'num_leaves': 11, 'reg_alpha': 0.02387225386312356,
    'reg_lambda': 1.2196200544739068e-09, 'scale_pos_weight': 231.48637373544372,
    'subsample': 0.7079619705989065}
)
lgbm_params_search_128_610 = dict(new_lgbm_params)
lgbm_params_search_128_610.update({
    'colsample_bytree': 0.7773614836495996, 'learning_rate': 0.2, 'max_depth': 10, 'min_child_samples': 10,
    'min_child_weight': 0, 'num_leaves': 11, 'reg_alpha': 1.0, 'reg_lambda': 1e-09,
    'scale_pos_weight': 249.99999999999994, 'subsample': 0.6870745956370757}
)
lgbm_params_l1 = dict(new_lgbm_params)
lgbm_params_l1.update({
    'reg_alpha': 1.0
})
lgbm_params_pub_entire_set = dict(new_lgbm_params)
lgbm_params_pub_entire_set.update({
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'learning_rate': 0.2, # 【consider using 0.1】
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'scale_pos_weight': 200, # because training data is extremely unbalanced
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth), default=31
        'max_depth': 3,  # -1 means no limit, default=-1
        'min_data_per_leaf': 100,  # alias=min_data_per_leaf , min_data, min_child_samples, default=20
        'max_bin': 100,  # Number of bucketed bin for feature values,default=255
        'subsample': 0.7,  # Subsample ratio of the training instance.default=1.0, alias=bagging_fraction
        'subsample_freq': 1,  # k means will perform bagging at every k iteration, <=0 means no enable,alias=bagging_freq,default=0
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.alias:feature_fraction
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf),default=1e-3,Like min_data_in_leaf, it can be used to deal with over-fitting
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 24, # should be equal to REAL cores:http://xgboost.readthedocs.io/en/latest/how_to/external_memory.html
        'verbose': 0
})

lgbm_params_pub_entire_set_no_early_iter_153 = dict(lgbm_params_pub_entire_set)
lgbm_params_pub_entire_set_no_early_iter_153.update({
        'early_stopping_round': 153,
        'num_boost_round': 153,
    })

lgbm_params_pub_entire_set_no_early_iter_205 = dict(lgbm_params_pub_entire_set)
lgbm_params_pub_entire_set_no_early_iter_205.update({
        'early_stopping_round': 205,
        'num_boost_round': 205,
    })

lgbm_params_pub_asraful_kernel = dict(new_lgbm_params)
lgbm_params_pub_asraful_kernel.update({
        'learning_rate': 0.10,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced
    })

new_lgbm_params_feature_fraction = {**new_lgbm_params, ** {
    'feature_fraction': 0.5
}}

new_lgbm_params_iter_600 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 600,
    'num_boost_round': 600,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}
new_lgbm_params_100_cat_smooth = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
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
    'nthread': 10,
    'verbose': 9,
    'early_stopping_round': 20,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0,
    'cat_smooth':100
}
new_lgbm_params_early_300 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
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
    'nthread': 10,
    'verbose': 9,
    'num_boost_round':300,
    'early_stopping_round': 300,
    # 'is_unbalance': True,
    'scale_pos_weight': 99.0
}


new_lgbm_params_early_50 = dict(new_lgbm_params)
new_lgbm_params_early_50['early_stopping_round'] = 50
new_lgbm_params_early_50['nthread'] = 24


new_lgbm_params_early_415 = dict(new_lgbm_params)
new_lgbm_params_early_415['num_boost_round'] = 415
new_lgbm_params_early_415['early_stopping_round'] = 415


public_kernel_lgbm_params = dict(new_lgbm_params)
public_kernel_lgbm_params.update({
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200, # because training data is extremely unbalanced
        'early_stopping_round': 30
    })

new_lagbm_params_from_115_0 = dict(new_lgbm_params)
new_lagbm_params_from_115_0.update( \
    {'colsample_bytree': 1.0, 'learning_rate': 0.1773256374384233, 'max_depth': 3, 'min_child_samples': 200, 'min_child_weight': 0,
    'min_split_gain': 0.0007911719321269061, 'num_leaves': 11, 'reg_alpha': 2.355979159306278e-08,
    'reg_lambda': 0.9016760858543618, 'scale_pos_weight': 260.6441151527916, 'subsample': 1.0, 'subsample_for_bin': 457694, 'subsample_freq': 0} )

new_lgbm_params1 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
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
    'nthread': 5,
    'verbose': 9,
    'early_stopping_round': 100, #20,
    # 'is_unbalance': True,
    'scale_pos_weight': 200.0
}

