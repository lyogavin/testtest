import copy
import pprint
import json
from train_utils.constants import *
from train_utils.model_params import *
from train_utils.features_def import *
import collections
from train_utils.utils import *


logger = getLogger()




class ConfigScheme:
    def __init__(self, predict=False, train=True, ffm_data_gen=False,
                 train_filter=None,
                 val_filter=shuffle_sample_filter,
                 test_filter=None,
                 lgbm_params=default_lgbm_params,
                 discretization=0,
                 mock_test_with_val_data_to_test=False,
                 train_start_time=train_time_range_start,
                 train_end_time=train_time_range_end,
                 val_start_time=val_time_range_start,
                 val_end_time=val_time_range_end,
                 gen_ffm_test_data=False,
                 add_hist_statis_fts=False,
                 seperate_hist_files=False,
                 train_wordbatch=False,
                 log_discretization=False,
                 predict_wordbatch=False,
                 use_interactive_features=False,
                 wordbatch_model='FM_FTRL',
                 train_wordbatch_streaming=False,
                 new_train=False,
                 train_from=None,
                 train_to=None,
                 val_from=None,
                 val_to=None,
                 new_predict = False,
                 run_theme = '',
                 add_features_list = add_features_list_origin,
                 use_ft_cache = False,
                 use_ft_cache_from = None,
                 qcut = 0,
                 add_second_ft = False,
                 use_lgbm_fts = False,
                 sync_mode = False,
                 normalization = False,
                 add_10min_ft = False,
                 pick_hours_weighted = False,
                 adversial_val_weighted = False,
                 adversial_val_ft=False,
                 add_in_test_frequent_dimensions = None,
                 add_lgbm_fts_from_saved_model = False,
                 train_smoothcvr_cache_from = None,
                 train_smoothcvr_cache_to = None,
                 test_smoothcvr_cache_from = None,
                 test_smoothcvr_cache_to = None,
                 add_lgbm_fts_from_saved_model_count = 20,
                 use_hourly_alpha_beta = False,
                 add_lgbm_fts_from_saved_model_filename = 'train_config_124_3_model.txt',
                 add_lgbm_fts_from_saved_model_predictors_pickle_filename = \
                         'train_config_124_3_model_predictors.pickle',
                 lgbm_stacking_val_from = None,
                 lgbm_stacking_val_to =None,
                 new_lib_ffm_output = False,
                 use_hour_group = None,
                 add_n_min_as_hour = None,
                 auto_type_cast = False,
                 val_smoothcvr_cache_from = None,
                 val_smoothcvr_cache_to = None,
                 dump_train_data=False,
                 use_neg_sample = False,
                 neg_sample_seed=888,
                 use_scvr_cache_file = False,
                 ft_search_op = 'smoothcvr',
                 submit_prediction = False,
                 val_filter_test_hours = False,
                 lgbm_seed = None,
                 lgbm_seed_test_list = None,
                 test_important_fts = False
                 ):
        self.predict = predict
        self.train = train
        self.ffm_data_gen = ffm_data_gen
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.lgbm_params = lgbm_params
        self.discretization = discretization
        self.mock_test_with_val_data_to_test = mock_test_with_val_data_to_test
        self.train_start_time = train_start_time
        self.train_end_time = train_end_time
        self.val_start_time = val_start_time
        self.val_end_time = val_end_time
        self.gen_ffm_test_data = gen_ffm_test_data
        self.add_hist_statis_fts = add_hist_statis_fts
        self.seperate_hist_files = seperate_hist_files
        self.train_wordbatch = train_wordbatch
        self.log_discretization = log_discretization
        self.predict_wordbatch = predict_wordbatch
        self.use_interactive_features = use_interactive_features
        self.wordbatch_model = wordbatch_model
        self.train_wordbatch_streaming = train_wordbatch_streaming
        self.new_train = new_train
        self.train_from=train_from
        self.train_to=train_to
        self.val_from=val_from
        self.val_to=val_to
        self.new_predict = new_predict
        self.run_theme = run_theme
        self.add_features_list = add_features_list
        self.use_ft_cache = use_ft_cache
        self.use_ft_cache_from = use_ft_cache_from
        self.qcut = qcut
        self.add_second_ft = add_second_ft
        self.use_lgbm_fts = use_lgbm_fts
        self.sync_mode = sync_mode
        self.normalization = normalization
        self.add_10min_ft = add_10min_ft
        self.pick_hours_weighted = pick_hours_weighted
        self.adversial_val_weighted = adversial_val_weighted
        self.adversial_val_ft = adversial_val_ft
        self.add_in_test_frequent_dimensions = add_in_test_frequent_dimensions
        self.add_lgbm_fts_from_saved_model = add_lgbm_fts_from_saved_model
        self.train_smoothcvr_cache_from = train_smoothcvr_cache_from
        self.train_smoothcvr_cache_to = train_smoothcvr_cache_to
        self.val_smoothcvr_cache_from = val_smoothcvr_cache_from
        self.val_smoothcvr_cache_to = val_smoothcvr_cache_to
        self.test_smoothcvr_cache_from = test_smoothcvr_cache_from
        self.test_smoothcvr_cache_to = test_smoothcvr_cache_to
        self.add_lgbm_fts_from_saved_model_count = add_lgbm_fts_from_saved_model_count
        self.use_hourly_alpha_beta = use_hourly_alpha_beta
        self.add_lgbm_fts_from_saved_model_filename = add_lgbm_fts_from_saved_model_filename
        self.add_lgbm_fts_from_saved_model_predictors_pickle_filename = \
            add_lgbm_fts_from_saved_model_predictors_pickle_filename
        self.lgbm_stacking_val_from = lgbm_stacking_val_from
        self.lgbm_stacking_val_to = lgbm_stacking_val_to
        self.new_lib_ffm_output = new_lib_ffm_output
        self.use_hour_group = use_hour_group
        self.add_n_min_as_hour = add_n_min_as_hour
        self.auto_type_cast = auto_type_cast
        self.dump_train_data = dump_train_data
        self.use_neg_sample = use_neg_sample
        self.neg_sample_seed = neg_sample_seed
        self.use_scvr_cache_file = use_scvr_cache_file
        self.ft_search_op = ft_search_op
        self.submit_prediction = submit_prediction
        self.val_filter_test_hours = val_filter_test_hours
        self.lgbm_seed = lgbm_seed
        self.lgbm_seed_test_list = lgbm_seed_test_list
        self.test_important_fts = test_important_fts




train_config_103_11 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_103_12 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_stnc
                                   )

train_config_103_13 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_varnc,
                                   use_ft_cache=True
                                   )

train_config_103_14 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_next_n_click,
                                   use_ft_cache=False
                                   )

train_config_103_15 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                   qcut = 0.98
                                   )

train_config_103_16 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_no_app,
                                   use_ft_cache=False
                                   )

train_config_103_20 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                   add_second_ft=True
                                   )

train_config_103_20 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                   add_second_ft=True
                                   )


train_config_103_21 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lagbm_params_from_115_0,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_103_22 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_with_test_supplement',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_103_23 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_103_26 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_9_3pm,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict_with_test_supplement',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_103_27 = ConfigScheme(False, False, False,
                               None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_early_300,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_9_3pm,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict_with_test_supplement',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_119 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter_1_to_6,
                                 None,
                                 lgbm_params={**new_lgbm_params, **{'num_boost_round':20}},
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_lgbm_fts',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )


train_config_120 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter_1_to_6,
                                 None,
                                 lgbm_params={**new_lgbm_params, **{'num_boost_round':20}},
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                use_lgbm_fts=True
                                   )


train_config_120_2 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter_1_to_6,
                                 None,
                                 lgbm_params={**new_lgbm_params, **{'num_boost_round':30}},
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False,
                                use_lgbm_fts=True,
                                  sync_mode=True
                                   )


train_config_116 = ConfigScheme(False, False, False,
                                None,
                                shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )

train_config_116_3 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )

train_config_116_4 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                  discretization=50,
                                  )


train_config_116_5 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                new_predict=True,
                                  wordbatch_model='FTRL',
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )


train_config_116_6 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model',
                                  wordbatch_model='NN_ReLU_H1',
                                  new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )

train_config_117_1 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_117_3 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_117_4 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_from_pub_ftrl,
                                   use_ft_cache=False
                                   )

train_config_117_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_8_3pm,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                discretization=50,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_117_6 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                log_discretization=True,
                                 run_theme='ffm_data_gen_seperately',
                                  new_predict=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_117_8 = copy.deepcopy(train_config_117_3)
train_config_117_8.log_discretization = True



train_config_117_9 = copy.deepcopy(train_config_117_8)
train_config_117_9.add_lgbm_fts_from_saved_model = True
train_config_117_9.add_features_list = add_features_list_origin_no_channel_next_click_days

train_config_117_10 = copy.deepcopy(train_config_117_8)
train_config_117_10.add_lgbm_fts_from_saved_model = True
train_config_117_10.add_features_list = add_features_list_origin_no_channel_next_click_days
train_config_117_10.add_lgbm_fts_from_saved_model_count = 7


train_config_117_11 = copy.deepcopy(train_config_117_8)
train_config_117_11.add_lgbm_fts_from_saved_model = True
train_config_117_11.add_features_list = add_features_list_origin_no_channel_next_click_days
train_config_117_11.add_lgbm_fts_from_saved_model_count = 7
train_config_117_11.add_lgbm_fts_from_saved_model_filename = 'train_config_124_33_model.txt'
train_config_117_11.add_lgbm_fts_from_saved_model_predictors_pickle_filename = \
    'train_config_124_33_model_predictors.pickle'


train_config_117_12 = copy.deepcopy(train_config_117_8)
train_config_117_12.add_features_list = add_features_list_smooth_cvr
train_config_117_12.train_smoothcvr_cache_from = id_8_4am
train_config_117_12.train_smoothcvr_cache_to = id_8_3pm
train_config_117_12.test_smoothcvr_cache_from = id_9_4am
train_config_117_12.test_smoothcvr_cache_to = id_9_3pm
train_config_117_12.use_interactive_features = True
train_config_117_12.train_to = id_9_3pm_reserve_last_250w

train_config_117_12.val_filter = None
train_config_117_12.val_from = id_9_3pm_reserve_last_250w
train_config_117_12.val_to = id_9_3pm

train_config_117_13 = copy.deepcopy(train_config_117_8)
train_config_117_13.add_features_list = add_features_list_smooth_cvr
train_config_117_13.train_smoothcvr_cache_from = id_8_4am
train_config_117_13.train_smoothcvr_cache_to = id_8_3pm
train_config_117_13.test_smoothcvr_cache_from = id_9_4am
train_config_117_13.test_smoothcvr_cache_to = id_9_3pm
train_config_117_13.use_interactive_features = True
train_config_117_13.train_to = id_9_3pm_reserve_last_250w

train_config_117_13.val_filter = None
train_config_117_13.val_from = id_9_3pm_reserve_last_250w
train_config_117_13.val_to = id_9_3pm
train_config_117_13.new_lib_ffm_output = True
train_config_117_13.log_discretization = False



train_config_121_1 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_3,
                                 shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_121_2 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_10,
                                 shuffle_sample_filter_1_to_10,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_121_5 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_3,
                                 shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_121_6 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_3,
                                 shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )
train_config_122 = ConfigScheme(False, False, False,
                                None,
                                  shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='online_model_ffm',
                                new_predict=True,
                                use_interactive_features=True,
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 log_discretization=True
                                  )


train_config_124 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )



train_config_124_1 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=[id_8_4am,id_9_4am],
                                 train_to=[id_8_3pm, id_9_3pm],
                                 val_from=id_7_4am,
                                 val_to=id_7_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )

train_config_124_2 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=[id_7_4am, id_8_4am,id_9_4am],
                                 train_to=[id_7_3pm, id_8_3pm, id_9_3pm],
                                 val_from=1,
                                 val_to=id_7_4am - 1,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )
train_config_124_3 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )

train_config_124_4 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                  normalization=True
                                   )


train_config_124_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_early_300,
                                 new_predict= True,
                                 train_from=id_7_3pm,
                                 train_to=id_9_3pm,
                                 val_from=id_9_3pm,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )

train_config_124_6 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_early_300,
                                 new_predict= True,
                                 train_from=id_8_4am,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )


train_config_124_9 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_from_pub_ftrl
                                   )



train_config_125 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_10mincvr,
                                add_10min_ft=True
                                   )
train_config_125_4 = ConfigScheme(False, False, False,
                                None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_10mincvr,
                                add_10min_ft=True
                                   )


train_config_124_7 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4am,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )

train_config_124_8 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days,
                                 pick_hours_weighted = True
                                   )

train_config_124_10 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_iter_600,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days,
                                 pick_hours_weighted = True
                                   )

train_config_124_11 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_add_best_nunique
                                   )

train_config_124_12 = train_config_124_10

train_config_124_14 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params_feature_fraction,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_add_best_nunique
                                   )

train_config_124_16 = ConfigScheme(False, False, False,
                                 None,
                                 None,
                                 None,
                                 lgbm_params=new_lgbm_params_early_415,
                                 new_predict= True,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days,
                                 pick_hours_weighted = True
                                   )

train_config_124_17 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_7_3pm,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click_days
                                   )
train_config_124_18 = copy.deepcopy(train_config_124_3)
train_config_124_18.adversial_val_weighted = True

train_config_124_19 = copy.deepcopy(train_config_124)
train_config_124_19.lgbm_params = lgbm_params_search_128_114

train_config_124_20 = copy.deepcopy(train_config_124_17)
train_config_124_20.train_from = 0
train_config_124_20.train_to = id_7_3pm

train_config_124_21 = copy.deepcopy(train_config_124_3)
train_config_124_21.adversial_val_ft = True


train_config_124_22 = copy.deepcopy(train_config_124)
train_config_124_22.adversial_val_weighted = True


train_config_124_23 = copy.deepcopy(train_config_124)
train_config_124_23.add_in_test_frequent_dimensions = ['channel']
train_config_124_23.add_features_list = add_features_list_origin_no_channel_next_click_ip_freq_ch



train_config_124_25 = copy.deepcopy(train_config_124_23)
train_config_124_25.adversial_val_weighted = True

train_config_124_26 = copy.deepcopy(train_config_124)
train_config_124_26.add_features_list = add_features_list_origin_no_channel_next_click_best_ct_nu_from_search


train_config_124_28 = copy.deepcopy(train_config_124)
train_config_124_28.add_features_list = add_features_list_origin_no_channel_next_click_best_ct_nu_from_search_28


train_config_124_29 = copy.deepcopy(train_config_124)
train_config_124_29.add_features_list = add_features_list_smooth_cvr

train_config_124_30 = copy.deepcopy(train_config_124)
train_config_124_30.add_features_list = add_features_list_smooth_cvr
train_config_124_30.train_smoothcvr_cache_from = id_8_4am
train_config_124_30.train_smoothcvr_cache_to = id_8_3pm
train_config_124_30.test_smoothcvr_cache_from = id_9_4am
train_config_124_30.test_smoothcvr_cache_to = id_9_3pm

train_config_124_31 = copy.deepcopy(train_config_124)
train_config_124_31.lgbm_params = lgbm_params_search_128_610

train_config_124_33 = copy.deepcopy(train_config_124)
train_config_124_33.add_features_list = []


train_config_124_35 = copy.deepcopy(train_config_124)
train_config_124_35.add_features_list = add_features_list_smooth_cvr
train_config_124_35.train_smoothcvr_cache_from = id_8_4am
train_config_124_35.train_smoothcvr_cache_to = id_8_3pm
train_config_124_35.test_smoothcvr_cache_from = id_9_4am
train_config_124_35.test_smoothcvr_cache_to = id_9_3pm
train_config_124_35.use_hourly_alpha_beta = True


train_config_124_36 = copy.deepcopy(train_config_124_35)
train_config_124_36.add_features_list = add_features_list_smooth_cvr_from_search_121_13


train_config_124_37 = copy.deepcopy(train_config_124_36)
train_config_124_37.add_features_list = add_features_list_smooth_cvr_from_search_121_13_reduced


train_config_124_38 = copy.deepcopy(train_config_124)
train_config_124_38.train_from = id_7_0am
train_config_124_38.train_to = id_9_0am
train_config_124_38.val_from = id_9_4am
train_config_124_38.val_to = id_9_3pm

train_config_124_39 = copy.deepcopy(train_config_124)
train_config_124_39.train_from = id_8_0am
train_config_124_39.train_to = id_9_0am
train_config_124_39.val_from = id_9_4am
train_config_124_39.val_to = id_9_3pm

train_config_124_40 = copy.deepcopy(train_config_124_3)
train_config_124_40.train_to = id_9_3pm_reserve_last_250w
train_config_124_40.lgbm_stacking_val_from = id_9_3pm_reserve_last_250w
train_config_124_40.lgbm_stacking_val_to = id_9_3pm


train_config_124_41 = copy.deepcopy(train_config_124)
train_config_124_41.train_from = id_7_0am
train_config_124_41.train_to = id_9_0am
train_config_124_41.val_from = id_9_4am
train_config_124_41.val_to = id_9_3pm
train_config_124_41.use_hour_group = 3

train_config_124_42 = copy.deepcopy(train_config_124_3)
train_config_124_42.add_n_min_as_hour = 15

train_config_124_43 = copy.deepcopy(train_config_124)
train_config_124_43.train_from = id_7_0am
train_config_124_43.train_to = id_9_0am
train_config_124_43.val_from = id_9_4am
train_config_124_43.val_to = id_9_3pm
train_config_124_43.use_hour_group = 6


train_config_124_44 = copy.deepcopy(train_config_124_5)
train_config_124_44.lgbm_params = new_lgbm_params
train_config_124_44.add_features_list = add_features_list_origin_no_channel_next_click_no_day

train_config_124_45 = ConfigScheme(False, False, False,
                                 None,
                                 None,
                                 None,
                                 lgbm_params=new_lgbm_params_early_50,
                                 new_predict= True,
                                 train_from=0,
                                 train_to=id_9_4pm_reserve_last_250w,
                                 val_from=id_9_4pm_reserve_last_250w,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_no_day
                                   )


train_config_124_45 = ConfigScheme(False, False, False,
                                 None,
                                 None,
                                 None,
                                 lgbm_params=new_lgbm_params_early_50,
                                 new_predict= True,
                                 train_from=0,
                                 train_to=id_9_4pm_reserve_last_250w,
                                 val_from=id_9_4pm_reserve_last_250w,
                                 val_to=id_9_4pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_no_day_scvr
                                   )


train_config_126_1 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_2 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=[id_7_4am, id_8_4am,id_9_4am],
                                 train_to=[id_7_3pm, id_8_3pm, id_9_3pm],
                                 val_from=1,
                                 val_to=id_7_4am - 1,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_3 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_3pm,
                                 train_to=id_9_4pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_4 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_from_pub_ftrl
                                   )

train_config_126_5 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_7_3pm,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )

train_config_126_6 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_7_3pm,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                 pick_hours_weighted = True
                                  )

train_config_126_11 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params_feature_fraction,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )
train_config_126_12 = copy.deepcopy(train_config_126_6)
train_config_126_12.val_from = id_7_4am
train_config_126_12.val_to = id_7_3pm


train_config_126_13 = copy.deepcopy(train_config_126_1)
train_config_126_13.add_features_list = add_features_list_pub_asraful_kernel


train_config_126_14 = copy.deepcopy(train_config_126_1)
train_config_126_14.add_features_list = add_features_list_pub_asraful_kernel
train_config_126_14.lgbm_params =  lgbm_params_pub_asraful_kernel

train_config_126_15 = copy.deepcopy(train_config_126_1)
train_config_126_15.add_features_list = add_features_list_smooth_cvr
#train_config_126_15.train_smoothcvr_cache_from = id_7_4am
#train_config_126_15.train_smoothcvr_cache_to = id_7_3pm

train_config_126_16 = copy.deepcopy(train_config_126_1)
train_config_126_16.add_features_list = add_features_list_smooth_cvr
train_config_126_16.use_hourly_alpha_beta = True
train_config_126_16.train_smoothcvr_cache_from = id_7_4am
train_config_126_16.train_smoothcvr_cache_to = id_7_3pm


train_config_126_17 = copy.deepcopy(train_config_126_1)
train_config_126_17.add_features_list = add_features_list_fts_search
train_config_126_17.use_hourly_alpha_beta = True
#train_config_126_17.train_smoothcvr_cache_from = 0
#train_config_126_17.train_smoothcvr_cache_to = id_7_4am
train_config_126_17.train_from = id_7_4am
train_config_126_17.train_to = id_8_3pm
train_config_126_17.train_filter = random_sample_filter_0_2
train_config_126_17.val_filter = random_sample_filter_0_2

train_config_126_18 = copy.deepcopy(train_config_126_1)
train_config_126_18.add_features_list = add_features_list_pub_entire_set
train_config_126_18.use_hourly_alpha_beta = True
#train_config_126_17.train_smoothcvr_cache_from = 0
#train_config_126_17.train_smoothcvr_cache_to = id_7_4am
train_config_126_18.train_from = id_7_4am
train_config_126_18.train_to = id_8_3pm
train_config_126_18.train_filter = random_sample_filter_0_2
train_config_126_18.val_filter = random_sample_filter_0_2

train_config_126_19 = copy.deepcopy(train_config_126_18)
train_config_126_19.add_features_list = add_features_list_fts_search_reduced_gain

train_config_126_20 = copy.deepcopy(train_config_126_18)
train_config_126_20.add_features_list = add_features_list_fts_search_reduced_split

train_config_126_21 = copy.deepcopy(train_config_126_18)
train_config_126_21.add_features_list = add_features_list_origin_no_channel_next_click

train_config_126_22 = copy.deepcopy(train_config_126_18)
train_config_126_22.add_features_list = add_features_list_origin_no_channel_next_click_no_day

train_config_126_23 = copy.deepcopy(train_config_126_22)
train_config_126_23.train_from = 0

train_config_126_24 = copy.deepcopy(train_config_126_20)
train_config_126_24.train_from = 0


train_config_126_25 = copy.deepcopy(train_config_126_18)
train_config_126_25.train_from = 0

train_config_126_26 = copy.deepcopy(train_config_126_23)
train_config_126_26.run_theme = 'train_and_predict'


train_config_126_27 = copy.deepcopy(train_config_126_24)
train_config_126_27.run_theme = 'train_and_predict'


train_config_126_28 = copy.deepcopy(train_config_126_25)
train_config_126_28.run_theme = 'train_and_predict'


train_config_126_29 = copy.deepcopy(train_config_126_18)
train_config_126_29.add_features_list = add_features_list_origin_no_channel_next_click
train_config_126_29.train_from = 0


train_config_126_30 = copy.deepcopy(train_config_126_25)
train_config_126_30.lgbm_params = lgbm_params_pub_entire_set


train_config_126_31 = copy.deepcopy(train_config_126_30)
train_config_126_31.run_theme = 'train_and_predict'


train_config_126_33 = copy.deepcopy(train_config_126_18)
train_config_126_33.add_features_list = add_features_list_fts_search_reduced_split
train_config_126_33.val_filter=random_sample_filter_0_5



train_config_126_34 = copy.deepcopy(train_config_126_27)
train_config_126_34.val_filter=random_sample_filter_0_5

train_config_126_35 = copy.deepcopy(train_config_126_26)
train_config_126_35.val_filter=random_sample_filter_0_5



train_config_121_7 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_121_8 = copy.deepcopy(train_config_121_7)
train_config_121_9 = copy.deepcopy(train_config_121_7)
train_config_121_10 = copy.deepcopy(train_config_121_7)
train_config_121_10.lgbm_params=lgbm_params_l1

train_config_121_11 = copy.deepcopy(train_config_121_7)
train_config_121_11.lgbm_params=lgbm_params_l1
train_config_121_11.adversial_val_weighted = True

train_config_121_12 = copy.deepcopy(train_config_121_8)
train_config_121_12.lgbm_params=lgbm_params_l1
train_config_121_12.adversial_val_weighted = True

train_config_121_13 = copy.deepcopy(train_config_121_8)

train_config_121_13.train_smoothcvr_cache_from = id_7_4am
train_config_121_13.train_smoothcvr_cache_to = id_7_3pm

train_config_126_9 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params_100_cat_smooth,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                   )


train_config_128 = ConfigScheme(False, False, False,
                                  random_sample_filter_0_5,
                                 random_sample_filter_0_5,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='lgbm_params_search',
                                 add_features_list=add_features_list_origin_no_channel_next_click,
                                   use_ft_cache=False
                                   )

train_config_131_1 = copy.deepcopy(train_config_117_8)
train_config_131_1.val_filter = None
train_config_131_1.new_predict = False
train_config_131_1.val_from = id_7_4am
train_config_131_1.val_to = id_7_3pm



train_config_131_3 = copy.deepcopy(train_config_124_3)
train_config_131_3.new_predict = False
train_config_131_3.lgbm_stacking_val_from = id_7_4am
train_config_131_3.lgbm_stacking_val_to = id_7_3pm


train_config_131_4 = copy.deepcopy(train_config_124_9)
train_config_131_4.new_predict = False
train_config_131_4.lgbm_stacking_val_from = id_7_4am
train_config_131_4.lgbm_stacking_val_to = id_7_3pm


train_config_131_5 = copy.deepcopy(train_config_124_11)
train_config_131_5.new_predict = False
train_config_131_5.lgbm_stacking_val_from = id_7_4am
train_config_131_5.lgbm_stacking_val_to = id_7_3pm


train_config_131_6 = copy.deepcopy(train_config_124_17)
train_config_131_6.new_predict = False
train_config_131_6.lgbm_stacking_val_from = id_7_4am
train_config_131_6.lgbm_stacking_val_to = id_7_3pm


train_config_131_7 = copy.deepcopy(train_config_124_37)
train_config_131_7.new_predict = False
train_config_131_7.lgbm_stacking_val_from = id_7_4am
train_config_131_7.lgbm_stacking_val_to = id_7_3pm


train_config_131_8 = copy.deepcopy(train_config_117_11)
train_config_131_8.val_filter = None
train_config_131_8.new_predict = False
train_config_131_8.val_from = id_7_4am
train_config_131_8.val_to = id_7_3pm



train_config_133_78_baseline = ConfigScheme(False, False, False,
                                  None,
                                 None,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= False,
                                 train_from=id_7_0am,
                                 train_to=id_9_0am,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_no_channel_next_click_no_day
                                   )
train_config_133_1 = copy.deepcopy(train_config_133_78_baseline)
train_config_133_1.train_from = id_8_4am
train_config_133_1.train_to = id_8_3pm

train_config_133_2 = copy.deepcopy(train_config_133_1)
train_config_133_2.add_features_list = add_features_list_fts_search
train_config_133_2.lgbm_params = lgbm_params_pub_entire_set
train_config_133_2.auto_type_cast = True
train_config_133_2.use_ft_cache = True

#train_config_133_2.dump_train_data = True

train_config_133_3 = copy.deepcopy(train_config_133_1)
train_config_133_3.add_features_list = add_features_list_fts_search_reduced_split_scvr

train_config_133_4 = copy.deepcopy(train_config_133_78_baseline)
train_config_133_4.add_features_list = add_features_list_origin_no_channel_next_click_no_day_scvr
train_config_133_4.train_smoothcvr_cache_from = 0
train_config_133_4.train_smoothcvr_cache_to = id_7_0am
train_config_133_4.val_smoothcvr_cache_from = id_7_0am
train_config_133_4.val_smoothcvr_cache_to = id_9_0am

train_config_133_5 = copy.deepcopy(train_config_133_1)
train_config_133_5.train_from = [id_7_4am,id_8_4am]
train_config_133_5.train_to = [id_7_3pm,id_8_3pm]

train_config_133_6 = copy.deepcopy(train_config_133_2)
train_config_133_6.train_from = [id_7_4am,id_8_4am]
train_config_133_6.train_to = [id_7_3pm,id_8_3pm]
train_config_133_6.use_ft_cache = True

train_config_133_7 = copy.deepcopy(train_config_133_4)
train_config_133_7.lgbm_params = lgbm_params_pub_entire_set
train_config_133_7.auto_type_cast = True
train_config_133_7.use_ft_cache = True


train_config_133_9 = copy.deepcopy(train_config_133_6)
train_config_133_9.use_neg_sample = False

train_config_133_10 = copy.deepcopy(train_config_133_6)
train_config_133_10.use_neg_sample = True


train_config_133_11 = copy.deepcopy(train_config_133_6)
train_config_133_11.use_neg_sample = True
train_config_133_11.add_features_list= add_features_list_origin_no_channel_next_click_no_day


train_config_133_12 = copy.deepcopy(train_config_133_6)
train_config_133_12.use_neg_sample = True
train_config_133_12.add_features_list = add_features_list_fts_search_reduced_split_scvr

train_config_133_10_4 = copy.deepcopy(train_config_133_6)
train_config_133_10_4.use_neg_sample = True
train_config_133_10_4.neg_sample_seed = 666


train_config_133_11_2 = copy.deepcopy(train_config_133_11)
train_config_133_11_2.neg_sample_seed = 666


train_config_133_12_2 = copy.deepcopy(train_config_133_12)
train_config_133_12_2.neg_sample_seed = 666

train_config_133_13 = copy.deepcopy(train_config_133_4)
train_config_133_13.train_from = [id_7_4am,id_8_4am]
train_config_133_13.train_to = [id_7_3pm,id_8_3pm]
train_config_133_13.use_neg_sample = True
train_config_133_13.use_ft_cache = True

train_config_133_13_2 = copy.deepcopy(train_config_133_13)
train_config_133_13_2.neg_sample_seed = 666

train_config_133_14 = copy.deepcopy(train_config_133_12)
train_config_133_14.add_features_list = add_features_list_fts_search_reduced_split_scvr_add_var

train_config_133_15 = copy.deepcopy(train_config_133_12)
train_config_133_15.train_smoothcvr_cache_from = 0
train_config_133_15.train_smoothcvr_cache_to = id_7_0am
train_config_133_15.val_smoothcvr_cache_from = id_7_0am
train_config_133_15.val_smoothcvr_cache_to = id_9_0am

train_config_133_16 = copy.deepcopy(train_config_133_14)
train_config_133_16.add_features_list = add_features_list_fts_search_reduced_split_scvr_add_var_only_1

train_config_133_17 = copy.deepcopy(train_config_133_15)
train_config_133_17.add_features_list = add_features_list_fts_search_reduced_split_scvr_only_1

train_config_133_18 = copy.deepcopy(train_config_133_12)
train_config_133_18.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1

train_config_133_19 = copy.deepcopy(train_config_133_15)
train_config_133_19.use_scvr_cache_file = True

train_config_133_20 = copy.deepcopy(train_config_133_15)
train_config_133_20.use_scvr_cache_file = True
train_config_133_20.add_features_list = add_features_list_fts_search_reduced_split
train_config_133_20.run_theme = 'train_and_predict_ft_search'

train_config_133_21 = copy.deepcopy(train_config_133_12)
train_config_133_21.add_features_list = add_features_list_fts_search_reduced_split
train_config_133_21.run_theme = 'train_and_predict_ft_search'
train_config_133_21.ft_search_op = 'cumcount'

train_config_133_22 = copy.deepcopy(train_config_133_21)
train_config_133_22.ft_search_op = 'var'

train_config_133_23 = copy.deepcopy(train_config_133_21)
train_config_133_23.ft_search_op = 'mean'


train_config_133_24 = copy.deepcopy(train_config_133_18)
train_config_133_24.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1_var_best


train_config_133_25 = copy.deepcopy(train_config_133_18)
train_config_133_25.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1_scvr_best

train_config_133_26 = copy.deepcopy(train_config_133_15)
train_config_133_26.use_scvr_cache_file = True
train_config_133_26.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_26.run_theme = 'train_and_predict_ft_search'


train_config_133_27 = copy.deepcopy(train_config_133_12)
train_config_133_27.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_27.run_theme = 'train_and_predict_ft_search'
train_config_133_27.ft_search_op = 'cumcount'

train_config_133_28 = copy.deepcopy(train_config_133_15)
train_config_133_28.use_scvr_cache_file = True
train_config_133_28.add_features_list = add_features_list_search_28

train_config_133_29 = copy.deepcopy(train_config_133_12)
train_config_133_29.add_features_list = add_features_list_search_29

train_config_133_30 = copy.deepcopy(train_config_133_12)
train_config_133_30.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_30.run_theme = 'train_and_predict_ft_search'
train_config_133_30.ft_search_op = 'mean'

train_config_133_31 = copy.deepcopy(train_config_133_12)
train_config_133_31.add_features_list = add_features_list_search_31
train_config_133_31.neg_sample_seed = 666

train_config_133_32 = copy.deepcopy(train_config_133_18)
train_config_133_32.new_predict = True
train_config_133_32.run_theme = 'train_and_predict'
train_config_133_32.train_from = [id_8_4am,id_9_4am]
train_config_133_32.train_to = [id_8_3pm,id_9_3pm]
train_config_133_32.val_from = id_9_3pm
train_config_133_32.val_to = id_9_3pm + 10000
train_config_133_32.lgbm_params = lgbm_params_pub_entire_set_no_early_iter_153

train_config_133_33 = copy.deepcopy(train_config_133_32)
train_config_133_33.add_features_list = add_features_list_33
train_config_133_33.lgbm_params = lgbm_params_pub_entire_set_no_early_iter_205
train_config_133_33.submit_prediction = True

train_config_133_34 = copy.deepcopy(train_config_133_32)
train_config_133_34.add_features_list = add_features_list_34
train_config_133_34.lgbm_params = lgbm_params_pub_entire_set_no_early_iter_178
train_config_133_34.submit_prediction = True

train_config_133_35 = copy.deepcopy(train_config_133_32)
train_config_133_35.add_features_list = add_features_list_35
train_config_133_35.lgbm_params = lgbm_params_pub_entire_set_no_early_iter_154
train_config_133_35.submit_prediction = True

train_config_133_36 = copy.deepcopy(train_config_133_18)
train_config_133_36.val_filter_test_hours = True

train_config_133_37 = copy.deepcopy(train_config_133_18)
train_config_133_37.add_features_list = add_features_list_34
train_config_133_37.val_filter_test_hours = True


train_config_133_38 = copy.deepcopy(train_config_133_18)
train_config_133_38.add_features_list = add_features_list_33
train_config_133_38.val_filter_test_hours = True

train_config_133_39 = copy.deepcopy(train_config_133_38)
train_config_133_39.lgbm_seed = 999

train_config_133_40 = copy.deepcopy(train_config_133_38)
train_config_133_40.lgbm_seed = 555

train_config_133_41 = copy.deepcopy(train_config_133_38)
train_config_133_41.lgbm_seed = 333

train_config_133_42 = copy.deepcopy(train_config_133_38)
train_config_133_42.lgbm_seed_test_list = [999, 555, 333]
train_config_133_42.lgbm_params = ({**train_config_133_42.lgbm_params,
                                    **{'subsample': 0.9}})


train_config_133_43 = copy.deepcopy(train_config_133_38)
train_config_133_43.lgbm_seed_test_list = [999, 555, 333]
train_config_133_43.lgbm_params = ({**train_config_133_43.lgbm_params,
                                    **{'subsample': 0.95}})



train_config_133_44 = copy.deepcopy(train_config_133_38)
train_config_133_44.lgbm_seed_test_list = [999, 555, 333]
train_config_133_44.lgbm_params = ({**train_config_133_44.lgbm_params,
                                    **{'subsample': 0.9, 'colsample_bytree':0.95}})


train_config_133_45 = copy.deepcopy(train_config_133_38)
train_config_133_45.lgbm_seed_test_list = [999, 555, 333]
train_config_133_45.lgbm_params = ({**train_config_133_45.lgbm_params,
                                    **{'subsample': 0.95, 'colsample_bytree':0.95}})




train_config_133_46 = copy.deepcopy(train_config_133_38)
train_config_133_46.lgbm_seed_test_list = [999, 555, 333]
train_config_133_46.lgbm_params = ({**train_config_133_46.lgbm_params,
                                    **{'subsample': 1.0, 'colsample_bytree':1.0}})


train_config_133_47 = copy.deepcopy(train_config_133_18)
train_config_133_47.val_filter_test_hours = True
train_config_133_47.lgbm_params = ({**train_config_133_47.lgbm_params,
                                    **{'subsample': 1.0, 'colsample_bytree':1.0}})


train_config_133_48 = copy.deepcopy(train_config_133_18)
train_config_133_48.val_filter_test_hours = True
train_config_133_48.lgbm_params = ({**train_config_133_48.lgbm_params,
                                    **{'subsample': 1.0, 'colsample_bytree':1.0}})
train_config_133_48.add_features_list = add_features_list_33


train_config_133_49 = copy.deepcopy(train_config_133_18)
train_config_133_49.val_filter_test_hours = True
train_config_133_49.lgbm_params = ({**train_config_133_49.lgbm_params,
                                    **{'subsample': 1.0, 'colsample_bytree':1.0}})
train_config_133_49.add_features_list = add_features_list_34
train_config_133_49.train_smoothcvr_cache_from = 0
train_config_133_49.train_smoothcvr_cache_to = id_7_0am
train_config_133_49.val_smoothcvr_cache_from = id_7_0am
train_config_133_49.val_smoothcvr_cache_to = id_9_0am

train_config_133_50 = copy.deepcopy(train_config_133_18)
train_config_133_50.val_filter_test_hours = True
train_config_133_50.lgbm_params = ({**train_config_133_50.lgbm_params,
                                    **{'subsample': 1.0, 'colsample_bytree':1.0}})
train_config_133_50.add_features_list = add_features_list_35


train_config_133_51 = copy.deepcopy(train_config_133_32)
train_config_133_51.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_51.lgbm_params = ({**train_config_133_51.lgbm_params,
                                    **{'subsample': 1.0,
                                       'colsample_bytree':1.0,
                                        'early_stopping_round': 222,
                                        'num_boost_round': 222,
                                       }})
train_config_133_51.submit_prediction = True

train_config_133_52 = copy.deepcopy(train_config_133_32)
train_config_133_52.add_features_list = add_features_list_33
train_config_133_52.lgbm_params = ({**train_config_133_52.lgbm_params,
                                    **{'subsample': 1.0,
                                       'colsample_bytree':1.0,
                                        'early_stopping_round': 169,
                                        'num_boost_round': 169,
                                       }})
train_config_133_52.submit_prediction = True


train_config_133_53 = copy.deepcopy(train_config_133_32)
train_config_133_53.add_features_list = add_features_list_34
train_config_133_53.lgbm_params = ({**train_config_133_53.lgbm_params,
                                    **{'subsample': 1.0,
                                       'colsample_bytree':1.0,
                                        'early_stopping_round': 157,
                                        'num_boost_round': 157,
                                       }})
train_config_133_53.submit_prediction = True


train_config_133_54 = copy.deepcopy(train_config_133_32)
train_config_133_54.add_features_list = add_features_list_35
train_config_133_54.lgbm_params = ({**train_config_133_54.lgbm_params,
                                    **{'subsample': 1.0,
                                       'colsample_bytree':1.0,
                                        'early_stopping_round': 204,
                                        'num_boost_round': 204,
                                       }})
train_config_133_54.submit_prediction = True


train_config_133_55 = copy.deepcopy(train_config_133_18)
train_config_133_55.val_filter_test_hours = True
train_config_133_55.lgbm_params = ({**train_config_133_55.lgbm_params,
                                    **{'subsample': 1.0, 'colsample_bytree':1.0}})
train_config_133_55.add_features_list = add_features_list_fts_search


train_config_133_56 = copy.deepcopy(train_config_133_18)
train_config_133_56.val_filter_test_hours = True
train_config_133_56.lgbm_params = ({**train_config_133_56.lgbm_params,
                                    **{'subsample': 1.0,
                                       'colsample_bytree':1.0,
                                        'early_stopping_round': 74,
                                        'num_boost_round': 74,
                                    }})
train_config_133_56.new_predict = True
train_config_133_56.run_theme = 'train_and_predict'
train_config_133_56.train_from = [id_8_4am,id_9_4am]
train_config_133_56.train_to = [id_8_3pm,id_9_3pm]
train_config_133_56.val_from = id_9_3pm
train_config_133_56.val_to = id_9_3pm + 10000
train_config_133_56.add_features_list = add_features_list_fts_search
train_config_133_56.submit_prediction = True

train_config_133_57 = copy.deepcopy(train_config_133_55)
train_config_133_57.add_features_list = add_features_list_fts_search_reduced_according_55

train_config_133_58 = copy.deepcopy(train_config_133_55)
train_config_133_58.lgbm_params = [
    lgbm_params_pub_entire_set_test_early_stop_50,
    lgbm_params_pub_entire_set_test_early_stop_200,
    lgbm_params_pub_entire_set_test_depth_5,
    lgbm_params_pub_entire_set_test_depth_5_leave_9,
    lgbm_params_pub_entire_set_test_scale_pos_50,
    lgbm_params_pub_entire_set_test_scale_pos_90,
    lgbm_params_pub_entire_set_test_early_stop_400
]

train_config_133_59 = copy.deepcopy(train_config_133_55)
train_config_133_59.lgbm_params = [
    lgbm_params_pub_entire_set_new_test_1,
    lgbm_params_pub_entire_set_new_test_2,
    lgbm_params_pub_entire_set_new_test_3,
    lgbm_params_pub_entire_set_new_test_4,
    lgbm_params_pub_entire_set_new_test_5,
]


train_config_133_60 = copy.deepcopy(train_config_133_55)
train_config_133_60.lgbm_params = [
    lgbm_params_pub_entire_set_new_test_6,
    lgbm_params_pub_entire_set_new_test_7,
    lgbm_params_pub_entire_set_new_test_8,
]
train_config_133_61 = copy.deepcopy(train_config_133_55)
train_config_133_61.lgbm_params = lgbm_params_pub_entire_set_new_test_3
train_config_133_61.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1


train_config_133_62 = copy.deepcopy(train_config_133_61)
train_config_133_62.new_predict = True
train_config_133_62.train_from = [id_8_4am,id_9_4am]
train_config_133_62.train_to = [id_8_3pm,id_9_3pm]
train_config_133_62.val_from = id_9_3pm
train_config_133_62.val_to = id_9_3pm + 10000
train_config_133_62.submit_prediction = True
train_config_133_62.lgbm_params = ({**train_config_133_62.lgbm_params,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round': 284
                                        }})

train_config_133_63 = copy.deepcopy(train_config_133_62)
train_config_133_63.add_features_list = add_features_list_fts_search
train_config_133_63.lgbm_params = ({**train_config_133_63.lgbm_params,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':255
                                       }})

train_config_133_64 = copy.deepcopy(train_config_133_59)
train_config_133_64.add_features_list = add_features_list_fts_search
train_config_133_64.lgbm_params = [
    lgbm_params_pub_entire_set_new_test_1,
    lgbm_params_pub_entire_set_new_test_2,
    lgbm_params_pub_entire_set_new_test_3,
    lgbm_params_pub_entire_set_new_test_4,
    lgbm_params_pub_entire_set_new_test_5,
    lgbm_params_pub_entire_set_new_test_6,
    lgbm_params_pub_entire_set_new_test_7,
    lgbm_params_pub_entire_set_new_test_8,
    lgbm_params_pub_entire_set_test_early_stop_50,
    lgbm_params_pub_entire_set_test_early_stop_200,
    lgbm_params_pub_entire_set_test_depth_5,
    lgbm_params_pub_entire_set_test_depth_5_leave_9,
    lgbm_params_pub_entire_set_test_scale_pos_50,
    lgbm_params_pub_entire_set_test_scale_pos_90,
    lgbm_params_pub_entire_set_test_early_stop_400,
]

train_config_133_65 = copy.deepcopy(train_config_133_64)
train_config_133_65.lgbm_params = lgbm_params_pub_entire_set_new_test_3
train_config_133_65.train_from = id_7_0am
train_config_133_65.train_to = id_9_0am

train_config_133_66 = copy.deepcopy(train_config_133_65)
train_config_133_66.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1


train_config_133_67 = copy.deepcopy(train_config_133_65)
train_config_133_67.new_predict = True
train_config_133_67.train_from = 0
train_config_133_67.train_to = id_all_train
train_config_133_67.val_from = id_9_3pm
train_config_133_67.val_to = id_9_3pm + 10000
train_config_133_67.submit_prediction = True
train_config_133_67.add_features_list = add_features_list_fts_search
train_config_133_67.lgbm_params = ({**train_config_133_67.lgbm_params,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':423
                                       }})
train_config_133_68 = copy.deepcopy(train_config_133_67)
train_config_133_68.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_68.lgbm_params = ({**train_config_133_68.lgbm_params,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':418
                                       }})
train_config_133_69 = copy.deepcopy(train_config_133_65)
train_config_133_69.add_features_list = train_config_133_69.add_features_list + get_cols_com('var')


train_config_133_70 = copy.deepcopy(train_config_133_65)
train_config_133_70.add_features_list = train_config_133_69.add_features_list + get_cols_com('nunique')

train_config_133_71 = copy.deepcopy(train_config_133_70)
train_config_133_71.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1

train_config_133_72 = copy.deepcopy(train_config_133_69)
train_config_133_72.lgbm_params = [
    lgbm_params_pub_entire_set_new_test_1,
    lgbm_params_pub_entire_set_new_test_2,
    lgbm_params_pub_entire_set_new_test_3,
    lgbm_params_pub_entire_set_new_test_4,
    lgbm_params_pub_entire_set_new_test_5,
    lgbm_params_pub_entire_set_new_test_6,
    lgbm_params_pub_entire_set_new_test_7,
    lgbm_params_pub_entire_set_new_test_8,
    lgbm_params_pub_entire_set_test_early_stop_50,
    lgbm_params_pub_entire_set_test_early_stop_200,
    lgbm_params_pub_entire_set_test_depth_5,
    lgbm_params_pub_entire_set_test_depth_5_leave_9,
    lgbm_params_pub_entire_set_test_scale_pos_50,
    lgbm_params_pub_entire_set_test_scale_pos_90,
    lgbm_params_pub_entire_set_test_early_stop_400,
]

train_config_133_73 = copy.deepcopy(train_config_133_71)
train_config_133_73.lgbm_params = ({**train_config_133_73.lgbm_params,
                                    **{ 'early_stopping_round': 50,
                                        'scale_pos_weight':1.0
                                       }})

train_config_133_74 = copy.deepcopy(train_config_133_69)
train_config_133_74.lgbm_params = ({**train_config_133_74.lgbm_params,
                                    **{ 'early_stopping_round': 50,
                                        'scale_pos_weight':1.0
                                       }})
train_config_133_75 = copy.deepcopy(train_config_133_67)
train_config_133_75.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_75.lgbm_params = ({**train_config_133_75.lgbm_params,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':675,
                                        'scale_pos_weight': 1.0
                                        }})

train_config_133_76 = copy.deepcopy(train_config_133_67)
train_config_133_76.add_features_list = train_config_133_69.add_features_list + get_cols_com('var')
train_config_133_76.lgbm_params = ({**train_config_133_76.lgbm_params,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':409,
                                        'scale_pos_weight': 1.0
                                        }})

train_config_133_77 = copy.deepcopy(train_config_133_73)
train_config_133_77.train_from = 0
train_config_133_77.add_features_list = add_features_list_fts_search
train_config_133_77.lgbm_params = ({**train_config_133_77.lgbm_params,
                                    **{ 'early_stopping_round': 50,
                                        'scale_pos_weight':1.0
                                       }})
train_config_133_77.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1


train_config_133_78 = copy.deepcopy(train_config_133_77)
train_config_133_78.add_features_list = add_features_list_fts_search
train_config_133_78.test_important_fts = True

train_config_133_79 = copy.deepcopy(train_config_133_77)
train_config_133_79.add_features_list = train_config_133_79.add_features_list + get_cols_com('var') + get_cols_com('count') + get_cols_com('nunique')

train_config_133_80 = copy.deepcopy(train_config_133_77)
train_config_133_80.add_features_list = train_config_133_80.add_features_list + get_cols_com('mean') + get_cols_com('cumcount') + get_cols_com('smoothcvr')

train_config_133_81 = copy.deepcopy(train_config_133_66)
train_config_133_81.test_important_fts = True
train_config_133_81.add_features_list = train_config_133_81.add_features_list + get_cols_com('var')

train_config_133_82 = copy.deepcopy(train_config_133_66)
train_config_133_82.test_important_fts = True


train_config_133_83 = copy.deepcopy(train_config_133_66)
train_config_133_83.test_important_fts = True
train_config_133_83.add_features_list = train_config_133_83.add_features_list + get_cols_com('mean')

train_config_133_84 = copy.deepcopy(train_config_133_66)
train_config_133_85 = copy.deepcopy(train_config_133_66)
train_config_133_85.add_features_list = add_features_list_origin_no_channel_next_click_no_day

train_config_133_86 = copy.deepcopy(train_config_133_66)
train_config_133_86.add_features_list = train_config_133_86.add_features_list + get_cols_com('var')

train_config_133_87 = copy.deepcopy(train_config_133_86)
train_config_133_87.lgbm_params = [
    lgbm_params_pub_entire_set_new_test_1,
    lgbm_params_pub_entire_set_new_test_2,
    lgbm_params_pub_entire_set_new_test_3,
    lgbm_params_pub_entire_set_new_test_4,
    lgbm_params_pub_entire_set_new_test_5,
    lgbm_params_pub_entire_set_new_test_6,
    lgbm_params_pub_entire_set_new_test_7,
    lgbm_params_pub_entire_set_new_test_8,
    lgbm_params_pub_entire_set_test_early_stop_50,
    lgbm_params_pub_entire_set_test_early_stop_200,
    lgbm_params_pub_entire_set_test_depth_5,
    lgbm_params_pub_entire_set_test_depth_5_leave_9,
    lgbm_params_pub_entire_set_test_scale_pos_50,
    lgbm_params_pub_entire_set_test_scale_pos_90,
    lgbm_params_pub_entire_set_test_early_stop_400,
]

train_config_133_88 = copy.deepcopy(train_config_133_86)
train_config_133_88.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_5_leave_20,
    lgbm_params_pub_entire_set_test_depth_5_leave_15,
    lgbm_params_pub_entire_set_test_depth_4_leave_20,
    lgbm_params_pub_entire_set_test_depth_5_leave_25,
    lgbm_params_pub_entire_set_test_depth_5_leave_30,
    lgbm_params_pub_entire_set_test_depth_5_leave_20_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_30_scale_1
]


train_config_133_89 = copy.deepcopy(train_config_133_86)
train_config_133_89.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_5_leave_25_scale_1,
    lgbm_params_pub_entire_set_test_depth_4_leave_30_scale_1
]

train_config_133_90 = copy.deepcopy(train_config_133_86)
train_config_133_90.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_5_leave_25_scale_1,
    lgbm_params_pub_entire_set_test_depth_4_leave_30_scale_1
]
train_config_133_90.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1

train_config_133_91 = copy.deepcopy(train_config_133_90)
train_config_133_91.lgbm_params = lgbm_params_pub_entire_set_test_depth_5_leave_20_scale_1



train_config_133_92 = copy.deepcopy(train_config_133_65)
train_config_133_92.new_predict = True
train_config_133_92.train_from = 0
train_config_133_92.train_to = id_all_train
train_config_133_92.val_from = id_9_3pm
train_config_133_92.val_to = id_9_3pm + 10000
train_config_133_92.submit_prediction = True
train_config_133_92.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1
train_config_133_92.lgbm_params = ({**lgbm_params_pub_entire_set_test_depth_5_leave_25_scale_1,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':325
                                       }})

train_config_133_93 = copy.deepcopy(train_config_133_92)
train_config_133_93.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1 + get_cols_com('var')
train_config_133_93.lgbm_params = ({**lgbm_params_pub_entire_set_test_depth_5_leave_20_scale_1,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':402
                                       }})
train_config_133_94 = copy.deepcopy(train_config_133_93)
train_config_133_94.lgbm_params = ({**lgbm_params_pub_entire_set_test_depth_5_leave_25_scale_1,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':300
                                       }})


train_config_133_95 = copy.deepcopy(train_config_133_89)
train_config_133_95.add_features_list = add_features_list_fts_search_reduced_split_add_counting_1 + \
                                        get_cols_com('var') + \
                                        get_cols_com('nunique')

train_config_133_95.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_5_leave_25_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_20_scale_1,
    lgbm_params_pub_entire_set_test_depth_4_leave_30_scale_1,
]

train_config_133_96 = copy.deepcopy(train_config_133_95)
train_config_133_96.new_predict = True
train_config_133_96.train_from = 0
train_config_133_96.train_to = id_all_train
train_config_133_96.val_from = id_9_3pm
train_config_133_96.val_to = id_9_3pm + 10000
train_config_133_96.submit_prediction = True
train_config_133_96.lgbm_params = ({**lgbm_params_pub_entire_set_test_depth_5_leave_20_scale_1,
                                    **{ 'early_stopping_round': None,
                                        'num_boost_round':281
                                       }})

train_config_133_97 = copy.deepcopy(train_config_133_95)
train_config_133_97.add_features_list = get_cols_com('count') + \
                                        get_cols_com('cumcount') + \
                                        get_cols_com('smoothcvr') + \
                                        get_cols_com('nunique') + \
                                        [
                                            {'group': ['ip', 'app', 'device', 'os', 'is_attributed'],
                                             'op': 'nextclick'},
                                        ]

                                        # get_cols_com('var') + \
                                        # get_cols_com('mean') + \

train_config_133_97.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_5_leave_25_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_20_scale_1,
    lgbm_params_pub_entire_set_test_depth_4_leave_30_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_30_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_35_scale_1,
]

train_config_133_98 = copy.deepcopy(train_config_133_97)
train_config_133_98.lgbm_params = [
    {**lgbm_params_pub_entire_set_test_depth_5_leave_30_scale_1, **{'learning_rate': 0.05}},
    {**lgbm_params_pub_entire_set_test_depth_5_leave_30_scale_1, **{'learning_rate': 0.02}},
    {**lgbm_params_pub_entire_set_test_depth_5_leave_35_scale_1, **{'learning_rate': 0.05}},
    {**lgbm_params_pub_entire_set_test_depth_5_leave_35_scale_1, **{'learning_rate': 0.02}},
]
train_config_133_99 = copy.deepcopy(train_config_133_97)
train_config_133_99.lgbm_params = [
    lgbm_params_pub_entire_set_new_test_1,
    lgbm_params_pub_entire_set_new_test_2,
    lgbm_params_pub_entire_set_new_test_3,
    lgbm_params_pub_entire_set_new_test_4,
    lgbm_params_pub_entire_set_new_test_5,
    lgbm_params_pub_entire_set_new_test_6,
    lgbm_params_pub_entire_set_new_test_7,
    lgbm_params_pub_entire_set_new_test_8,
    lgbm_params_pub_entire_set_test_early_stop_50,
    lgbm_params_pub_entire_set_test_early_stop_200,
    lgbm_params_pub_entire_set_test_depth_5,
    lgbm_params_pub_entire_set_test_depth_5_leave_9,
    lgbm_params_pub_entire_set_test_scale_pos_50,
    lgbm_params_pub_entire_set_test_scale_pos_90,
    lgbm_params_pub_entire_set_test_early_stop_400,
]
train_config_133_100 = copy.deepcopy(train_config_133_97)
train_config_133_100.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_5_leave_30_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_35_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_50_scale_1,
    lgbm_params_pub_entire_set_test_depth_5_leave_75_scale_1,
    lgbm_params_pub_entire_set_test_depth_6_leave_50_scale_1,
    lgbm_params_pub_entire_set_test_depth_6_leave_75_scale_1,
    lgbm_params_pub_entire_set_test_depth_4_leave_50_scale_1
    ]

train_config_133_101 = copy.deepcopy(train_config_133_97)
train_config_133_101.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_6_leave_50_scale_1,
    lgbm_params_pub_entire_set_test_depth_6_leave_75_scale_1,
    lgbm_params_pub_entire_set_test_depth_7_leave_50_scale_1,
    lgbm_params_pub_entire_set_test_depth_7_leave_75_scale_1,
    lgbm_params_pub_entire_set_test_depth_8_leave_50_scale_1,
    lgbm_params_pub_entire_set_test_depth_8_leave_75_scale_1,
    ]

train_config_133_102 = copy.deepcopy(train_config_133_97)
train_config_133_102.lgbm_params = [
    lgbm_params_pub_entire_set_test_depth_6_leave_30_scale_1,
    lgbm_params_pub_entire_set_test_depth_6_leave_25_scale_1,
    lgbm_params_pub_entire_set_test_depth_7_leave_30_scale_1,
    lgbm_params_pub_entire_set_test_depth_7_leave_25_scale_1,
    lgbm_params_pub_entire_set_test_depth_8_leave_30_scale_1,
    lgbm_params_pub_entire_set_test_depth_8_leave_25_scale_1,
    ]

debug = False

def use_config_scheme(str):
    ret = eval(str)
    if debug:
        ret.train_from = debug_train_from
        ret.train_to = debug_train_to
        ret.val_from=debug_val_from
        ret.val_to=debug_val_to

    logger.debug('deduping the add_feature_list:')
    ret.add_features_list = [json.loads(e) for e in collections.OrderedDict(\
        {json.dumps(r):None for r in ret.add_features_list})]

    logger.info('using config var name and test log: %s', str)
    ret.config_name = str

    # detailed config item handling.....
    if ret.lgbm_seed is not None:
        ret.lgbm_params.update({
            'drop_seed':ret.lgbm_seed,
            'feature_fraction_seed': ret.lgbm_seed,
            'bagging_seed': ret.lgbm_seed, # alias=bagging_fraction_seed
            'data_random_seed': ret.lgbm_seed # random seed for data partition in parallel learning (not include feature parallel)
        })
    if ret.use_ft_cache_from is None:
        ret.use_ft_cache_from = 'cache' #ret.config_name



    logger.info('config values: %s', pprint.pformat(vars(ret)))

    try:
        os.mkdir('./config_backup')
    except:
        None

    fname = './config_backup/%s' % str
    with open(fname,'w') as f:
        f.write(pprint.pformat(vars(ret)))
        logger.info('config dumped to %s', fname)

    return ret


#config_scheme_to_use = use_config_scheme('train_config_133_36')
