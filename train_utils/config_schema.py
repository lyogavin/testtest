import copy
from pprint import pprint

from train_utils.constants import *
from train_utils.model_params import *
from train_utils.features_def import *


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
                 dump_train_data=False
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

debug = False

def use_config_scheme(str):
    ret = eval(str)
    if debug:
        ret.train_from = debug_train_from
        ret.train_to = debug_train_to
        ret.val_from=debug_val_from
        ret.val_to=debug_val_to
    print('using config var name and test log: ', str)
    ret.config_name = str
    if ret.use_ft_cache_from is None:
        ret.use_ft_cache_from = 'cache' #ret.config_name
    print('config values: ')
    pprint(vars(ret))


    return ret


config_scheme_to_use = use_config_scheme('train_config_133_6')
