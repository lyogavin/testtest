train_predict_config_without_val_sample = ConfigScheme(True, True, False, val_filter=None)
train_config = ConfigScheme(False, True, False,
                            train_filter=shuffle_sample_filter,
                            train_start_time = val_time_range_start,
                            train_end_time=val_time_range_end,
                            val_start_time=train_time_range_start,
                            val_end_time=train_time_range_end)


train_config_with_hist_st_only_data_with_hist = ConfigScheme(False, True, False,
                            train_filter=hist_ft_sample_filter,
                            val_filter=hist_ft_sample_filter,
                            test_filter=hist_ft_sample_filter,
                            train_start_time = val_time_range_start,
                            train_end_time=val_time_range_end,
                            val_start_time=train_time_range_start,
                            val_end_time=train_time_range_end,
                            add_hist_statis_fts=True)

train_config1 = ConfigScheme(False, True, False,
                            train_filter=shuffle_sample_filter,
                            train_start_time = val_time_range_start,
                            train_end_time=val_time_range_end,
                            val_start_time=train_time_range_start,
                            val_end_time=train_time_range_end,
                             lgbm_params=new_lgbm_params1
                             )
ffm_data_config = ConfigScheme(False, False, True,shuffle_sample_filter_1_to_10,
                               shuffle_sample_filter_1_to_10,shuffle_sample_filter_1_to_10k,  discretization=100,
                               gen_ffm_test_data=True)

ffm_data_config_no_filter_disc_50  = ConfigScheme(False, False, True,None,
                               None,None,  discretization=50,
                               gen_ffm_test_data=True)
ffm_data_config_disc_50 = ConfigScheme(False, False, True,None,
                               shuffle_sample_filter,None,  discretization=50,
                               gen_ffm_test_data=True)

ffm_data_config_train = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=100,
                                     )

ffm_data_config_train_discretization_75 = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=75,
                                     )
ffm_data_config_train_discretization_50 = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=50,
                                     )

ffm_data_config_train_discretization_25 = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=25,
                                     )

ffm_data_config_train_discretization_30 = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=30,
                                     )

ffm_data_config_train_discretization_20 = ConfigScheme(False, False, True,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter,
                                     shuffle_sample_filter_1_to_10k,
                                     train_start_time = val_time_range_start,
                                     train_end_time=val_time_range_end,
                                     val_start_time=train_time_range_start,
                                     val_end_time=train_time_range_end,
                                     discretization=20,
                                     )

ffm_data_config_mock_test = ConfigScheme(False, False, True,shuffle_sample_filter_1_to_10,
                                         shuffle_sample_filter_1_to_10,shuffle_sample_filter_1_to_10k,
                                         discretization=100,
                                         mock_test_with_val_data_to_test=True)


train_predict_new_lgbm_params_config = ConfigScheme(True, True, False,
                                                    shuffle_sample_filter,
                                                    shuffle_sample_filter,
                                                    None,
                                                    lgbm_params=new_lgbm_params)


train_predict_filter_app_12_config = ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter2,
                                                  val_filter=field_sample_filter_app_filter2,
                                                  test_filter=field_sample_filter_app_filter2)

train_predict_filter_app_18_14_config = ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter3,
                                                     val_filter=field_sample_filter_app_filter3,
                                                     test_filter=field_sample_filter_app_filter3)

train_predict_filter_app_8_11_config = ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter4,
                                                    val_filter=field_sample_filter_app_filter4,
                                                    test_filter=field_sample_filter_app_filter4)

train_predict_filter_channel_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_channel_filter,
                 val_filter=field_sample_filter_channel_filter,
                 test_filter=field_sample_filter_channel_filter                 )


train_predict_filter_app_8_11_new_lgbm_params_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter4,
                 val_filter=field_sample_filter_app_filter4,
                 test_filter=field_sample_filter_app_filter4,
                 lgbm_params=new_lgbm_params
                 )
train_predict_filter_app_12_new_lgbm_params_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter2,
                 val_filter=field_sample_filter_app_filter2,
                 test_filter=field_sample_filter_app_filter2,
                 lgbm_params=new_lgbm_params
                 )
train_predict_filter_app_18_14_new_lgbm_params_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_app_filter3,
                 val_filter=field_sample_filter_app_filter3,
                 test_filter=field_sample_filter_app_filter3,
                 lgbm_params=new_lgbm_params
                 )
train_predict_filter_channel_new_lgbm_params_config = \
    ConfigScheme(True, True, False, train_filter=field_sample_filter_channel_filter,
                 val_filter=field_sample_filter_channel_filter,
                 test_filter=field_sample_filter_channel_filter,
                 lgbm_params=new_lgbm_params
                 )

ffm_data_config_80 = ConfigScheme(False, False, True,shuffle_sample_filter_1_to_2,
                                  shuffle_sample_filter_1_to_2,None,  discretization=50,
                               gen_ffm_test_data=True)

train_config_81 = ConfigScheme(False, True, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=True, add_hist_statis_fts=True,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               lgbm_params=new_lgbm_params
                               )
train_config_81 = ConfigScheme(False, True, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=True, add_hist_statis_fts=True,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               lgbm_params=new_lgbm_params
                               )


train_config_85 = ConfigScheme(True, True, False,
                               shuffle_sample_filter_1_to_2,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=True, add_hist_statis_fts=True,
                               lgbm_params=new_lgbm_params
                               )




train_config_86 = ConfigScheme(False, False, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               train_wordbatch=True
                               )


train_config_88 = ConfigScheme(False, False, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               train_wordbatch=True,
                               discretization=50
                               )

train_config_88_2 = ConfigScheme(False, False, False,
                               None,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_wordbatch=True,
                               predict_wordbatch = True,
                               log_discretization=False,
                               discretization=50
                               )

train_config_89_4 = ConfigScheme(False, False, False,
                               None,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_wordbatch=True,
                               predict_wordbatch = True,
                               log_discretization=True,
                               use_interactive_features=True
                               )
train_config_89_5 = ConfigScheme(False, False, False,
                               None,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_wordbatch=False,
                               predict_wordbatch = True,
                               log_discretization=True,
                               use_interactive_features=True,
                               train_wordbatch_streaming=True,
                               train_start_time=None
                               )
train_config_88_3 = ConfigScheme(False, False, False,
                               None,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_wordbatch=True,
                               predict_wordbatch = True,
                               log_discretization=True,
                               discretization=0
                               )
train_config_89 = ConfigScheme(False, False, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               train_wordbatch=True,
                               log_discretization=True
                               )

train_config_89_6 = ConfigScheme(False, False, False,
                               shuffle_sample_filter,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=False, add_hist_statis_fts=False,
                               train_start_time=val_time_range_start,
                               train_end_time=val_time_range_end,
                               val_start_time=train_time_range_start,
                               val_end_time=train_time_range_end,
                               train_wordbatch=True,
                               log_discretization=True,
                               use_interactive_features=True
                               )

train_config_87_3 = ConfigScheme(True, True, False,
                               None,
                               shuffle_sample_filter,
                               None,
                               seperate_hist_files=True, add_hist_statis_fts=True,
                               lgbm_params=new_lgbm_params
                               )


train_config_94_2 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_2,
                                 shuffle_sample_filter_1_to_2,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )


train_config_94_1 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )
train_config_94_3 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )
train_config_94_8 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_2,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )

train_config_97 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_2_11,
                                 new_train= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=False,
                                 grid_search_ft_gen=True
                                 )


train_config_98 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms'
                                  )




train_config_94_14 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_96_8_10,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )

train_config_94_15 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_2_11,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )


train_config_106_3 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )
train_config_106_5 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )



train_config_94_8 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_2,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )

train_config_94_13 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_2_11,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 new_predict=True
                                 )
train_config_97 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_2_11,
                                 new_train= False,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='grid_search_ft_gen'
                                 )

train_config_94_14 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_96_8_10,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )

train_config_98 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms'
                                  )
train_config_96 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='lgbm_params_search'
                                  )

train_config_99 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_20,
                                 shuffle_sample_filter_1_to_20,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms'
                                  )
train_config_99_4 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter_1_to_20,
                                 shuffle_sample_filter_1_to_20,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='grid_search_ft_coms_plus_lgbm_searcher'
                                  )
train_config_100 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_97478
                                  )
train_config_100_3 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_search_99_1
                                  )


train_config_102 = ConfigScheme(False, False, False,
                                 shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=lgbm_params_from_search_101,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )


train_config_103 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_train= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict'
                                 )

train_config_104 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=public_train_from,
                                 train_to=public_train_to,
                                 val_from=public_val_from,
                                 val_to=public_val_to,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )
train_config_105 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )

train_config_106 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately'
                                  )
train_config_106_3 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )
train_config_106_5 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )

train_config_106_6 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )

train_config_106_7 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=ft_coms_from_public
                                  )

train_config_108 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )

train_config_108_2 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )


train_config_108_3 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_2,
                                  shuffle_sample_filter_1_to_2,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=public_train_from,
                                 train_to=public_train_to,
                                 val_from=public_val_from,
                                 val_to=public_val_to,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )


train_config_108_4 = ConfigScheme(False, False, False,
                                  shuffle_sample_filter_1_to_2,
                                  shuffle_sample_filter_1_to_2,
                                 None,
                                 lgbm_params=public_kernel_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict_gen_fts_seperately',
                                 new_predict=True,
                                 add_features_list=ft_coms_from_public
                                  )

train_config_106_8 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict'
                                  )




train_config_103_5 = ConfigScheme(False, False, False,
                                 None,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='train_and_predict'
                                 )
train_config_103_6 = ConfigScheme(False, False, False,
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
                                 add_features_list=ft_coms_from_public
                                 )
train_config_103_7 = ConfigScheme(False, False, False,
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
                                 add_features_list=ft_coms_from_public_astype
                                 )
train_config_103_8 = ConfigScheme(False, False, False,
                                 None,
                                 None,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 new_predict= True,
                                 train_from=public_train_from,
                                 train_to=public_train_to,
                                 val_from=public_val_from,
                                 val_to=public_val_to,
                                 run_theme='train_and_predict',
                                 add_features_list=ft_coms_from_public_astype
                                 )



train_config_106_10 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=add_features_list_origin_astype
                                   )

train_config_106_11 = ConfigScheme(False, False, False,
                                shuffle_sample_filter,
                                 shuffle_sample_filter,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_8_4am,
                                 train_to=id_8_3pm,
                                 val_from=id_9_4am,
                                 val_to=id_9_3pm,
                                 run_theme='train_and_predict',
                                 add_features_list=ft_coms_from_public_astype
                                  )

train_config_115 = ConfigScheme(False, False, False,
                                shuffle_sample_filter_1_to_3,
                                shuffle_sample_filter_1_to_3,
                                 None,
                                 lgbm_params=new_lgbm_params,
                                 train_from=id_9_4am,
                                 train_to=id_9_3pm,
                                 val_from=id_8_4am,
                                 val_to=id_8_3pm,
                                 run_theme='lgbm_params_search',
                                 add_features_list=add_features_list_origin_no_channel_next_click
                                  )




search_features_list = [

    # ====================
    # my best features
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'channel', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'os', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['channel', 'hour', 'ip'], 'op': 'var'},
    {'group': ['app', 'os', 'channel', 'ip'], 'op': 'skew'},
    {'group': ['app', 'channel', 'ip'], 'op': 'mean'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

best_single_group_in_search = [
    {'group': ['app', 'device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip'], 'op': 'mean'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

ft_coms_97478=[
    {'group': ['app', 'device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip'], 'op': 'var'},
    {'group': ['device', 'os', 'ip'], 'op': 'mean'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os','hour','ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_from_public=[
    # importance 27 .. 4:
    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'cumcount'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique'},


    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},

    {'group': ['ip', 'os'], 'op': 'cumcount'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique'},

    # count:
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},

    # var:
    {'group': ['ip','day','channel','hour'], 'op': 'var'},
    {'group': ['ip','app', 'os', 'hour'], 'op': 'var'},
    {'group': ['ip','app', 'channel', 'day'], 'op': 'var'},

    # mean:
    {'group': ['ip','app', 'channel','hour'], 'op': 'mean'},

    #{'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]

ft_coms_from_public_astype=[
    # importance 27 .. 4:
    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'cumcount'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique'},


    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},

    {'group': ['ip', 'os'], 'op': 'cumcount'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique'},

    # count:
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count', 'astype':'uint16'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count', 'astype':'uint16'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count', 'astype':'uint16'},

    # var:
    {'group': ['ip','day','channel','hour'], 'op': 'var'},
    {'group': ['ip','app', 'os', 'hour'], 'op': 'var'},
    {'group': ['ip','app', 'channel', 'day'], 'op': 'var'},

    # mean:
    {'group': ['ip','app', 'channel','hour'], 'op': 'mean'},

    #{'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_from_public_astype_all_set_type=[
    # importance 27 .. 4:
    {'group': ['ip', 'channel'], 'op': 'nunique', 'astype':'int64'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'cumcount', 'astype':'int64'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique', 'astype':'int64'},


    {'group': ['ip', 'app'], 'op': 'nunique', 'astype':'int64'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique', 'astype':'int64'},
    {'group': ['ip', 'device'], 'op': 'nunique', 'astype':'int64'},
    {'group': ['app', 'channel'], 'op': 'nunique', 'astype':'int64'},

    {'group': ['ip', 'os'], 'op': 'cumcount', 'astype':'int64'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique', 'astype':'int64'},

    # count:
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count', 'astype':'uint16'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count', 'astype':'uint16'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count', 'astype':'uint16'},

    # var:
    {'group': ['ip','day','channel','hour'], 'op': 'var', 'astype':'float64'},
    {'group': ['ip','app', 'os', 'hour'], 'op': 'var', 'astype':'float64'},
    {'group': ['ip','app', 'channel', 'day'], 'op': 'var', 'astype':'float64'},

    # mean:
    {'group': ['ip','app', 'channel','hour'], 'op': 'mean', 'astype':'float64'},

    #{'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick', 'astype':'int64'}
]
ft_coms_search_99_1=[
    # importance 27 .. 4:
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    # importance 3:
    {'group': ['app', 'channel', 'hour'], 'op': 'cumcount'},
    {'group': ['app', 'hour'], 'op': 'mean'},
    {'group': ['device', 'os', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_search_99_2_2=[
    # importance 4:
    {'group': ['device', 'channel'], 'op': 'cumcount'},
    {'group': ['device', 'os', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'os', 'channel', 'ip'], 'op': 'count'},
    {'group': ['app', 'ip', 'is_attributed'], 'op': 'count'},
    # importance 27 .. 5:
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    # importance 3:

    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
ft_coms_search_99_2=[
    # importance 27 .. 5:
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'os', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'os', 'ip', 'is_attributed'], 'op': 'count'},
    {'group': ['device', 'hour', 'ip', 'is_attributed'], 'op': 'count'},
    # importance 3:

    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
]
add_features_list_origin_astype = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count', 'astyep':'uint16'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
    ]



new_new_lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.2,
    # 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 255,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': 4,
    'verbose': 0,
    'scale_pos_weight': 200.0
}

lgbm_params_from_search_0_35 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 0.5758903957135874, 'learning_rate': 1.1760632807147045, 'max_depth': 9,
 'min_child_samples': 200, 'min_child_weight': 6, 'min_split_gain': 1.0, 'n_estimators': 86, 'num_leaves': 31,
 'reg_alpha': 8.954987962970492, 'reg_lambda': 1000.0, 'scale_pos_weight': 6.1806180811037486, 'subsample': 1.0,
 'subsample_for_bin': 740701, 'subsample_freq': 0}


lgbm_params_from_search_2_11 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 0.01, 'learning_rate': 1.040731554567982, 'max_depth': 0, 'min_child_samples': 51,
    'min_child_weight': 7, 'min_split_gain': 1.0109536459124555, 'n_estimators': 300, 'num_leaves': 31,
    'reg_alpha': 1.2420399086947274, 'reg_lambda': 1000.0, 'scale_pos_weight': 4.652061504081354,
    'subsample': 1.0, 'subsample_for_bin': 571876, 'subsample_freq': 0}


lgbm_params_from_search_96_8_10 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 0.6021467085338628, 'learning_rate': 0.37861782981986614, 'max_depth': 5,
    'min_child_samples': 171, 'min_child_weight': 1, 'min_split_gain': 0.004337458691335552,
    'n_estimators': 249, 'num_leaves': 10, 'reg_alpha': 8.1447463220916e-05, 'reg_lambda': 64.24440555484793,
    'scale_pos_weight': 4.350080991126866, 'subsample': 0.5332391076871872, 'subsample_for_bin': 470584,
    'subsample_freq': 1
}

lgbm_params_from_search_0_81 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 0,
    'colsample_bytree': 1.0,
    'learning_rate': 1.3817527732999606,
    'max_depth': 7,
    'min_child_samples': 92,
    'min_child_weight': 10,
    'min_split_gain': 1.0096460744834064,
    'n_estimators': 61,
    'num_leaves': 26,
    'reg_alpha': 6.082283392201092,
    'reg_lambda': 1000.0,
    'scale_pos_weight': 1.9673649490776584,
    'subsample': 1.0,
    'subsample_for_bin': 800000,
    'subsample_freq': 1
}

lgbm_params_from_search_101 = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'verbose': 9,
    'min_child_weight': 3, 'subsample_for_bin': 542556, 'learning_rate': 0.30803266868575857,
    'subsample_freq': 0, 'max_depth': 8, 'subsample': 1.0, 'num_leaves': 7, 'reg_lambda': 28.66709162037324,
    'early_stopping_round': 371, 'min_split_gain': 0.0028642748250716932, 'reg_alpha': 7.351107378081451e-05,
    'scale_pos_weight': 29.983824928443436, 'colsample_bytree': 0.8639270134191158, 'min_child_samples': 200}


