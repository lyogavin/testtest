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