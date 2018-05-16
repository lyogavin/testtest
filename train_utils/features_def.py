
add_features_list_origin_no_channel_next_click = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]

add_features_list_origin_no_channel_next_click_no_day = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
]


add_features_list_origin_no_channel_next_click_no_day_scvr = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'smoothcvr'},
]

add_features_list_fts_search_reduced_split_scvr = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},

    {'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'smoothcvr'},

]

add_features_list_fts_search_reduced_split_scvr_only_1 = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},

    {'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},

]

add_features_list_fts_search_reduced_split_add_counting_1 = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},

    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},

]

add_features_list_33 = \
    add_features_list_fts_search_reduced_split_add_counting_1 + [
    {'group': ['app', 'hour','ip','is_attributed'], 'op': 'cumcount'}, #app_hour_ip_is_attributedcumcount
    ]

add_features_list_34 = \
    add_features_list_fts_search_reduced_split_add_counting_1 + [
    {'group': ['app','ip','is_attributed'], 'op': 'smoothcvr'}, #app_ip_is_attributedsmoothcvr
    ]
add_features_list_search_28 = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'is_attributed'], 'op': 'count'},

    #{'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    #{'group': ['ip', 'app'], 'op': 'nunique'},

    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},

]
add_features_list_search_29 = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    #{'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},

    #{'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},

    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'device','channel', 'is_attributed'], 'op': 'cumcount'},
]
add_features_list_search_31 = [
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    #{'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},

    #{'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},

    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'hour'], 'op': 'mean'},
]

add_features_list_fts_search_reduced_split_add_counting_1_var_best = \
    add_features_list_fts_search_reduced_split_add_counting_1 + [
    {'group': ['device', 'os','channel','hour', 'ip'], 'op': 'var'},
    ]

add_features_list_fts_search_reduced_split_add_counting_1_scvr_best = \
    add_features_list_fts_search_reduced_split_add_counting_1 + [
        {'group': ['os', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    ]

add_features_list_fts_search_reduced_split_scvr_add_var = add_features_list_fts_search_reduced_split_scvr + [
    {'group': ['ip', 'day', 'channel','hour'], 'op': 'var'},
    {'group': ['ip', 'app', 'os','hour'], 'op': 'var'},
    {'group': ['ip', 'app', 'channel','day'], 'op': 'var'},
]
add_features_list_fts_search_reduced_split_scvr_add_var_only_1 = add_features_list_fts_search_reduced_split_scvr + [
    {'group': ['ip', 'day', 'channel','hour'], 'op': 'var'},
]

add_features_list_smooth_cvr_from_search_121_13 = [

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'channel', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'hour', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['hour', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'channel', 'is_attributed'], 'op': 'smoothcvr'}
]

add_features_list_smooth_cvr_from_search_121_13_reduced = [

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},

    #{'group': ['ip', 'channel', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'hour', 'is_attributed'], 'op': 'smoothcvr'},
    #{'group': ['hour', 'ip', 'is_attributed'], 'op': 'smoothcvr'}
    #{'group': ['os', 'channel', 'is_attributed'], 'op': 'smoothcvr'}
]

add_features_list_pub_entire_set = [

    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'day','hour'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app','os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device','os'], 'op': 'nunique'},

    {'group': ['ip','os'], 'op': 'cumcount'},
    {'group': ['ip','device','os','app'], 'op': 'cumcount'},
    {'group': ['ip','device','os','channel'], 'op': 'cumcount'},

    {'group': ['ip', 'app','channel','is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'os','app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'day', 'channel','hour'], 'op': 'var'},
    {'group': ['ip', 'app', 'os','hour'], 'op': 'var'},
    {'group': ['ip', 'app', 'channel','day'], 'op': 'var'},
    {'group': ['ip', 'app', 'channel','hour'], 'op': 'mean'},

]

add_features_list_fts_search = [

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    #{'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},

    # no smooth cvr first
    #{'group': ['app', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    #{'group': ['os', 'ip', 'is_attributed'], 'op': 'smoothcvr'},
    #{'group': ['app', 'hour', 'is_attributed'], 'op': 'smoothcvr'},

    #{'group': ['ip', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    #{'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'},

    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'day','hour'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app','os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device','os','app'], 'op': 'nunique'},
    {'group': ['ip', 'device','os','app'], 'op': 'cumcount'},

    {'group': ['ip','os'], 'op': 'cumcount'},
    {'group': ['ip','device','os','channel'], 'op': 'cumcount'},

    {'group': ['ip', 'app','channel','is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'os','app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'day', 'channel','hour'], 'op': 'var'},
    {'group': ['ip', 'app', 'os','hour'], 'op': 'var'},
    {'group': ['ip', 'app', 'channel','day'], 'op': 'var'},
    {'group': ['ip', 'app', 'channel','hour'], 'op': 'mean'},

]


add_features_list_fts_search_reduced_gain = [

    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
]


add_features_list_fts_search_reduced_split = [

    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app', 'is_attributed'], 'op': 'count'},

]
add_features_list_smooth_cvr = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'os', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'device', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['hour', 'is_attributed'], 'op': 'smoothcvr'},
    #{'group': ['ip', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['app', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['device', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['os', 'is_attributed'], 'op': 'smoothcvr'},
    {'group': ['channel', 'is_attributed'], 'op': 'smoothcvr'},

    # for debuging the smooth cvrs:
    #{'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'smoothcvr'}, #cheating ft, low val auc, avoid it

    #{'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'mean'},
    #{'group': ['hour', 'is_attributed'], 'op': 'mean'},
    #{'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'count'},
    #{'group': ['hour', 'is_attributed'], 'op': 'count'}

    ]


add_features_list_origin_no_channel_next_click_best_ct_nu_from_search = [
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['os', 'hour', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'device', 'os', 'hour', 'ip'], 'op': 'nunique'},

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]
add_features_list_origin_no_channel_next_click_best_ct_nu_from_search_28 = [
    {'group': ['ip', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'},

    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]

add_features_list_origin_no_channel_next_click_ip_freq_ch = \
    add_features_list_origin_no_channel_next_click + [
        {'group': ['ip', 'in_test_frequent_channel', 'is_attributed'], 'op': 'count'}
    ]

add_features_list_pub_asraful_kernel  = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},

    {'group': ['ip', 'channel'], 'op': 'nunique'},
    {'group': ['ip', 'device', 'os', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'day', 'hour'], 'op': 'nunique'},
    {'group': ['ip', 'app'], 'op': 'nunique'},
    {'group': ['ip', 'app', 'os'], 'op': 'nunique'},
    {'group': ['ip', 'device'], 'op': 'nunique'},
    {'group': ['app', 'channel'], 'op': 'nunique'},

    {'group': ['ip', 'os'], 'op': 'cumcount'},
    {'group': ['ip','device','os', 'app'], 'op': 'cumcount'},

    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'app', 'os', 'hour'], 'op': 'var'}
    ]

add_features_add_best_nunique = add_features_list_origin_no_channel_next_click + [
    {'group': ['app', 'channel', 'ip'], 'op': 'nunique'},
    {'group': ['app', 'channel', 'hour', 'ip'], 'op': 'nunique'}
]


add_features_from_pub_ftrl = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},

    {'group': ['ip', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'device', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'channel', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'channel'], 'op': 'nunique'}
    ]


add_features_list_origin_no_channel_next_click_10mincvr = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['id_10min', 'is_attributed'], 'op': 'mean'}
]



add_features_list_origin_no_channel_next_click_days = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'day', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day','in_test_hh', 'is_attributed'], 'op': 'count'}
    ]


add_features_list_origin_no_channel_next_click_no_app = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    ]


add_features_list_origin_no_channel_next_click_stnc = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    # st nc:
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.98'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.02'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'min'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'var'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'mean'}
    #,{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'skew'}
    ]

add_features_list_origin_no_channel_next_click_varnc = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    # st nc:
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.98'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.02'},
    {'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'var'},
    #,{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'skew'}
    ]

add_features_list_origin_no_channel_next_click_next_n_click = [

    # ====================
    # my best features
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextclick'},
    {'group': ['ip', 'app', 'device', 'os', 'is_attributed'], 'op': 'nextnclick'},
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'}
    # st nc:
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.98'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'qt0.02'},
    #{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'var'},
    #,{'group': ['ip', 'app', 'device', 'os', 'ip_app_device_os_is_attributednextclick'], 'op': 'skew'}
    ]

add_features_list_origin = [

    # ====================
    # my best features
    {'group': ['ip', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'day', 'hour', 'app', 'os', 'is_attributed'], 'op': 'count'},
    {'group': ['app', 'day', 'hour', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'in_test_hh', 'is_attributed'], 'op': 'count'},
    {'group': ['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], 'op': 'nextclick'}
    # =====================
    # try word batch featuers:
    # =====================
    # {'group': ['ip', 'day', 'hour'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['ip', 'app'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['ip', 'app', 'os'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['ip', 'device'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['app', 'channel'], 'with_hist': False, 'counting_col': 'os'},
    # ======================

    # {'group':['app'], 'with_hist': False, 'counting_col':'channel'},
    # {'group': ['os'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['device'], 'with_hist': False, 'counting_col': 'channel'},
    # {'group': ['channel'], 'with_hist': False, 'counting_col': 'os'},
    # {'group': ['hour'], 'with_hist': False, 'counting_col': 'os'},

    # {'group':['ip','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip','os', 'app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip','hour','channel'], 'with_hist': with_hist_profile, 'counting_col':'os'},
    # {'group':['ip','hour','os'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['ip','hour','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
    # {'group':['channel','app'], 'with_hist': with_hist_profile, 'counting_col':'os'},
    # {'group':['channel','os'], 'with_hist': with_hist_profile, 'counting_col':'app'},
    # {'group':['channel','app','os'], 'with_hist': with_hist_profile, 'counting_col':'device'},
    # {'group':['os','app'], 'with_hist': with_hist_profile, 'counting_col':'channel'},
]
