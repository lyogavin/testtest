
# path = '../input/'
path = '../input/talkingdata-adtracking-fraud-detection/'
path_train_hist = '../input/data_with_hist/'
path_test_hist = '../input/data_with_hist/'
path_train = path + 'train.csv'
path_train_sample = path + 'train_sample.csv'
path_test = path + 'test.csv'
path_test_sample = path + 'test_sample.csv'
path_test_supplement = path + 'test_supplement.csv'
path_test_supplement_sample = path + 'test_supplement_sample.csv'


ft_cache_path = '../input/talkingdata-adtracking-fraud-detection/ft_cache/'


train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']

#categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
categorical = ['app', 'device', 'os', 'channel', 'hour']
# with ip:
# categorical = ['app', 'device', 'os', 'channel', 'hour', 'ip']

pick_hours={4, 5, 10, 13, 14}
cvr_columns_lists = [
    ['ip', 'app', 'device', 'os', 'channel'],
    # ['app', 'os'],
    # ['app','channel'],
    # ['app', 'device'],
    ['ip', 'device'],
    ['ip']
    # , ['os'], ['channel']
]
agg_types = ['non_attr_count', 'cvr']

acro_names = {
    'ip': 'I',
    'app': 'A',
    'device': 'D',
    'os': 'O',
    'channel': 'C',
    'hour': 'H',
    'ip_day_hourcount': 'IDH-',
    'ip_day_hour_oscount': 'IDHO-',
    'ip_day_hour_appcount': 'IDHA-',
    'ip_day_hour_app_oscount': 'IDHAO-',
    'ip_app_oscount': "IAO-",
    'ip_appcount': "IA-",
    'ip_devicecount': "ID-",
    'app_channelcount': "AC-",
    'app_day_hourcount': 'ADH-',
    'ip_in_test_hhcount': "IITH-",
    'next_click': 'NC',
    'app_channel': 'AC',
    'os_channel': 'OC',
    'app_device': 'AD',
    'app_os_channel': 'AOC',
    'ip_app': 'IA',
    'app_os': 'AO'
}

most_freq_values_in_test_data  = {
    'device': [],
    'app':[],
    'os':[],
    'channel':[
        107,265,232,477,178,153,134,259,128,442,127,379,205,466,121,137,439,489,145,480,135,469,219,122,215,
        435,244,237,347,409,140,334,452,328,377,211,236,424,173,130,459,116,19,115,315,401,125,212,340,
        349,258,266,234,213,105,445,376,386,481,319,3,343,463,478,412,278,430,101,111,400,364,497,124,
        402,487,325,243,417,326,18,113,448,242,150,17,317,488,21
    ], # 88, 99.1%
    'ip':[]
}
least_freq_values_in_test_data  = {
    'device': [],
    'app':[],
    'os':[],
    'channel':[
        172, 149, 138, 322, 490, 169, 251, 458, 483, 281, 222, 181, 223, 256, 352, 407, 233, 216, 123, 4, 420, 114, 272,
        408, 410, 341, 455, 0, 261, 356, 262, 126, 457, 174, 208, 311, 416, 498, 414, 277, 353, 451, 332, 391, 479, 15,
        110, 22, 465, 253, 456, 460, 108, 203, 160, 320, 484, 450, 274, 5, 24, 419, 120, 268, 446, 330, 360, 361, 421,
        224, 30, 453, 118, 13, 129, 182, 171, 245, 406, 282, 411, 449, 225, 333, 210, 371, 404, 280, 373, 467
    ], # 90, < 1%
    'ip':[]
}
hist_st = []
iii = 0
for type in agg_types:
    for cvr_columns in cvr_columns_lists:
        new_col_name = '_'.join(cvr_columns) + '_' + type
        hist_st.append(new_col_name)

field_sample_filter_channel_filter = {'filter_type': 'filter_field',
                                      'filter_field': 'channel',
                                      'filter_field_values': [107, 477, 265]}
field_sample_filter_app_filter1 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [12, 18, 14]}
field_sample_filter_app_filter2 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [12]}
field_sample_filter_app_filter3 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [18, 14]}
field_sample_filter_app_filter4 = {'filter_type': 'filter_field',
                                   'filter_field': 'app',
                                   'filter_field_values': [8, 11]}

train_time_range_start = '2017-11-09 04:00:00'
train_time_range_end = '2017-11-09 15:00:00'

val_time_range_start = '2017-11-08 04:00:00'
val_time_range_end = '2017-11-08 15:00:00'

test_time_range_start = '2017-11-10 04:00:00'
test_time_range_end = '2017-11-10 15:00:00'
9308569
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
id_9_4pm_reserve_last_250w = id_9_4pm - 250*10000



sample_from_list = [0, 50000]
sample_to_list = [49998, 90000]

public_train_from = 109903890
public_train_to =  147403890
public_val_from = 147403890
public_val_to = 149903890

debug_train_from = 0
debug_train_to = 90000
debug_val_from=90000
debug_val_to=100000


shuffle_sample_filter = {'filter_type': 'sample', 'sample_count': 6}
shuffle_sample_filter_1_to_6 = {'filter_type': 'sample', 'sample_count': 6}

shuffle_sample_filter_1_to_2 = {'filter_type': 'sample', 'sample_count': 2}
shuffle_sample_filter_1_to_3 = {'filter_type': 'sample', 'sample_count': 3}

shuffle_sample_filter_1_to_10 = {'filter_type': 'sample', 'sample_count': 10}
shuffle_sample_filter_1_to_20 = {'filter_type': 'sample', 'sample_count': 20}
shuffle_sample_filter_1_to_10k = {'filter_type': 'sample', 'sample_count': 1}

hist_ft_sample_filter = {'filter_type': 'hist_ft'}

random_sample_filter_0_5 = {'filter_type': 'random_sample', 'frac': 0.5}

random_sample_filter_0_2 = {'filter_type': 'random_sample', 'frac': 0.2}


dtypes = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32'
}


process_poll_size = 10

neg_sample_rate = 200