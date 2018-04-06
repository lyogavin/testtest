import pandas as pd
import numpy as py

path = '../input/' 

path_test = 'test_fe.csv.05-04-2018_12-36-48'

submit = pd.read_csv(path_test, usecols=['click_id'])
ffm_output = pd.read_csv('new_new_test.sp.prd', usecols=['click'])

submit['is_attributed'] = ffm_output['click']
print(submit)

#submit = submit.merge(ffm_output, how='left')
#submit['is_attributed

#submit = submit.rename(columns={'click':'is_attributed'})

submit.to_csv('ffm_submit.csv', index=False)








