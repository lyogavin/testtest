import pandas as pd

prd = pd.read_csv('new_test.sp.prd', header=0, usecols=['is_attributed'])

test = pd.read_csv('test_fe.csv.04-04-2018_19-20-51', header=0, usecols=['click_id','app'])

test['is_attributed'] = prd


del test['app']

test.to_csv('new_test.sp.prd.added', index=False)

print('added')