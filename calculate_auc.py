import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse,sys
from sklearn import metrics


if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('label_path')
parser.add_argument('predict_path')
args = vars(parser.parse_args())

def calculate_auc(label, predict):
    #if getattr(val['predict'], "values", None) is None:
    #    return 0
    fpr, tpr, thresholds = metrics.roc_curve(label,
                                             predict)
    auc = metrics.auc(fpr, tpr)
    return auc

print('reading label data:')


label = pd.read_csv(args['label_path'], dtype='int', usecols=['is_attributed'])
print('label len:', len(label))

predict = pd.read_csv(args['predict_path'], dtype='int', usecols=['is_attributed'])
print('label len:', len(label))


print('auc:', calculate_auc(label.values, predict.values))