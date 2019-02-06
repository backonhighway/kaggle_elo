import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import numpy as np
import pandas as pd
from elo.common import evaluator

train = pd.read_csv("../input/train.csv")
len(train[train['target'] < -20])  # 2207
len(train[train['target'] < -20]) / len(train)  # 0.0108

sub_with_outlier = pd.read_csv("../sub/org_param_oof.csv")
sub_without_outlier = pd.read_csv("../sub/no_out2_oof.csv")
binary_oof_test = pd.read_csv("../sub/bin_oof.csv")
logistic_test = pd.read_csv("../sub/bin_oof.csv")
sub_with_outlier.columns = ["card_id", "target"]
sub_without_outlier.columns = ["card_id", "target"]
binary_oof_test.columns = ["card_id", "target"]
logistic_test.columns = ["card_id", "target"]
binary_oof_test = np.array(binary_oof_test["target"])

# imputing non-outlier lgbs
# ================================================================================#
threshold = 0.013
sub_with_outlier['outlier'] = binary_oof_test
index = (sub_with_outlier['outlier'] < threshold) & (logistic_test['target'] < 0.2)
sub_with_outlier.loc[index, 'target'] = sub_without_outlier.loc[index, 'target']

index = (sub_with_outlier['outlier'] > threshold) & (sub_with_outlier['outlier'] < 0.02)  # 0.02
sub_with_outlier.loc[index, 'target'] = (sub_with_outlier.loc[index, 'target'] + sub_without_outlier.loc[
    index, 'target']) / 2
# ================================================================================#


sub_with_outlier['target2'] = sub_with_outlier['target']

# assign outlier
# ================================================================================#
as_outlier_th = 0.51
outlier_value = -33.2193
bin_ = 0.01
th_list = list(np.arange(0.3, as_outlier_th, bin_))

th = th_list[0]
for th in th_list:
    index = sub_with_outlier['outlier'] > th
    sub_with_outlier.loc[index, 'target'] = outlier_value * (th + (bin_ / 2)) - 1

index = sub_with_outlier['outlier'] > as_outlier_th
sub_with_outlier.loc[index, 'target'] = outlier_value
# ================================================================================#


index = sub_with_outlier['target2'] < -19
sub_with_outlier.loc[index, 'target'] = outlier_value

index = (sub_with_outlier['target2'] < -15) & (sub_with_outlier['target'] != outlier_value)
sub_with_outlier.loc[index, 'target'] = sub_with_outlier.loc[index, 'target'] - 5

index = (sub_with_outlier['target2'] < -10) & (sub_with_outlier['target2'] > -15) & (
            sub_with_outlier['target'] != outlier_value)
sub_with_outlier.loc[index, 'target'] = sub_with_outlier.loc[index, 'target'] - 1

index = (logistic_test['target'] > 0.94) & (sub_with_outlier['target'] != outlier_value)
sub_with_outlier.loc[index, 'target'] = outlier_value

index = sub_with_outlier['target'] <= outlier_value
sub_with_outlier.loc[index, 'target'] = outlier_value

sub_with_outlier[['card_id', 'target']].to_csv("../post/first_pp.csv", index=False)
print(sub_with_outlier.shape)
print(sub_with_outlier.describe())

_train = train[["card_id", "target"]].copy()
_train.columns = ["card_id", "y_true"]
sub_with_outlier = pd.merge(sub_with_outlier, _train, on="card_id", how="inner")
print(sub_with_outlier.shape)
score = evaluator.rmse(sub_with_outlier["target"], sub_with_outlier["y_true"])
print(score)
