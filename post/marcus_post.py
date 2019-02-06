import os

path = "D://Gdrive//Marcus//competition//Elo//"
os.chdir(path)
import numpy as np
import pandas as pd

train = pd.read_pickle(path + 'raw_pickle//' + 'train')
len(train[train['target'] < -20])  # 2207
len(train[train['target'] < -20]) / len(train)  # 0.0108

sub_with_outlier = pd.read_csv(path + 'sub//' + 'ridge_v1.csv')  # ridge stack of 20 different seeds lgbs
sub_without_outlier = pd.read_csv(path + 'sub//' + 'v10_re4.csv')  # ridge stack of 20 different seeds non-outlier lgbs

binary_oof_test = pd.read_pickle(
    path + 'oof//outlier_lgb_v3_kh_time_feature2_oof_test')  # binary model of whether outlier or not
logistic_test = pd.read_csv(path + 'sub//outlier_lgb_logistic.csv')  # logistic stack of 20 different binary models

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

sub_with_outlier[['card_id', 'target']].to_csv(path + 'sub//' + 'test9.csv', index=False)