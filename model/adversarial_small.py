import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.utils import drop_col_util
from elo.fe import jit_fe
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.TRAIN1)
test = csv_io.read_file(path_const.TEST1)
new_trans = csv_io.read_file(path_const.RE_NEW_TRANS1)
old_trans = csv_io.read_file(path_const.RE_OLD_TRANS1)
new_trans6 = csv_io.read_file(path_const.NEW_TRANS6)
old_trans6 = csv_io.read_file(path_const.OLD_TRANS6)
print(train.shape)
print(test.shape)
timer.time("load csv in ")

train = pd.merge(train, new_trans, on="card_id", how="left")
train = pd.merge(train, old_trans, on="card_id", how="left")
train = pd.merge(train, new_trans6, on="card_id", how="left")
train = pd.merge(train, old_trans6, on="card_id", how="left")
#
test = pd.merge(test, new_trans, on="card_id", how="left")
test = pd.merge(test, old_trans, on="card_id", how="left")
test = pd.merge(test, new_trans6, on="card_id", how="left")
test = pd.merge(test, old_trans6, on="card_id", how="left")
# print(train.shape)
# print(test.shape)
#
fer = jit_fe.JitFe()
train = fer.do_fe(train)
test = fer.do_fe(test)

# 3.660 - 3.658
use_col = [
    "new_trans_elapsed_days_max", "new_trans_elapsed_days_min", "new_trans_elapsed_days_mean",  # 0.001
    "old_trans_elapsed_days_max", "old_trans_elapsed_days_min", "old_trans_elapsed_days_mean",  # 0.025 mean001
    "new_last_day",  # 0.005
    "old_installments_sum", "old_installments_mean",  # 0.005
    "old_month_nunique", "old_woy_nunique",  # 0.010
    "old_merchant_id_nunique",  # 0.002
    "new_month_lag_mean", "old_month_lag_mean", "elapsed_days",  # 0.010
    "new_purchase_amount_max", "new_purchase_amount_count", "new_purchase_amount_mean",  # 0.020
    "old_purchase_amount_max", "old_purchase_amount_count", "old_purchase_amount_mean",  # 0.002
    "old_category_1_mean", "new_category_1_mean",  # 0.006
    "old_authorized_flag_sum",  # "old_authorized_flag_mean", bad?
    "old_no_city_purchase_amount_min",  # 0.003
    "old_no_city_purchase_amount_max", "old_no_city_purchase_amount_mean",  # 0.002
    "rec1_purchase_amount_count",  # 0.005
    "old_month_lag_max",  # 0.002
    "new_time_diff_mean", "new_trans_elapsed_days_std",  # 0.002
    #"old_month_diff_mean", "old_pa2_month_diff_min",  # 0.004
]
use_col += [
    # "old_first_buy", "old_last_buy" worse
    # "proper_new_purchase_amount_sum" worse
    # "new_trans_elapsed_days_std", worse
    # "new_purchase_amount_sum", "old_purchase_amount_sum",  # 0.0005?
    # "old_time_diff_std" worse
]
train_x = train.drop(columns=["card_id", "target"])
test_x = test.drop(columns=["card_id"])
train["tt"] = 0
test["tt"] = 1
train_y = train["tt"]
test_y = test["tt"]
all_x = pd.concat([train_x, test_x], axis=0)
all_y = pd.concat([train_y, test_y], axis=0)

timer.time("prepare train in ")
print(train_x.shape)
print(test_x.shape)
print(all_x.shape)
print(all_y.shape)

bagging_num = 1
split_num = 4
for bagging_index in range(bagging_num):
    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=99 * bagging_index)
    logger.print("random_state=" + str(99*bagging_index))
    lgb = pocket_lgb.AdversarialLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(all_x, all_y):
        X_train, X_test = all_x.iloc[train_index], all_x.iloc[test_index]
        y_train, y_test = all_y.iloc[train_index], all_y.iloc[test_index]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        score = model.best_score["valid_0"]["auc"]
        total_score += score
        y_pred = model.predict(test_x)
        valid_set_pred = model.predict(X_test)
        models.append(model)
        timer.time("done one set in")

    lgb.show_feature_importance(models[0], path_const.FEATURE_GAIN)
    avg_score = str(total_score / split_num)
    logger.print("average score= " + avg_score)
    timer.time("end train in ")




