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
new_trans = csv_io.read_file(path_const.RE_NEW_TRANS1)
old_trans = csv_io.read_file(path_const.RE_OLD_TRANS1)
new_trans6 = csv_io.read_file(path_const.NEW_TRANS6)
old_trans6 = csv_io.read_file(path_const.OLD_TRANS6)
print(train.shape)
timer.time("load csv in ")

train = pd.merge(train, new_trans, on="card_id", how="left")
train = pd.merge(train, old_trans, on="card_id", how="left")
train = pd.merge(train, new_trans6, on="card_id", how="left")
train = pd.merge(train, old_trans6, on="card_id", how="left")
# print(train.shape)
# print(test.shape)
#
fer = jit_fe.JitFe()
train = fer.do_fe(train)

train_y = train["target"]
base_col = [
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
]
drop_col = [
    "card_id", "target",  # "feature_1", "feature_2", "feature_3",
    "old_weekend_mean", "new_weekend_mean", "new_authorized_flag_mean",
    "old_null_state", "new_null_state", "new_null_install", #"old_null_install",
    "old_cat3_pur_mean", "new_cat3_pur_mean", "old_cat2_pur_mean", "new_cat2_pur_mean",
    "new_category_4_mean",  # "new_merchant_group_id_nunique", "old_merchant_group_id_nunique"
    "new_mon_nunique_mean", "new_woy_nunique_mean",
    # "new_month_lag_ptp", "new_month_lag_min",
    "new_purchase_amount_skew",  # "new_purchase_amount_std",
    "old_purchase_amount_skew",  # "old_purchase_amount_std",
    # "new_category_2_nunique", "old_category_2_nunique",
]
train_x = drop_col_util.drop_col(train, drop_col)

try_col = [c for c in train.columns if c not in drop_col and c not in base_col]
print(len(try_col))
outliers = (train["target"] < -30).astype(int).values
split_num = 4
col_list = []
score_list = []
for i, c in enumerate(try_col):
    use_col = base_col + [c]
    _train_x = train_x[use_col]

    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=0)
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    for train_index, test_index in skf.split(train, outliers):
        X_train, X_test = _train_x.iloc[train_index], _train_x.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
    avg_score = str(total_score / split_num)
    col_list.append(c)
    score_list.append(avg_score)

ret_df = pd.DataFrame({
    "col": col_list,
    "score": score_list
})

ret_df = ret_df.sort_values(by="score")
ret_df.to_csv("../output/eval_features.csv")

# base: 3.6504563125282385

