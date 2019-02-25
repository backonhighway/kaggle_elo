import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.loader import input_loader
from sklearn import model_selection
from elo.utils import random_col_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

loader = input_loader.GoldenLoader()
train, test = loader.load_team_input_v63()
timer.time("load csv in")

base_col = loader.small_col + loader.team_small_col
drop_col = [
    "card_id", "target",  # "feature_1", "feature_2", "feature_3",
    "old_weekend_mean", "new_weekend_mean", "new_authorized_flag_mean",
    "old_null_state", "new_null_state", "new_null_install",  # "old_null_install",
    "old_cat3_pur_mean", "new_cat3_pur_mean", "old_cat2_pur_mean", "new_cat2_pur_mean",
    "new_category_4_mean",  # "new_merchant_group_id_nunique", "old_merchant_group_id_nunique"
    "new_mon_nunique_mean", "new_woy_nunique_mean",
    # "new_month_lag_ptp", "new_month_lag_min",
    "new_purchase_amount_skew",  # "new_purchase_amount_std",
    "old_purchase_amount_skew",  # "old_purchase_amount_std",
    "pred_new", "old_same_buy_count", "old_purchase_amount_nunique", "new_purchase_amount_nunique",
    "old_installments_nunique", "new_installments_nunique",  # "pred_new_pur_max",
]
try_col = [c for c in train.columns if c not in drop_col and c not in base_col]
base_col_prob = [1.0, 0.9]
try_col_prob = [0.1, 0.3, 0.5, 0.7]

train_y = train["target"]
outliers = (train["target"] < -30).astype(int).values
split_num = 5
skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=4590)
lgb = pocket_lgb.GoldenLgb()
col_selector = random_col_selector.RandomColumnSelector(base_col, try_col, base_col_prob, try_col_prob)
exp_log_list = list()
for i in range(10):
    exp_log = dict()
    exp_log["exp_idx"] = i

    use_col, base_col_p, try_col_p = col_selector.select_col(i)
    exp_log["used_col"] = str(use_col)
    exp_log["base_col_p"] = base_col_p
    exp_log["try_col_p"] = try_col_p

    train_x = train[use_col]
    test_x = test[use_col]

    submission = pd.DataFrame()
    submission["card_id"] = test["card_id"]
    submission["target"] = 0
    train_cv = pd.DataFrame()
    train_cv["card_id"] = train["card_id"]
    train_cv["cv_pred"] = 0

    no_out_idx = train[train["target"] > -33].index
    train_preds = []
    for train_index, test_index in skf.split(train, outliers):
        _train_idx = [i for i in train_index if i in no_out_idx]
        _test_idx = [i for i in test_index if i in no_out_idx]
        X_train, X_test = train_x.iloc[_train_idx], train_x.iloc[_test_idx]
        y_train, y_test = train_y.iloc[_train_idx], train_y.iloc[_test_idx]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)

        y_pred = model.predict(test_x)
        valid_set_pred = model.predict(train_x.iloc[test_index])

        submission["target"] = submission["target"] + y_pred
        train_id = train.iloc[test_index]
        train_cv_prediction = pd.DataFrame()
        train_cv_prediction["card_id"] = train_id["card_id"]
        train_cv_prediction["cv_pred"] = valid_set_pred
        train_preds.append(train_cv_prediction)
        timer.time("done one set in")

    train_output = pd.concat(train_preds, axis=0)
    train_cv["cv_pred"] += train_output["cv_pred"]

    y_true = train_y
    y_pred = train_cv["cv_pred"]
    rmse_score = evaluator.rmse(y_true, y_pred)
    exp_log["cv_score"] = rmse_score

    submission["target"] = submission["target"] / split_num
    sub_file = path_const.get_subset_exp_sub(i)
    submission.to_csv(sub_file, index=False)

    train_cv["cv_pred"] = train_cv["cv_pred"]
    oof_file = path_const.get_subset_exp_oof(i)
    train_cv.to_csv(oof_file, index=False)

    exp_log_list.append(exp_log)

exp_df = pd.DataFrame(exp_log_list)
exp_df.to_csv(path_const.EXP_LOG, index=False)


