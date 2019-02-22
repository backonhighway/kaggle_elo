import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.loader import input_loader
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

loader = input_loader.GoldenLoader()
train, test = loader.load_team_input_v63()
base_col = loader.small_col
timer.time("load csv in")

train_y = train["target"]
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
try_col = [c for c in train.columns if c not in drop_col and c not in base_col and c not in base_col]
print(len(try_col))
outliers = (train["target"] < -30).astype(int).values
split_num = 5
col_list = []
score_list = []
for i, c in enumerate(try_col):
    print(c)
    use_col = base_col + [c]
    train_x = train[use_col]

    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=4590)
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    for train_index, test_index in skf.split(train, outliers):
        X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
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
ret_df.to_csv("../output/eval_features.csv", index=False)

# base: 3.6407709483327446

