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
old_trans3 = csv_io.read_file(path_const.OLD_TRANS3)
new_trans6 = csv_io.read_file(path_const.NEW_TRANS6)
old_trans6 = csv_io.read_file(path_const.OLD_TRANS6)
old_trans9 = csv_io.read_file(path_const.OLD_TRANS9)
print(train.shape)
print(test.shape)
timer.time("load csv in ")

train = pd.merge(train, new_trans, on="card_id", how="left")
train = pd.merge(train, old_trans, on="card_id", how="left")
train = pd.merge(train, old_trans3, on="card_id", how="left")
train = pd.merge(train, new_trans6, on="card_id", how="left")
train = pd.merge(train, old_trans6, on="card_id", how="left")
train = pd.merge(train, old_trans9, on="card_id", how="left")
#
test = pd.merge(test, new_trans, on="card_id", how="left")
test = pd.merge(test, old_trans, on="card_id", how="left")
test = pd.merge(test, old_trans3, on="card_id", how="left")
test = pd.merge(test, new_trans6, on="card_id", how="left")
test = pd.merge(test, old_trans6, on="card_id", how="left")
test = pd.merge(test, old_trans9, on="card_id", how="left")

print(train.shape)
print(test.shape)
#
fer = jit_fe.JitFe()
train = fer.do_fe(train)
test = fer.do_fe(test)

pred_train = csv_io.read_file(path_const.NEW_DAY_PRED_OOF)
pred_test = csv_io.read_file(path_const.NEW_DAY_PRED_SUB)
train = pd.merge(train, pred_train, on="card_id", how="left")
train["pred_diff"] = train["pred_new"] - train["new_to_last_day"]
test = pd.merge(test, pred_test, on="card_id", how="left")
test["pred_diff"] = test["pred_new"] - test["new_to_last_day"]

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
    # "old_null_merchant", "new_null_merchant",
    "old_ym_target_encode_mean", "new_ym_target_encode_mean",
    "old_hour_target_encode_mean", "new_hour_target_encode_mean",
    # "old_subsector_id_target_encode_mean",
    # "new_merchant_id_target_encode_mean", "old_merchant_id_target_encode_mean",
    "pred_new", "old_same_buy_count", "old_purchase_amount_nunique", "new_purchase_amount_nunique",
    "old_installments_nunique", "new_installments_nunique",
]
train_x = drop_col_util.drop_col(train, drop_col)
test_x = drop_col_util.drop_col(test, drop_col)

# drop_col_like = [
#     "new_month_lag"
# ]
# train_x = drop_col_util.drop_col_like(train_x, drop_col_like)
# test_x = drop_col_util.drop_col_like(test_x, drop_col_like)

timer.time("prepare train in ")
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)


submission = pd.DataFrame()
submission["card_id"] = test["card_id"]
submission["target"] = 0
train_cv = pd.DataFrame()
train_cv["card_id"] = train["card_id"]
train_cv["cv_pred"] = 0

outliers = (train["target"] < -30).astype(int).values
bagging_num = 1
split_num = 4
for bagging_index in range(bagging_num):
    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=99 * bagging_index)
    logger.print("random_state=" + str(99*bagging_index))
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train, outliers):
        X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        valid_set_pred = model.predict(X_test)
        models.append(model)

        submission["target"] = submission["target"] + y_pred
        train_id = train.iloc[test_index]
        train_cv_prediction = pd.DataFrame()
        train_cv_prediction["card_id"] = train_id["card_id"]
        train_cv_prediction["cv_pred"] = valid_set_pred
        train_preds.append(train_cv_prediction)
        timer.time("done one set in")

    train_output = pd.concat(train_preds, axis=0)
    train_cv["cv_pred"] += train_output["cv_pred"]

    lgb.show_feature_importance(models[0], path_const.FEATURE_GAIN)
    avg_score = str(total_score / split_num)
    logger.print("average score= " + avg_score)
    timer.time("end train in ")


submission["target"] = submission["target"] / (bagging_num * split_num)
submission.to_csv(path_const.OUTPUT_SUB, index=False)

train_cv["cv_pred"] = train_cv["cv_pred"] / bagging_num
train_cv.to_csv(path_const.OUTPUT_OOF, index=False)

y_true = train_y
y_pred = train_cv["cv_pred"]
rmse_score = evaluator.rmse(y_true, y_pred)
logger.print("evaluator rmse score= " + str(rmse_score))

print(train["target"].describe())
logger.print(train_cv.describe())
logger.print(submission.describe())
timer.time("done submission in ")

