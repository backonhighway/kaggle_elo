import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb
from elo.utils import drop_col_util
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.TRAIN1)
test = csv_io.read_file(path_const.TEST1)
new_trans = csv_io.read_file(path_const.NEW_TRANS1)
old_trans = csv_io.read_file(path_const.OLD_TRANS1)
timer.time("load csv in ")

train = pd.merge(train, new_trans, on="card_id", how="left")
train = pd.merge(train, old_trans, on="card_id", how="left")
test = pd.merge(test, new_trans, on="card_id", how="left")
test = pd.merge(test, old_trans, on="card_id", how="left")

train_y = train["target"]
drop_col = [
    "card_id", "target", "feature_1", "feature_2", "feature_3",
    "old_weekend_mean", "new_weekend_mean", "new_authorized_flag_mean",
    "old_null_state", "new_null_state", "new_null_install", #"old_null_install",
]
# from elo.common import pred_cols
# for c in pred_cols.CAT_COLS:
#     train[c] = np.where(train[c] < 0, 0, train[c]+1)
#     test[c] = np.where(test[c] < 0, 0, test[c]+1)
train_x = drop_col_util.drop_col(train, drop_col)
test_x = drop_col_util.drop_col(test, drop_col)
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

bagging_num = 1
split_num = 4
for bagging_index in range(bagging_num):
    skf = model_selection.KFold(n_splits=split_num, shuffle=True, random_state=99 * bagging_index)
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train):
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

print(train["target"].describe())
logger.print(train_cv.describe())
logger.print(submission.describe())
timer.time("done submission in ")

