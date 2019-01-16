import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb
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
train_x = train.drop(columns=["card_id", "target"])
test_x = test.drop(columns=["card_id"])
timer.time("prepare train in ")

submission = pd.DataFrame()
submission["card_id"] = test["card_id"]
submission["target"] = 0
train_cv = pd.DataFrame()
train_cv["card_id"] = train["card_id"]
train_cv["cv_pred"] = 0

bagging_num = 2
split_num = 5
for bagging_index in range(bagging_num):
    skf = model_selection.KFold(n_splits=split_num, shuffle=True, random_state=71 * bagging_index)
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train):
        X_train, X_test = train_x[train_index], train_x[test_index]
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

