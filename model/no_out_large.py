import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.utils import drop_col_util
from elo.loader import input_loader
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train, test = input_loader.GoldenLoader().load_team_input_v63()
timer.time("load csv")

drop_col = ["card_id", "target", "outliers"]
pred_col = [c for c in train.columns if c not in drop_col]
train_x = train[pred_col]
train_y = train["target"]
test_x = test[pred_col]
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
no_out_idx = train[train["target"] > -33].index
bagging_num = 1
split_num = 5
for bagging_index in range(bagging_num):
    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=4590)
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train, outliers):
        _train_idx = [i for i in train_index if i in no_out_idx]
        _test_idx = [i for i in test_index if i in no_out_idx]
        X_train, X_test = train_x.iloc[_train_idx], train_x.iloc[_test_idx]
        y_train, y_test = train_y.iloc[_train_idx], train_y.iloc[_test_idx]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        valid_set_pred = model.predict(train_x.iloc[test_index])
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
print(train_cv.shape)
train_cv.to_csv(path_const.OUTPUT_OOF, index=False)

y_true = train_y
y_pred = train_cv["cv_pred"]
rmse_score = evaluator.rmse(y_true, y_pred)
logger.print("evaluator rmse score= " + str(rmse_score))

print(train["target"].describe())
logger.print(train_cv.describe())
logger.print(submission.describe())
timer.time("done submission in ")

