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

data = input_loader.GoldenLoader().load_small_pred_new()
train, test, train_x, train_y, test_x = data
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(train_y.describe())
train_y = train_y.fillna(60)
print(train_x.columns)
timer.time("load csv")

mean_val = 6.040527
train_x["mean_val"] = mean_val
rmse_score = evaluator.rmse(train_y, train_x["mean_val"])
print(rmse_score)

pred_col_name = "pred_new"
submission = pd.DataFrame()
submission["card_id"] = test["card_id"]
submission[pred_col_name] = 0
train_cv = pd.DataFrame()
train_cv["card_id"] = train["card_id"]
train_cv[pred_col_name] = 0

outliers = (train["target"] < -30).astype(int).values
bagging_num = 1
split_num = 5
random_state = 4590
for bagging_index in range(bagging_num):
    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=random_state)
    logger.print("random_state=" + str(random_state))
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

        submission[pred_col_name] = submission[pred_col_name] + y_pred
        train_id = train.iloc[test_index]
        train_cv_prediction = pd.DataFrame()
        train_cv_prediction["card_id"] = train_id["card_id"]
        train_cv_prediction[pred_col_name] = valid_set_pred
        train_preds.append(train_cv_prediction)
        timer.time("done one set in")

    train_output = pd.concat(train_preds, axis=0)
    train_cv[pred_col_name] += train_output[pred_col_name]

    lgb.show_feature_importance(models[0], path_const.FEATURE_GAIN)
    avg_score = str(total_score / split_num)
    logger.print("average score= " + avg_score)
    timer.time("end train in ")


submission[pred_col_name] = submission[pred_col_name] / (bagging_num * split_num)
submission.to_csv(path_const.NEW_DAY_PRED_SUB, index=False)

train_cv[pred_col_name] = train_cv[pred_col_name] / bagging_num
train_cv.to_csv(path_const.NEW_DAY_PRED_OOF, index=False)

y_true = train_y
y_pred = train_cv[pred_col_name]
rmse_score = evaluator.rmse(y_true, y_pred)
logger.print("evaluator rmse score= " + str(rmse_score))

print(train_y.describe())
logger.print(train_cv.describe())
logger.print(submission.describe())
timer.time("done submission in ")

