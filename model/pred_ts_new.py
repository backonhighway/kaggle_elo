import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.loader import ts_input_loader
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

loader = ts_input_loader.GoldenLoader2()
train, test = loader.load_org_input()
ts_train, ts_test = loader.load_ts_input_new()
print(ts_train.shape)
print(ts_test.shape)
timer.time("load csv")

pred_col = loader.get_pred_col(ts_train.columns)
print(pred_col)
test_x = ts_test[pred_col]

pred_col_name = "pred_from_new_ts"
submission = pd.DataFrame()
submission["card_id"] = ts_test["card_id"]
submission[pred_col_name] = 0
train_cv = pd.DataFrame()
train_cv["card_id"] = ts_train["card_id"]
train_cv[pred_col_name] = 0

outliers = (train["target"] < -30).astype(int).values
bagging_num = 1
split_num = 5
random_state = 4590
for bagging_index in range(bagging_num):
    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=random_state)
    logger.print("random_state=" + str(random_state))
    lgb = pocket_lgb.TsLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train, outliers):
        train_id, test_id = train.iloc[train_index]["card_id"], train.iloc[test_index]["card_id"]
        _ts_train = ts_train[ts_train["card_id"].isin(train_id)]
        _ts_test = ts_train[ts_train["card_id"].isin(test_id)]
        X_train, X_test = _ts_train[pred_col], _ts_test[pred_col]
        y_train, y_test = _ts_train["target"], _ts_test["target"]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        valid_set_pred = model.predict(X_test)
        models.append(model)

        submission[pred_col_name] = submission[pred_col_name] + y_pred
        train_cv_prediction = pd.DataFrame()
        train_cv_prediction["card_id"] = _ts_test["card_id"]
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
submission.to_csv(path_const.NEW_TS_PRED_SUB, index=False)

train_cv[pred_col_name] = train_cv[pred_col_name] / bagging_num
train_cv.to_csv(path_const.NEW_TS_PRED_OOF, index=False)

y_true = ts_train["target"]
y_pred = train_cv[pred_col_name]
y_zero = np.zeros(y_true.shape)
print(y_true.shape)
print(y_zero.shape)
rmse_score = evaluator.rmse(y_true, y_zero)
logger.print("evaluator rmse score= " + str(rmse_score))
rmse_score = evaluator.rmse(y_true, y_pred)
logger.print("evaluator rmse score= " + str(rmse_score))

print(y_true.describe())
logger.print(train_cv.describe())
logger.print(submission.describe())
timer.time("done submission in ")

