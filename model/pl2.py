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

# base score=3.636285831929972
# pl in a cv way
train, test = input_loader.GoldenLoader().load_team_input_v63()
test_pl = input_loader.GoldenLoader().load_pseudo_labels()

whole_sub = pd.DataFrame()
whole_sub["card_id"] = test["card_id"]
whole_sub["target"] = 0
whole_cv = pd.DataFrame()
whole_cv["card_id"] = train["card_id"]
whole_cv["cv_pred"] = 0

skf = model_selection.KFold(n_splits=5, shuffle=True, random_state=4590)
for pl_use_idx, pl_sub_idx in skf.split(test_pl):
    use_pl = test_pl.iloc[pl_use_idx]
    sub_pl = test_pl.iloc[pl_sub_idx]
    use_pl = pd.merge(use_pl, test, on="card_id", how="left")
    sub_pl = pd.merge(sub_pl, test, on="card_id", how="left")
    print(use_pl.shape)
    timer.time("load csv")

    drop_col = ["card_id", "target", "outliers"]
    # loader = input_loader.GoldenLoader()
    # pred_col = loader.team_small_col + loader.small_col
    # pred_col = [c for c in pred_col if c not in drop_col]
    pred_col = [c for c in train.columns if c not in drop_col]
    train_x = train[pred_col]
    train_y = train["target"]
    test_x = sub_pl[pred_col]
    pl_x = use_pl[pred_col]
    pl_y = use_pl["target"]
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)

    submission = pd.DataFrame()
    submission["card_id"] = sub_pl["card_id"]
    submission["_target"] = 0
    train_cv = pd.DataFrame()
    train_cv["card_id"] = train["card_id"]
    train_cv["_cv_pred"] = 0

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
            X_train = pd.concat([X_train, pl_x], axis=0)
            y_train = pd.concat([y_train, pl_y], axis=0)

            model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
            score = model.best_score["valid_0"]["rmse"]
            total_score += score
            y_pred = model.predict(test_x)
            valid_set_pred = model.predict(X_test)
            models.append(model)

            submission["_target"] = submission["_target"] + y_pred
            train_id = train.iloc[test_index]
            train_cv_prediction = pd.DataFrame()
            train_cv_prediction["card_id"] = train_id["card_id"]
            train_cv_prediction["_cv_pred"] = valid_set_pred
            train_preds.append(train_cv_prediction)
            timer.time("done one set in")

        train_output = pd.concat(train_preds, axis=0)
        train_cv["_cv_pred"] += train_output["_cv_pred"]

        lgb.show_feature_importance(models[0], path_const.FEATURE_GAIN)
        avg_score = str(total_score / split_num)
        logger.print("average score= " + avg_score)
        timer.time("end train in ")

    submission["_target"] = submission["_target"] / (bagging_num * split_num)
    whole_sub = pd.merge(whole_sub[["card_id", "target"]], submission, on="card_id", how="left")
    whole_sub["target"] = whole_sub["target"] + whole_sub["_target"].fillna(0)

    train_cv["_cv_pred"] = train_cv["_cv_pred"] / bagging_num
    whole_cv = pd.merge(whole_cv[["card_id", "cv_pred"]], train_cv, on="card_id", how="left")
    whole_cv["cv_pred"] = whole_cv["cv_pred"] + whole_cv["_cv_pred"]

    y_true = train_y
    y_pred = train_cv["_cv_pred"]
    rmse_score = evaluator.rmse(y_true, y_pred)
    logger.print("evaluator rmse score= " + str(rmse_score))

whole_sub = whole_sub[["card_id", "target"]]
whole_sub.to_csv(path_const.OUTPUT_SUB, index=False)

whole_cv = whole_cv[["card_id", "cv_pred"]]
whole_cv["cv_pred"] = whole_cv["cv_pred"] / 5
whole_cv.to_csv(path_const.OUTPUT_OOF, index=False)

y_true = train["target"]
y_pred = whole_cv["cv_pred"]
rmse_score = evaluator.rmse(y_true, y_pred)
logger.print("final rmse score= " + str(rmse_score))

print(train["target"].describe())
logger.print(whole_cv.describe())
logger.print(whole_sub.describe())
timer.time("done submission in ")

