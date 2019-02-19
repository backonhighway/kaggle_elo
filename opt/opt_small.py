import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.loader import input_loader
from sklearn import model_selection
import optuna

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

data = input_loader.GoldenLoader().load_small_input()
timer.time("load csv")

train, test, train_x, train_y, test_x = data
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

outliers = (train["target"] < -30).astype(int).values
split_num = 5
random_state = 4590
skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=random_state)
total_score = 0
models = []
train_preds = []


def objective(trial):
    num_leaves = trial.suggest_int("num_leaves", 16, 64)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 100)
    feature_fraction = trial.suggest_uniform("feature_fraction", 0.3, 0.9)
    bagging = trial.suggest_categorical("bagging", [True, False])
    l1 = trial.suggest_categorical("l1", [True, False])
    l2 = trial.suggest_categorical("l2", [True, False])
    max_bin = trial.suggest_int("max_bin", 30, 500)

    param = {
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'objective': 'regression',
        'max_depth': -1,
        'learning_rate': 0.01,
        "boosting": "gbdt",
        "feature_fraction": feature_fraction,
        "metric": 'rmse',
        "verbosity": -1,
        "seed": 99,
        "max_bin": max_bin
    }
    if bagging:
        param["bagging_freq"]: 1
        param["bagging_fraction"]: 0.9
        param["bagging_seed"]: 11
    if l1:
        param["lambda_l1"] = 0.1
    if l2:
        param["lambda_l2"] = 0.1

    return get_cv_score(param)


def get_cv_score(param):
    local_timer = pocket_timer.GoldenTimer(logger)
    lgb = pocket_lgb.OptLgb(param)
    for train_index, test_index in skf.split(train, outliers):
        X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        valid_set_pred = model.predict(X_test)

        train_id = train.iloc[test_index]
        train_cv_prediction = pd.DataFrame()
        train_cv_prediction["card_id"] = train_id["card_id"]
        train_cv_prediction["cv_pred"] = valid_set_pred
        train_preds.append(train_cv_prediction)

    train_output = pd.concat(train_preds, axis=0)
    local_timer.time("end train in ")
    train_output = pd.merge(train_output, train, on="card_id", how="left")
    score = evaluator.rmse(train_output["target"], train_output["cv_pred"])
    return score


timer.time("start study")
study = optuna.create_study()
study.optimize(objective, n_trials=100)
logger.print(study.best_params)
logger.print(study.best_value)
logger.print(study.best_trial)
timer.time("end study")


