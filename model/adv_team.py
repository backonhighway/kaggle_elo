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

drop_col = [
    "card_id", "target", "outliers",
    "old_subsector_id_target_encode_mean", "old_merchant_id_target_encode_mean"
]
pred_col = [c for c in train.columns if c not in drop_col]
train_x = train[pred_col]
train_y = train["target"]
test_x = test[pred_col]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)

train["tt"] = 0
test["tt"] = 1
train_y = train["tt"]
test_y = test["tt"]
all_x = pd.concat([train_x, test_x], axis=0)
all_y = pd.concat([train_y, test_y], axis=0)
timer.time("prepare train in ")

bagging_num = 1
split_num = 4
for bagging_index in range(bagging_num):
    skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=99 * bagging_index)
    logger.print("random_state=" + str(99*bagging_index))
    lgb = pocket_lgb.AdversarialLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(all_x, all_y):
        X_train, X_test = all_x.iloc[train_index], all_x.iloc[test_index]
        y_train, y_test = all_y.iloc[train_index], all_y.iloc[test_index]

        model = lgb.do_train_direct(X_train, X_test, y_train, y_test)
        score = model.best_score["valid_0"]["auc"]
        total_score += score
        y_pred = model.predict(test_x)
        valid_set_pred = model.predict(X_test)
        models.append(model)
        timer.time("done one set in")

    lgb.show_feature_importance(models[0], path_const.FEATURE_GAIN)
    avg_score = str(total_score / split_num)
    logger.print("average score= " + avg_score)
    timer.time("end train in ")




