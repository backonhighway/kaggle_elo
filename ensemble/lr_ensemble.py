import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.utils import drop_col_util
from sklearn.linear_model import LinearRegression

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.ORG_TRAIN)[["card_id", "target"]]
train1 = csv_io.read_file("../sub/big_oof.csv")
test1 = csv_io.read_file("../sub/big_sub.csv")
train2 = csv_io.read_file("../sub/small_oof.csv")
test2 = csv_io.read_file("../sub/small_sub.csv")
timer.time("load csv in ")

print(train.shape)
train1.columns = ["card_id", "big"]
train2.columns = ["card_id", "small"]
train = pd.merge(train, train1, on="card_id", how="inner")
train = pd.merge(train, train2, on="card_id", how="inner")
print(train.shape)
print("-----")

print("co-eff...")
print(train[["target", "big", "small"]].corr())

print("before score..")
score = evaluator.rmse(train["target"], train["big"])
print(score)
score = evaluator.rmse(train["target"], train["small"])
print(score)
print("-----")

train_x = train[["big", "small"]]
reg = LinearRegression().fit(train_x, train["target"])
y_pred = reg.predict(train_x)
score = evaluator.rmse(train["target"], y_pred)
print(score)




