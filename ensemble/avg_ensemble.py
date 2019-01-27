import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from elo.utils import drop_col_util
from sklearn import model_selection

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.ORG_TRAIN)[["card_id", "target"]]
train1 = csv_io.read_file("../sub/big_oof.csv")
test1 = csv_io.read_file("../sub/big_sub.csv")
train2 = csv_io.read_file("../sub/small_oof.csv")
test2 = csv_io.read_file("../sub/small_sub.csv")
timer.time("load csv in ")

train1.columns = ["card_id", "big"]
train2.columns = ["card_id", "small"]
train = pd.merge(train, train1, on="card_id", how="inner")
train = pd.merge(train, train2, on="card_id", how="inner")

train["avg"] = (train["big"] + train["small"]) / 2

score = evaluator.rmse(train["target"], train["big"])
print(score)
score = evaluator.rmse(train["target"], train["small"])
print(score)

score = evaluator.rmse(train["target"], train["avg"])
print(score)
