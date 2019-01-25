import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import numpy as np
import pandas as pd
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import evaluator

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.TRAIN1)
oof_sub = csv_io.read_file(path_const.OUTPUT_OOF)
timer.time("read csv")

y_true = train["target"]
y_pred = oof_sub["cv_pred"]
score = evaluator.rmse(y_true, y_pred)
print(score)

train_df = pd.merge(train, oof_sub, on="card_id", how="inner")
train_df["diff"] = train_df["target"] - train_df["cv_pred"]
print(train_df["diff"].describe())

