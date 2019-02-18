import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import numpy as np
import pandas as pd
import datetime
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const, evaluator

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()


def merge_it(org, df):
    ret_df = pd.merge(org, df, on="card_id", how="left")
    return ret_df


new = csv_io.read_file(path_const.FEAT_FROM_TS_NEW)
old = csv_io.read_file(path_const.FEAT_FROM_TS_OLD)
new2 = csv_io.read_file(path_const.FEAT_FROM_TS_NEW2)
old2 = csv_io.read_file(path_const.FEAT_FROM_TS_OLD2)
train = csv_io.read_file(path_const.ORG_TRAIN)[["card_id", "target"]]

train = merge_it(train, new)
train = merge_it(train, old)
train = merge_it(train, new2)
train = merge_it(train, old2)

eval_cols = [
    'pred_from_new_ts_mean', 'pred_from_old_ts_mean',
    'pred_from_new_ts2_mean', 'pred_from_old_ts2_mean',
]
print(train.describe())
print(train.shape)
train = train[train["pred_from_new_ts2_mean"].notnull()]
print(train.shape)

for c in eval_cols:
    score = evaluator.rmse(train["target"], train[c])
    print(c, "=", score)

