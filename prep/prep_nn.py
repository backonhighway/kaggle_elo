import numpy as np

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.TRAIN1)
trans = csv_io.read_file(path_const.TRANSACTIONS1)
timer.time("load csv")

train = train.sort_values(by=["card_id"])
trans = trans.sort_values(by=["card_id", "purchase_date"])

from elo.common import pocket_scaler
train_scale_col = ["elapsed_days"]
train_no_scale_col = [c for c in train.columns if c not in train_scale_col]
train = pocket_scaler.scale_df(train, train_scale_col, train_no_scale_col)

trans_scale_col = ["purchase_amount", "installments", "purchase_date"]
trans_no_scale_col = [
    "card_id", "most_recent_sales_range", "most_recent_purchases_range",
]
trans = pocket_scaler.scale_df(trans, trans_scale_col, trans_no_scale_col)

from sklearn import model_selection
train_, holdout_ = model_selection.train_test_split(train, test_size=0.2, random_state=99)
train_id, holdout_id = train_["card_id"], holdout_["card_id"]
train_trans_ = trans[trans["card_id"].isin(train_id)].copy()
holdout_trans_ = trans[trans["card_id"].isin(holdout_id)].copy()

timer.time("start reshape")
from elo.common import pocket_reshaper
# reshaper = pocket_reshaper.GoldenReshaper()
# trans_cat_col = [c for c in trans_no_scale_col if c != "card_id"]
# train_trans_num = reshaper.reshape(train_trans_, trans_scale_col)
# train_trans_cat = reshaper.reshape(train_trans_, trans_cat_col)
# hold_trans_num = reshaper.reshape(holdout_trans_, trans_scale_col)
# hold_trans_cat = reshaper.reshape(holdout_trans_, trans_cat_col)
trans_cat_col = [c for c in trans_no_scale_col if c != "card_id"]
reshaper = pocket_reshaper.GoldenReshaper(cat_col=trans_cat_col, num_col=trans_scale_col)
train_trans_num, train_trans_cat, hold_trans_num, hold_trans_cat = reshaper.do_para_reshape(train_trans_, holdout_trans_)
timer.time("prep data")

csv_io.output_csv(train_, path_const.TRAIN_)
csv_io.output_csv(holdout_, path_const.HOLD_)
csv_io.output_npy(train_trans_num, path_const.TT_NUM)
csv_io.output_npy(train_trans_cat, path_const.TT_CAT)
csv_io.output_npy(hold_trans_num, path_const.HT_NUM)
csv_io.output_npy(hold_trans_cat, path_const.HT_CAT)
timer.time("done output")

