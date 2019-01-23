import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
import pandas as pd

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.ORG_TRAIN, parse_date=["first_active_month"])
test = csv_io.read_file(path_const.ORG_TEST, parse_date=["first_active_month"])
newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, parse_date=["purchase_date"])
older = csv_io.read_file(path_const.ORG_HISTORICAL, parse_date=["purchase_date"])
timer.time("read csv")
print("-"*40)

newer = newer.groupby("card_id")["purchase_date"].agg("max").reset_index()
newer.columns = ["card_id", "last_day"]
older = older.groupby("card_id")["purchase_date"].agg("min").reset_index()
older.columns = ["card_id", "first_day"]

_train = pd.merge(train, newer, on="card_id", how="left")
_train = pd.merge(_train, older, on="card_id", how="left")
print(_train.shape)

pd.set_option('display.max_columns', 20)
cnt = 0
for idx, row in _train.iterrows():
    mask = (_train["feature_1"] == row["feature_1"]) & (_train["feature_2"] == row["feature_2"])\
           & (_train["feature_3"] == row["feature_3"]) & (_train["first_active_month"] == row["first_active_month"])\
           & (_train["last_day"] <= row["first_day"])
    dups = _train[mask]
    len_df = len(dups)
    if len_df > 0:
        print(row)
        print(dups)
        cnt += 1
        print("-----")
        if cnt > 10:
            exit(0)

