import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import lda_fe
import pandas as pd
from sklearn import preprocessing

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

# newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, nrows=1000*100, parse_date=["purchase_date"])
# older = csv_io.read_file(path_const.ORG_HISTORICAL, nrows=1000*1000, parse_date=["purchase_date"])
newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, parse_date=["purchase_date"])
older = csv_io.read_file(path_const.ORG_HISTORICAL, parse_date=["purchase_date"])
timer.time("read csv")
print("-"*40)

all_df = pd.concat([older, newer], axis=0)
print(newer.shape)
print(older.shape)
print(all_df.shape)
print(all_df.describe())

le_list = list()
for col in ["card_id"]:
    all_df[col] = all_df[col].fillna("null")
    le = preprocessing.LabelEncoder()
    le.fit(list(all_df[col].values.astype('str')))
    all_df[col] = le.transform(all_df[col].values.astype('str'))
    le_list.append(le)
all_df = all_df.sort_values(["card_id", "purchase_date"])
target_cols = ["merchant_category_id", "state_id", "month_lag"]
use_col = ["card_id"] + target_cols
all_df = all_df[use_col]
timer.time("done prep")

lda_df = lda_fe.GoldenLDA(timer).create_features(all_df, target_cols)
lda_df["card_id"] = le_list[0].inverse_transform(lda_df["card_id"])  # warning is due to sklearn version
print(lda_df.head())
print(lda_df.describe())
timer.time("done fe")

csv_io.output_csv(lda_df, path_const.LDA_ALL2)
timer.time("done output")


