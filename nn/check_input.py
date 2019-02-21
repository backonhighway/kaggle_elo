import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.loader import input_loader

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train, test = input_loader.GoldenLoader().load_team_input_v63()
org_train = csv_io.read_file(path_const.ORG_TRAIN)[["card_id"]]
org_test = csv_io.read_file(path_const.ORG_TEST)[["card_id"]]
timer.time("load csv")
print(train["card_id"].head())
print(org_train["card_id"].head())


cat_cols = input_loader.GoldenLoader().get_team_cat_col()

has_col = [c for c in cat_cols if c in train.columns]
print(has_col)

print(train[has_col].nunique())
