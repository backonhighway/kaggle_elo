import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io
from elo.loader import input_loader

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train, test = input_loader.GoldenLoader().load_team_input_v63()
timer.time("load csv")

cat_cols = input_loader.GoldenLoader().get_team_cat_col()

for c in cat_cols:
    print(c)
    print(c in train.columns)

