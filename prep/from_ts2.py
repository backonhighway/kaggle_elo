import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import from_ts_fe2
import pandas as pd

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

new_oof = csv_io.read_file(path_const.NEW_TS_PRED_OOF2)
new_sub = csv_io.read_file(path_const.NEW_TS_PRED_SUB2)
new_df = pd.concat([new_oof, new_sub], axis=0)
old_oof = csv_io.read_file(path_const.OLD_TS_PRED_OOF2)
old_sub = csv_io.read_file(path_const.OLD_TS_PRED_SUB2)
old_df = pd.concat([old_oof, old_sub], axis=0)
timer.time("read csv")
print("-"*40)

fe = from_ts_fe2.FromTsFe2("new")
new_df = fe.do_fe(new_df)
print(new_df.shape)

fe = from_ts_fe2.FromTsFe2("old")
old_df = fe.do_fe(old_df)
print(old_df.shape)
timer.time("done fe")

csv_io.output_csv(new_df, path_const.FEAT_FROM_TS_NEW2)
csv_io.output_csv(old_df, path_const.FEAT_FROM_TS_OLD2)
timer.time("done output")


