import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import numpy as np
import pandas as pd
import datetime
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS)
older = csv_io.read_file(path_const.ORG_HISTORICAL)
timer.time("read csv")
print("-"*40)

new_card_id = ["C_ID_8c93e4f7bd", "C_ID_c075d82804"]
newer = newer[newer["card_id"].isin(new_card_id)]
newer.to_csv("../output/temp_new.csv")

old_card_id = ["C_ID_49e0712653", "C_ID_7e8f7e2ff2", "C_ID_cf58d5be10"]
older = older[older["card_id"].isin(old_card_id)]
older.to_csv("../output/temp_old.csv")

