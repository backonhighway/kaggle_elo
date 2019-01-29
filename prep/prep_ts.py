import numpy as np

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import reshape_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

new_ts = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, parse_date=["purchase_date"])
timer.time("load csv")

timer.time("start reshape")
reshaper = reshape_fe.GoldenReshaper(split_num=16)
new_num, new_cat, new_key = reshaper.do_para_reshape(new_ts)
timer.time("done reshape")

csv_io.output_npy(new_num, path_const.NEW_NUM)
csv_io.output_npy(new_cat, path_const.NEW_CAT)
csv_io.output_npy(new_key, path_const.NEW_KEY)
timer.time("done output")

