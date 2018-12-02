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

trans = pd.concat([newer, older], axis=0)
g = trans.groupby("card_id")["merchant_id"].count().reset_index()
print(g.describe(percentiles=[.25, .5, .75, .90, .95, .99]))
print("-----")


"""
         merchant_id
count  325540.000000
mean       94.952064
std       107.134314
min         2.000000
25%        30.000000
50%        60.000000
75%       118.000000
90%       213.000000
95%       299.000000
99%       527.000000
max      5582.000000
-----
"""