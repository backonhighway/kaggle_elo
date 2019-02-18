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


def show_col(the_path):
    df = csv_io.read_file(the_path)
    timer.time("read csv")
    print(df.columns)
    print(df.describe())
    print("-" * 40)


show_col(path_const.OLD_TRANS13)
show_col(path_const.NEW_TRANS13)

