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

g = newer.groupby("state_id")["category_2"].value_counts()
print(g)
print("-----")
g = older.groupby("state_id")["category_2"].value_counts()
print(g)
print("-----")

g = newer.groupby("installments")["category_3"].value_counts()
print(g)
print("-----")
g = older.groupby("installments")["category_3"].value_counts()
print(g)
print("-----")

x = newer[newer["category_1"] == "Y"]
g = x.groupby("city_id")["category_1"].value_counts()
print(g)
print("-----")
g = older.groupby("city_id")["category_1"].value_counts()
print(g)
print("-----")
