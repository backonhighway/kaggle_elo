import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io
from elo.loader import input_loader

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train, test = input_loader.GoldenLoader().load_small_for_share()
timer.time("load csv")

train.to_pickle("../output/pocket_train_small_feats.pkl")
test.to_pickle("../output/pocket_test_small_feats.pkl")


