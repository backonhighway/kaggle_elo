import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import numpy as np
import pandas as pd
import datetime
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.loader import input_loader

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train, test = input_loader.GoldenLoader().load_small_input()


