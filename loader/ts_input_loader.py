import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const


class GoldenLoader2:
    def __init__(self):
        self.drop_for_pred_col = [
            "card_id", "target", "purchase_date",
            "merchant_id", "first_active_month"
        ]

    @staticmethod
    def load_org_input():
        csv_io = pocket_file_io.GoldenCsv()
        train = csv_io.read_file(path_const.ORG_TRAIN)
        test = csv_io.read_file(path_const.ORG_TEST)
        return train, test

    @staticmethod
    def load_ts_input_old():
        csv_io = pocket_file_io.GoldenCsv()
        train = csv_io.read_file(path_const.TS_OLD_TRAIN)
        test = csv_io.read_file(path_const.TS_OLD_TEST)
        return train, test

    @staticmethod
    def load_ts_input_new():
        csv_io = pocket_file_io.GoldenCsv()
        train = csv_io.read_file(path_const.TS_NEW_TRAIN)
        test = csv_io.read_file(path_const.TS_NEW_TEST)
        return train, test

    def get_pred_col(self, org_cols):
        return [c for c in org_cols if c not in self.drop_for_pred_col]
