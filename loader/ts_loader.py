import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.utils import drop_col_util
from elo.fe import jit_fe


class TsLoader:

    @staticmethod
    def load_ts():
        logger = pocket_logger.get_my_logger()
        timer = pocket_timer.GoldenTimer(logger)
        csv_io = pocket_file_io.GoldenCsv()

        num = csv_io.read_file(path_const.NEW_NUM)
        cat = csv_io.read_file(path_const.NEW_CAT)
        key = csv_io.read_file(path_const.NEW_KEY)
        timer.time("load ts")

    @staticmethod
    def load_ordered_small(key):
        ret_df = pd.DataFrame({"card_id": key})


    @staticmethod
    def load_small_input():
        logger = pocket_logger.get_my_logger()
        timer = pocket_timer.GoldenTimer(logger)
        csv_io = pocket_file_io.GoldenCsv()

        train = csv_io.read_file(path_const.TRAIN1)
        test = csv_io.read_file(path_const.TEST1)
        new_trans = csv_io.read_file(path_const.RE_NEW_TRANS1)
        old_trans = csv_io.read_file(path_const.RE_OLD_TRANS1)
        new_trans6 = csv_io.read_file(path_const.NEW_TRANS6)
        old_trans6 = csv_io.read_file(path_const.OLD_TRANS6)
        print(train.shape)
        print(test.shape)
        timer.time("load csv in ")

        train = pd.merge(train, new_trans, on="card_id", how="left")
        train = pd.merge(train, old_trans, on="card_id", how="left")
        train = pd.merge(train, new_trans6, on="card_id", how="left")
        train = pd.merge(train, old_trans6, on="card_id", how="left")
        #
        test = pd.merge(test, new_trans, on="card_id", how="left")
        test = pd.merge(test, old_trans, on="card_id", how="left")
        test = pd.merge(test, new_trans6, on="card_id", how="left")
        test = pd.merge(test, old_trans6, on="card_id", how="left")
        # print(train.shape)
        # print(test.shape)
        #
        fer = jit_fe.JitFe()
        train = fer.do_fe(train)
        test = fer.do_fe(test)

        train_y = train["target"]
        # 3.660 - 3.658
        use_col = [
            "new_trans_elapsed_days_max", "new_trans_elapsed_days_min", "new_trans_elapsed_days_mean",  # 0.001
            "old_trans_elapsed_days_max", "old_trans_elapsed_days_min", "old_trans_elapsed_days_mean",  # 0.025 mean001
            "new_last_day",  # 0.005
            "old_installments_sum", "old_installments_mean",  # 0.005
            "old_month_nunique", "old_woy_nunique",  # 0.010
            "old_merchant_id_nunique",  # 0.002
            "new_month_lag_mean", "old_month_lag_mean", "elapsed_days",  # 0.010
            "new_purchase_amount_max", "new_purchase_amount_count", "new_purchase_amount_mean",  # 0.020
            "old_purchase_amount_max", "old_purchase_amount_count", "old_purchase_amount_mean",  # 0.002
            "old_category_1_mean", "new_category_1_mean",  # 0.006
            "old_authorized_flag_sum",  # "old_authorized_flag_mean", bad?
            "old_no_city_purchase_amount_min",  # 0.003
            "old_no_city_purchase_amount_max", "old_no_city_purchase_amount_mean",  # 0.002
            "rec1_purchase_amount_count",  # 0.005
            "old_month_lag_max",  # 0.002
            "new_time_diff_mean", "new_trans_elapsed_days_std",  # 0.002
            "old_month_diff_mean", "old_pa2_month_diff_min",  # 0.004
        ]
        train_x = train[use_col]
        test_x = test[use_col]

        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        timer.time("prepare train in ")

        return train[["card_id", "target"]], test[["card_id"]], train_x, train_y, test_x
