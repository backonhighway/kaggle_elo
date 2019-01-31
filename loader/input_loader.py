import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.utils import drop_col_util
from elo.fe import jit_fe


class GoldenLoader:
    def __init__(self):
        self.small_col = [
            # "new_trans_elapsed_days_max", "new_trans_elapsed_days_min", "new_trans_elapsed_days_mean",  # 0.001
            "old_trans_elapsed_days_max", "old_trans_elapsed_days_min", "old_trans_elapsed_days_mean",  # 0.025 mean001
            # "new_last_day",  # 0.005
            "new_to_last_day",
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
            # "pred_diff"
        ]

        self.logger = pocket_logger.get_my_logger()
        self.timer = pocket_timer.GoldenTimer(self.logger)

    def load_small_input(self):
        train, test = self.load_whole_input()
        train_y = train["target"]
        use_col = self.small_col
        train_x = train[use_col]
        test_x = test[use_col]

        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        self.timer.time("prepare train in ")

        return train[["card_id", "target"]], test[["card_id"]], train_x, train_y, test_x

    def load_small_pred_new(self):
        train, test = self.load_whole_input()
        pred_col = [c for c in self.small_col if "new" not in c]
        train_x = train[pred_col]
        test_x = test[pred_col]
        train_y = train["new_to_last_day"]

        return train[["card_id", "target"]], test[["card_id"]], train_x, train_y, test_x

    def load_whole_input(self):
        csv_io = pocket_file_io.GoldenCsv()

        train = csv_io.read_file(path_const.TRAIN1)
        test = csv_io.read_file(path_const.TEST1)
        new_trans = csv_io.read_file(path_const.RE_NEW_TRANS1)
        old_trans = csv_io.read_file(path_const.RE_OLD_TRANS1)
        new_trans6 = csv_io.read_file(path_const.NEW_TRANS6)
        old_trans6 = csv_io.read_file(path_const.OLD_TRANS6)

        print(train.shape)
        print(test.shape)
        self.timer.time("load csv in ")

        train = pd.merge(train, new_trans, on="card_id", how="left")
        train = pd.merge(train, old_trans, on="card_id", how="left")
        train = pd.merge(train, new_trans6, on="card_id", how="left")
        train = pd.merge(train, old_trans6, on="card_id", how="left")
        #
        test = pd.merge(test, new_trans, on="card_id", how="left")
        test = pd.merge(test, old_trans, on="card_id", how="left")
        test = pd.merge(test, new_trans6, on="card_id", how="left")
        test = pd.merge(test, old_trans6, on="card_id", how="left")

        # pred_train = csv_io.read_file(path_const.NEW_DAY_PRED_OOF)
        # pred_test = csv_io.read_file(path_const.NEW_DAY_PRED_SUB)
        # train = pd.merge(train, pred_train, on="card_id", how="left")
        # train["pred_diff"] = train["pred_new"] - train["new_to_last_day"]
        # test = pd.merge(test, pred_test, on="card_id", how="left")
        # test["pred_diff"] = test["pred_new"] - test["new_to_last_day"]

        # print(train.shape)
        # print(test.shape)
        #
        fer = jit_fe.JitFe()
        train = fer.do_fe(train)
        test = fer.do_fe(test)
        return train, test

    @staticmethod
    def load_large_input():
        logger = pocket_logger.get_my_logger()
        timer = pocket_timer.GoldenTimer(logger)
        csv_io = pocket_file_io.GoldenCsv()

        train = csv_io.read_file(path_const.TRAIN1)
        test = csv_io.read_file(path_const.TEST1)
        new_trans = csv_io.read_file(path_const.RE_NEW_TRANS1)
        old_trans = csv_io.read_file(path_const.RE_OLD_TRANS1)
        old_trans3 = csv_io.read_file(path_const.OLD_TRANS3)
        new_trans6 = csv_io.read_file(path_const.NEW_TRANS6)
        old_trans6 = csv_io.read_file(path_const.OLD_TRANS6)
        old_trans9 = csv_io.read_file(path_const.OLD_TRANS9)
        print(train.shape)
        print(test.shape)
        timer.time("load csv in ")

        train = pd.merge(train, new_trans, on="card_id", how="left")
        train = pd.merge(train, old_trans, on="card_id", how="left")
        train = pd.merge(train, old_trans3, on="card_id", how="left")
        train = pd.merge(train, new_trans6, on="card_id", how="left")
        train = pd.merge(train, old_trans6, on="card_id", how="left")
        train = pd.merge(train, old_trans9, on="card_id", how="left")
        #
        test = pd.merge(test, new_trans, on="card_id", how="left")
        test = pd.merge(test, old_trans, on="card_id", how="left")
        test = pd.merge(test, old_trans3, on="card_id", how="left")
        test = pd.merge(test, new_trans6, on="card_id", how="left")
        test = pd.merge(test, old_trans6, on="card_id", how="left")
        test = pd.merge(test, old_trans9, on="card_id", how="left")
        # print(train.shape)
        # print(test.shape)
        #
        fer = jit_fe.JitFe()
        train = fer.do_fe(train)
        test = fer.do_fe(test)

        train_y = train["target"]
        drop_col = [
            "card_id", "target",  # "feature_1", "feature_2", "feature_3",
            "old_weekend_mean", "new_weekend_mean", "new_authorized_flag_mean",
            "old_null_state", "new_null_state", "new_null_install", #"old_null_install",
            "old_cat3_pur_mean", "new_cat3_pur_mean", "old_cat2_pur_mean", "new_cat2_pur_mean",
            "new_category_4_mean",  # "new_merchant_group_id_nunique", "old_merchant_group_id_nunique"
            "new_mon_nunique_mean", "new_woy_nunique_mean",
            # "new_month_lag_ptp", "new_month_lag_min",
            "new_purchase_amount_skew",  # "new_purchase_amount_std",
            "old_purchase_amount_skew",  # "old_purchase_amount_std",
            # "new_category_2_nunique", "old_category_2_nunique",
            # "old_null_merchant", "new_null_merchant",
            "old_ym_target_encode_mean", "new_ym_target_encode_mean",
            "old_hour_target_encode_mean", "new_hour_target_encode_mean",
            "old_subsector_id_target_encode_mean",
            "new_merchant_id_target_encode_mean", "old_merchant_id_target_encode_mean",
        ]
        train_x = drop_col_util.drop_col(train, drop_col)
        test_x = drop_col_util.drop_col(test, drop_col)

        return train_x, train_y, test_x


