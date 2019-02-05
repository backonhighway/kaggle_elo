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
            "old_mer_cnt_whole_mean",  # 0.001
        ]
        self.medium_col = self.small_col + ["pred_diff"]

        self.drop_col = [
            "card_id", "target",  # "feature_1", "feature_2", "feature_3",
            "old_weekend_mean", "new_weekend_mean", "new_authorized_flag_mean",
            "old_null_state", "new_null_state", "new_null_install",  # "old_null_install",
            "old_cat3_pur_mean", "new_cat3_pur_mean", "old_cat2_pur_mean", "new_cat2_pur_mean",
            "new_category_4_mean",  # "new_merchant_group_id_nunique", "old_merchant_group_id_nunique"
            "new_mon_nunique_mean", "new_woy_nunique_mean",
            # "new_month_lag_ptp", "new_month_lag_min",
            "new_purchase_amount_skew",  # "new_purchase_amount_std",
            "old_purchase_amount_skew",  # "old_purchase_amount_std",
            # "new_category_2_nunique", "old_category_2_nunique",
            # "old_null_merchant", "new_null_merchant",
            # "old_ym_target_encode_mean", "new_ym_target_encode_mean",
            # "old_hour_target_encode_mean", "new_hour_target_encode_mean",
            # "old_subsector_id_target_encode_mean",
            # "new_merchant_id_target_encode_mean", "old_merchant_id_target_encode_mean",
            "pred_new", "old_same_buy_count", "old_purchase_amount_nunique", "new_purchase_amount_nunique",
            "old_installments_nunique", "new_installments_nunique", "pred_new"
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

    def load_medium_input(self):
        train, test = self.load_whole_input()
        train_y = train["target"]
        use_col = self.medium_col
        train_x = train[use_col]
        test_x = test[use_col]

        print(train_x.shape)
        print(train_y.shape)
        print(test_x.shape)
        self.timer.time("prepare train in ")

        return train[["card_id", "target"]], test[["card_id"]], train_x, train_y, test_x

    def load_small_pred_new(self):
        train, test = self.load_whole_input(use_pred=False)
        pred_col = [c for c in self.small_col if "new" not in c]
        train_x = train[pred_col]
        test_x = test[pred_col]
        train_y = train["new_to_last_day"]

        return train[["card_id", "target"]], test[["card_id"]], train_x, train_y, test_x

    def load_large_input(self):
        train, test = self.load_whole_input()

        train_y = train["target"]
        train_x = drop_col_util.drop_col(train, self.drop_col)
        test_x = drop_col_util.drop_col(test, self.drop_col)

        return train[["card_id", "target"]], test[["card_id"]], train_x, train_y, test_x

    def load_for_share(self):
        train, test = self.load_whole_input()

        drop_col = self.drop_col.copy()
        drop_col.remove("card_id")
        drop_col.remove("target")
        train = drop_col_util.drop_col(train, drop_col)
        test = drop_col_util.drop_col(test, drop_col)
        return train, test

    def load_small_for_share(self):
        train, test = self.load_whole_input()

        use_col = ["card_id"] + self.medium_col
        test = test[use_col]
        use_col = ["target"] + use_col
        train = train[use_col]
        self.timer.time("prepare train in ")

        return train, test

    def load_whole_input(self, use_pred=True):
        logger = pocket_logger.get_my_logger()
        timer = pocket_timer.GoldenTimer(logger)
        csv_io = pocket_file_io.GoldenCsv()

        train = csv_io.read_file(path_const.ORG_TRAIN)[["card_id"]]
        test = csv_io.read_file(path_const.ORG_TEST)[["card_id"]]
        train_test_files = [
            (path_const.TRAIN1, path_const.TEST1),
            (path_const.NEW_DAY_PRED_OOF, path_const.NEW_DAY_PRED_SUB),
        ]
        if not use_pred:
            train_test_files = [
                (path_const.TRAIN1, path_const.TEST1),
            ]
        use_files = [
            path_const.RE_NEW_TRANS1,
            path_const.RE_OLD_TRANS1,
            path_const.OLD_TRANS3,
            path_const.NEW_TRANS6,
            path_const.OLD_TRANS6,
            path_const.OLD_TRANS9,
            path_const.NEW_TRANS11,
            path_const.OLD_TRANS11,
        ]
        for f in train_test_files:
            train, test = self.load_train_test_and_merge(train, test, f[0], f[1], csv_io)
        for f in use_files:
            train, test = self.load_file_and_merge(train, test, f, csv_io)

        print(train.shape)
        print(test.shape)
        timer.time("load csv in ")

        fer = jit_fe.JitFe()
        train = fer.do_fe(train)
        test = fer.do_fe(test)
        return train, test

    @staticmethod
    def load_file_and_merge(train, test, file_path, csv_io):
        new_file = csv_io.read_file(file_path)
        train = pd.merge(train, new_file, on="card_id", how="left")
        test = pd.merge(test, new_file, on="card_id", how="left")
        return train, test

    @staticmethod
    def load_train_test_and_merge(org_train, org_test, path_train, path_test, csv_io):
        new_train = csv_io.read_file(path_train)
        new_test = csv_io.read_file(path_test)
        train = pd.merge(org_train, new_train, on="card_id", how="left")
        test = pd.merge(org_test, new_test, on="card_id", how="left")
        return train, test



