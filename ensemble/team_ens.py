# -*- coding: utf-8 -*-
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn import model_selection


class GoldenLr:
    def __init__(self):
        self.csv_io = pocket_file_io.GoldenCsv()

    def doit(self):
        logger = pocket_logger.get_my_logger()
        timer = pocket_timer.GoldenTimer(logger)

        # (file_name, col_name)
        files = [
            ("team_v63", "lgb"),
            ("bin_team", "bin"),
            ("no_out_team", "no_out"),
            # ("rnd_feat_bridge", "rnd_feat"),
            ("small_team", "small")
        ]

        team_files = [
            'select_v44_ridge',
            'tune_stack_57_v1',
            'select_v51_ridge',
            # 'tune_stack_57_2_v1',
            'tune_stack_cgb_v1',
            # 'elo_rnd_feat_bridge',

            'outlier_lgb_v3_kh_time_feature2_pocket',
            # 'delete_outlier_kh_pocket_stack_correct_ridge',
            # 'outlier_lgb_pocket_logistic',
            'delete_outlier_kh_pocket_stack_correct2_ridge'
        ]
        team_files = [(t, t) for t in team_files]

        train, test = self.make_files(files)
        for f in team_files:
            train, test = self.add_team_file(f[0], f[1], train, test)
        timer.time("load csv in ")
        print(train.describe())

        self.print_corr(train, test, files)
        timer.time("corr check")
        self.print_score(train, files)
        timer.time("score check")
        self.do_cv_pred(train, test, files)

    def make_files(self, files):
        train = self.csv_io.read_file(path_const.ORG_TRAIN)
        train = train[["card_id", "target"]]
        test = self.csv_io.read_file(path_const.ORG_TEST)
        test = test[["card_id"]]

        for f in files:
            train, test = self.add_file(f[0], f[1], train, test)
        return train, test

    def add_file(self, file_name, col_name, org_train, org_test):
        train_file_name = "../sub/" + file_name + "_oof.csv"
        test_file_name = "../sub/" + file_name + "_sub.csv"
        another_train = self.csv_io.read_file(train_file_name)
        another_test = self.csv_io.read_file(test_file_name)
        another_train.columns = ["card_id", col_name]
        another_test.columns = ["card_id", col_name]
        ret_train = pd.merge(org_train, another_train, on="card_id", how="inner")
        ret_test = pd.merge(org_test, another_test, on="card_id", how="inner")
        return ret_train, ret_test

    @staticmethod
    def add_team_file(file_name, col_name, org_train, org_test):
        train_file_name = "../sub/" + file_name + "_oof_train"
        test_file_name = "../sub/" + file_name + "_oof_test"
        another_train = pd.read_pickle(train_file_name)
        another_test = pd.read_pickle(test_file_name)
        org_train[col_name] = another_train
        org_test[col_name] = another_test
        return org_train, org_test


    @staticmethod
    def print_corr(train, test, files):
        print("correlation check...")
        corr_col = ["target"] + [f[1] for f in files]
        print(train[corr_col].corr())
        test_corr_col = [f[1] for f in files]
        print(test[test_corr_col].corr())

    @staticmethod
    def print_score(train, files):
        for f in files:
            score = evaluator.rmse(train["target"], train[f[1]])
            print(score)

    @staticmethod
    def do_preds(train, test, files):
        print("------- do preds --------")
        ensemble_col = [f[1] for f in files]
        train_x = train[ensemble_col]
        reg = BayesianRidge().fit(train_x, train["target"])
        print(reg.coef_)
        y_pred = reg.predict(train_x)
        score = evaluator.rmse(train["target"], y_pred)
        print(score)

        test_x = test[ensemble_col]
        y_pred = reg.predict(test_x)
        sub = pd.DataFrame()
        sub["card_id"] = test["card_id"]
        sub["target"] = y_pred
        print(train["target"].describe())
        # print(train["big"].describe())
        print(sub["target"].describe())
        sub.to_csv(path_const.OUTPUT_ENS, index=False)

    @staticmethod
    def do_cv_pred(train, test, files):
        print("------- do preds --------")
        ensemble_col = [f[1] for f in files]
        train_x = train[ensemble_col]
        test_x = test[ensemble_col]
        train_y = train["target"]

        submission = pd.DataFrame()
        submission["card_id"] = test["card_id"]
        submission["target"] = 0

        outliers = (train["target"] < -30).astype(int).values
        split_num = 5
        skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=4590)
        train_preds = []
        for idx, (train_index, test_index) in enumerate(skf.split(train, outliers)):
            X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
            y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

            reg = BayesianRidge().fit(X_train, y_train)
            print(reg.coef_)
            valid_set_pred = reg.predict(X_test)
            score = evaluator.rmse(y_test, valid_set_pred)
            print(score)

            y_pred = reg.predict(test_x)
            submission["target"] = submission["target"] + y_pred
            train_id = train.iloc[test_index]
            train_cv_prediction = pd.DataFrame()
            train_cv_prediction["card_id"] = train_id["card_id"]
            train_cv_prediction["cv_pred"] = valid_set_pred
            train_preds.append(train_cv_prediction)

        train_output = pd.concat(train_preds, axis=0)

        submission["target"] = submission["target"] / split_num
        submission.to_csv(path_const.OUTPUT_SUB, index=False)

        train_output["cv_pred"] = np.clip(train_output["cv_pred"], -33.219281, 18.0)
        train_output.to_csv(path_const.OUTPUT_OOF, index=False)

        df_pred = pd.merge(train[["card_id", "target"]], train_output, on="card_id")
        rmse_score = evaluator.rmse(df_pred["target"], df_pred["cv_pred"])
        print(rmse_score)


obj = GoldenLr()
obj.doit()



