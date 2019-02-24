import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.common import pocket_lgb, evaluator, pocket_scaler
from elo.trainer import pocket_cv2
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
            ("rnd_feat_bridge", "rnd_feat"),
            ("small_team", "small")
        ]

        train, test = self.make_files(files)
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

        # do scaling
        key_cols = ["card_id", "target"]
        # not_scale_col = cat_cols + key_cols
        scale_col = [c for c in train.columns if c not in key_cols]

        train_x = pocket_scaler.rank_gauss(train_x, scale_col).fillna(0)
        test_x = pocket_scaler.rank_gauss(test_x, scale_col).fillna(0)
        data = (train, test, train_x, train_y, test_x)

        trainer = pocket_cv2.GoldenTrainer(epochs=20, batch_size=512)
        trainer.do_cv(data)


obj = GoldenLr()
obj.doit()



