import lightgbm as lgb
import pandas as pd
import numpy as np
from . import pocket_logger
from elo.common import pred_cols


class GoldenLgb:
    def __init__(self, seed=99, cat_col=pred_cols.CAT_COLS):
        self.train_param = self.kernel_train_param()
        self.category_col = cat_col
        self.drop_cols = [
        ]

    def do_train_direct(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=100,
                          num_boost_round=3000,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    def train_no_holdout(self, x_train, y_train):
        lgb_train = lgb.Dataset(x_train, y_train)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=None,
                          num_boost_round=300,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    @staticmethod
    def show_feature_importance(model, filename=None):
        fi = pd.DataFrame({
            "name": model.feature_name(),
            "importance_split": model.feature_importance(importance_type="split").astype(int),
            "importance_gain": model.feature_importance(importance_type="gain").astype(int),
        })
        fi = fi.sort_values(by="importance_gain", ascending=False)

        pd.set_option('display.max_columns', None)
        logger = pocket_logger.GoldenLogger()
        logger.print(fi)
        if filename is not None:
            empty = pd.DataFrame()
            empty.to_csv(filename, index=False, mode="a")
            fi.to_csv(filename, index=False, mode="a")

    @staticmethod
    def kernel_train_param():
        return {
            'num_leaves': 31,
            'min_data_in_leaf': 30,
            'objective': 'regression',
            'max_depth': -1,
            'learning_rate': 0.01,
            "boosting": "gbdt",
            "feature_fraction": 0.9,
            "bagging_freq": 1,
            "bagging_fraction": 0.9,
            "bagging_seed": 11,
            "metric": 'rmse',
            "lambda_l1": 0.1,
            "verbosity": -1,
            "random_state": 4590,
        }

    @staticmethod
    def optuna_train_param():
        return {
            'num_leaves': 63,
            'min_data_in_leaf': 89,
            'objective': 'regression',
            'max_depth': -1,
            'learning_rate': 0.01,
            "boosting": "gbdt",
            "feature_fraction": 0.55,
            # "bagging_freq": 1,
            # "bagging_fraction": 0.9,
            # "bagging_seed": 11,
            "metric": 'rmse',
            "lambda_l2": 0.1,
            "verbosity": -1,
            "random_state": 4590,
        }

    @staticmethod
    def fast_train_param(seed):
        return {
            'learning_rate': 0.02,
            'num_leaves': 31,
            'boosting': 'gbdt',
            'application': 'regression',
            'metric': 'rmse',
            'feature_fraction': .7,
            #"max_bin": 511,
            'seed': seed,
            'verbose': 0,
        }


class AdversarialLgb(GoldenLgb):
    def __init__(self, seed=99, cat_col=pred_cols.CAT_COLS):
        super().__init__()
        self.train_param = {
            'learning_rate': 0.02,
            'num_leaves': 31,
            'boosting': 'gbdt',
            'application': 'binary',
            'metric': 'auc',
            'feature_fraction': .7,
            #"max_bin": 511,
            'seed': seed,
            'verbose': 0,
        }
        self.target_col_name = "target"
        if cat_col is not None:
            self.category_col = cat_col

    def do_train_direct(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=100,
                          num_boost_round=3000,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model


class ShallowLgb(GoldenLgb):
    def __init__(self, seed=99, cat_col=pred_cols.CAT_COLS):
        super().__init__()
        self.train_param = {
            'learning_rate': 0.02,
            'num_leaves': 5,
            "max_depth": -1,
            'boosting': 'gbdt',
            'application': 'regression',
            'metric': 'rmse',
            'feature_fraction': .9,
            "max_bin": 10,
            'seed': seed,
            'verbose': 0,
        }
        self.target_col_name = "target"
        self.category_col = cat_col
        self.drop_cols = [
        ]

    def do_train_direct(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=100,
                          num_boost_round=300,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model


class TsLgb(GoldenLgb):
    def __init__(self, seed=99):
        super().__init__()
        self.train_param = {
            'num_leaves': 31,
            'min_data_in_leaf': 30,
            'objective': 'regression',
            'max_depth': -1,
            'learning_rate': 0.05,
            "boosting": "gbdt",
            "feature_fraction": 0.9,
            "metric": 'rmse',
            "verbosity": -1,
            "random_state": seed
        }
        self.target_col_name = "target"
        self.category_col = [
            # "state_id", "city_id"  # "category_3", "category_1", "category_2"
        ]
        self.drop_cols = [
        ]

    def do_train_direct(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=100,
                          num_boost_round=1000,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        print('End training...')
        return model


class OptLgb(GoldenLgb):
    def __init__(self, train_param):
        super().__init__()
        self.train_param = train_param

    def do_train_direct(self, x_train, x_test, y_train, y_test):
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test)

        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=1000,
                          num_boost_round=3000,
                          early_stopping_rounds=100,
                          categorical_feature=self.category_col)
        return model
