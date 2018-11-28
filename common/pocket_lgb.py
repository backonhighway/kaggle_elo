import lightgbm as lgb
import pandas as pd
import numpy as np
from . import pocket_logger
from plastic.common import pred_cols, pocket_eval


class GoldenLgb:
    def __init__(self, seed=99, cat_col=pred_cols.CAT_COLS):
        self.train_param = {
            'learning_rate': 0.05,
            'num_leaves': 31,
            'boosting': 'gbdt',
            'application': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': 14,  # without class99
            'feature_fraction': .7,
            #"max_bin": 511,
            'seed': seed,
            'verbose': 0,
        }
        self.target_col_name = "target"
        # self.category_col = [] TODO: is this better than None?
        if cat_col is not None:
            self.category_col = cat_col
        self.drop_cols = [
        ]


    def do_train(self, train_data, test_data, predict_col):
        tcn = self.target_col_name
        y_train = train_data[tcn]
        y_test = test_data[tcn]
        x_train = train_data[predict_col].drop(self.drop_cols, axis=1)
        x_test = test_data[predict_col].drop(self.drop_cols, axis=1)

        return self.do_train_direct(x_train, x_test, y_train, y_test)

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
                          feval=pocket_eval.GoldenEval.lgb_multi_weighted_logloss,
                          categorical_feature=self.category_col)
        print('End training...')
        return model

    def train_with_weight(self, x_train, x_test, y_train, y_test, w_train, w_test):
        lgb_train = lgb.Dataset(x_train, y_train, weight=w_train)
        lgb_eval = lgb.Dataset(x_test, y_test, weight=w_test)

        print('Start training...')
        model = lgb.train(self.train_param,
                          lgb_train,
                          valid_sets=[lgb_eval],
                          verbose_eval=50,
                          num_boost_round=300,
                          early_stopping_rounds=100,
                          feval=pocket_eval.GoldenEval.lgb_multi_weighted_logloss,
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
        fi = fi.sort_values(by="importance_split", ascending=False)

        pd.set_option('display.max_columns', None)
        print(fi)
        logger = pocket_logger.GoldenLogger()
        logger.info(fi)
        if filename is not None:
            fi.to_csv(filename, index=False, mode="a")


class AdversarialLgb:
    def __init__(self, seed=99, cat_col=pred_cols.CAT_COLS):
        self.train_param = {
            'learning_rate': 0.05,
            'num_leaves': 31,
            'boosting': 'gbdt',
            'application': 'binary',
            'metric': 'AUC',
            'feature_fraction': .7,
            #"max_bin": 511,
            'seed': seed,
            'verbose': 0,
        }
        self.target_col_name = "target"
        if cat_col is not None:
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