import pandas as pd
import numpy as np
import datetime


class OldFirstMerFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, old):
        old = self._do_prep(old)
        ret_df = self._do_agg(old)
        return ret_df

    @staticmethod
    def _do_prep(old):
        ret_df = old.sort_values(by=["card_id", "purchase_date"])
        ret_df = ret_df.groupby(["card_id", "merchant_id"]).first().reset_index()
        # ret_df["one"] = 1
        # ret_df["order"] = ret_df.groupby(["card_id", "merchant_id"])["one"].cumsum()
        # ret_df = ret_df[ret_df["order"] == 1]

        ret_df['authorized_flag'] = ret_df['authorized_flag'].map({'Y': 0, 'N': 1})
        ret_df['category_1'] = ret_df['category_1'].map({'Y': 1, 'N': 0})
        ret_df['category_3'] = ret_df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        ret_df["trans_elapsed_days"] = (datetime.date(2018, 6, 1) - ret_df['purchase_date'].dt.date).dt.days
        ret_df["installments"] = np.where(ret_df["installments"] == 999, -1, ret_df["installments"])

        ret_df['woy'] = ret_df['purchase_date'].dt.weekofyear
        ret_df['month'] = ret_df['purchase_date'].dt.month
        return ret_df

    def _do_agg(self, new):
        aggs = {
            "city_id": ["nunique"],
            "category_1": ["mean"],
            "installments": ["mean", "sum"],
            "category_3": ["mean"],
            "month_lag": ["mean", ],
            "purchase_amount": ["max", "min", "mean"],
            "state_id": ["nunique"],
            "trans_elapsed_days": ["mean", "max"],
            "authorized_flag": ["mean", "sum"],
            "month": ["nunique"],
            "woy": ["nunique"],
        }
        ret_df = new.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df

# do recent counts









