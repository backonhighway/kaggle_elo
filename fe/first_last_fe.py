import pandas as pd
import numpy as np
import datetime


class FirstLastFe:
    def __init__(self, prefix):
        self.prefix = prefix
        self.use_col = [
            "card_id", "authorized_flag", "city_id", "category_1", "installments",	"category_3",
            "merchant_category_id", "month_lag", "purchase_amount",
            "category_2", "state_id", "subsector_id",
            "dow", "trans_elapsed_days", "hour", "second", "minute", "woy", "month"
        ]

    def do_fe(self, df):
        df = self.do_prep(df)
        last_df = self._do_last(df)
        first_df = self._do_first(df)
        ret_df = pd.merge(last_df, first_df, on="card_id", how="left")
        return ret_df

    @staticmethod
    def do_prep(df):
        df = df.sort_values(by=["card_id", "purchase_date"])
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        df['dow'] = df['purchase_date'].dt.dayofweek
        df["trans_elapsed_days"] = (datetime.date(2018, 6, 1) - df['purchase_date'].dt.date).dt.days
        df['hour'] = df['purchase_date'].dt.hour
        df['second'] = df['purchase_date'].dt.second
        df['minute'] = df['purchase_date'].dt.minute
        df['woy'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        return df

    def _do_last(self, df):
        last_df = df[self.use_col].groupby("card_id").last().reset_index()
        last_df.columns = ["card_id"] + ["_".join([self.prefix, "last", c]) for c in last_df.columns[1:]]
        return last_df

    def _do_first(self, df):
        first_df = df[self.use_col].groupby("card_id").first().reset_index()
        first_df.columns = ["card_id"] + ["_".join([self.prefix, "first", c]) for c in first_df.columns[1:]]
        return first_df












