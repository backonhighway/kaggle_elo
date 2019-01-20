import pandas as pd
import numpy as np
import datetime


class AggFe2:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        ret_df = self.do_base_agg(df)
        return ret_df

    @staticmethod
    def do_prep(df):
        df['month'] = df['purchase_date'].dt.month
        df['woy'] = df['purchase_date'].dt.weekofyear
        df["day"] = df['purchase_date'].dt.day
        return df

    def do_base_agg(self, df):
        first_agg = {
            "month": ["nunique"],
            "woy": ["nunique"],
            "day": ["count"]
        }
        fa = df.groupby(["card_id", "merchant_id"]).agg(first_agg).reset_index()
        fa.columns = ["card_id", "merchant_id"] + ["mon_nunique", "woy_nunique", "cnt"]

        second_agg = {
            "mon_nunique": ["mean"],
            "woy_nunique": ["mean"],
            "cnt": ["mean"]
        }
        ret_df = fa.groupby("card_id").agg(second_agg).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in second_agg.keys() for agg in second_agg[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df

    def do_periodic_flag(self, df):


        df["unique_days"] = df.groupby(["card_id", "merchant_id", "days"]).transform("nunique")
























