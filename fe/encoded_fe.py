import pandas as pd
import numpy as np
import datetime


class EncodeFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        ret_df = self.do_base_agg(df, self.prefix)
        return ret_df

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        df["merchant_id"].fillna("none", inplace=True)
        df["category_2"].fillna(-1, inplace=True)

        df["mer_id_pur"] = df.groupby("merchant_id")["purchase_amount"].transform("mean")
        df["city_pur"] = df.groupby("city")["purchase_amount"].transform("mean")
        df["mer_cat_pur"] = df.groupby("merchant_category_id")["purchase_amount"].transform("mean")
        df["sub_sec_pur"] = df.groupby("subsector_id")["purchase_amount"].transform("mean")
        return df

    @staticmethod
    def do_base_agg(df, prefix):
        aggs = {
            "mer_id_pur": ["mean", "sum"],
            "city_pur": ["mean", "sum"],
            "mer_cat_pur": ["mean", "sum"],
            "sub_sec_pur": ["mean", "sum"]
        }
        all_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols
        return all_agg






