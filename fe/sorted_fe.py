import pandas as pd
import numpy as np
import datetime


class SortedFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        ret_df = self.do_base_agg(df)
        # if self.prefix == "old":
        #     recent = self.do_recent_feats(df)
        #     ret_df = pd.merge(ret_df, recent, on="card_id", how="left")
        #     cond_df = self.do_conditional(df)
        #     ret_df = pd.merge(ret_df, cond_df, on="card_id", how="left")
        #     repurchase = self.do_repurchase_rate(df)
        #     ret_df = pd.merge(ret_df, repurchase, on="card_id", how="left")
        return ret_df

    @staticmethod
    def do_prep(df):
        df = df.sort_values(by=["card_id", "purchase_date"])
        df["day"] = df['purchase_date'].dt.day
        return df

    def do_base_agg(self, df):
        ret_df = df.groupby("card_id")["day"].last().reset_index()
        ret_df.columns = ["card_id"] + ["_".join([self.prefix, "last_day"])]
        return ret_df












