import pandas as pd
import numpy as np
import datetime


class SortedFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)

        ret_df = self._do_day(df)

        # probably noise, but not bad
        # time_df = self._do_time_diff(df)
        # ret_df = pd.merge(ret_df, time_df, on="card_id", how="left")

        # auth_df = self._do_auth(df)
        # ret_df = pd.merge(ret_df, auth_df, on="card_id", how="left")
        # state_df = self._do_state(df)
        # ret_df = pd.merge(ret_df, state_df, on="card_id", how="left")

        # if self.prefix == "old":
        #     recent = self.do_recent_feats(df)
        #     ret_df = pd.merge(ret_df, recent, on="card_id", how="left")
        return ret_df

    @staticmethod
    def do_prep(df):
        df = df.sort_values(by=["card_id", "purchase_date"])
        df["day"] = df['purchase_date'].dt.day
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        return df

    def _do_day(self, df):
        last_df = df.groupby("card_id")["day"].last().reset_index()
        last_df.columns = ["card_id"] + ["_".join([self.prefix, "last_day"])]
        # first_df = df.groupby("card_id")["day"].first().reset_index()
        # first_df.columns = ["card_id"] + ["_".join([self.prefix, "first_day"])]
        #
        # ret_df = pd.merge(first_df, last_df, on="card_id", how="left")
        # return ret_df
        return last_df

    def _do_time_diff(self, df):
        df["prev_time"] = df.groupby("card_id")["purchase_date"].transform(lambda x: x.shift())
        df["time_diff"] = (df["purchase_date"] - df["prev_time"]).dt.total_seconds()

        aggs = {
            "time_diff": ["max", "mean", "min", "std", "skew"]
        }
        ret_df = df.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df

    def _do_state(self, df):
        use_df = df[df["state_id"] > 0]

        last_df = use_df.groupby("card_id")["state_id"].last().reset_index()
        last_col = "_".join([self.prefix, "last_state"])
        last_df.columns = ["card_id", last_col]
        first_df = use_df.groupby("card_id")["state_id"].first().reset_index()
        first_col = "_".join([self.prefix, "first_state"])
        first_df.columns = ["card_id", first_col]

        ret_df = pd.merge(first_df, last_df, on="card_id", how="left")
        flag_col = "_".join([self.prefix, "same_state"])
        ret_df[flag_col] = np.where(ret_df[last_col] == ret_df[first_col], 1, 0)

        return ret_df

    def _do_auth(self, df):
        df["prev_auth"] = df["authorized_flag"].shift()
        df["prev_mer"] = df["merchant_id"].shift()
        df["prev_pur"] = df["purchase_date"].shift()
        df["prev_amount"] = df["purchase_amount"].shift()
        df["time_diff"] = (df["purchase_date"] - df["prev_pur"]).dt.total_seconds()

        mask = (df["time_diff"] <= 600) & (df["prev_mer"] == df["merchant_id"]) &\
               (df["prev_amount"] == df["purchase_amount"]) & (df["prev_auth"] == 1)
        df["seq_non_auth"] = np.where(mask, 1, 0)

        aggs = {
            "seq_non_auth": ["sum", "mean"]
        }
        ret_df = df.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df












