import pandas as pd
import numpy as np


class PostRadFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        ret_df = self.do_base_agg(df)
        if self.prefix == "old":
            rec1_df = self._do_rec_feat(df, 0, "old_rec1")
            ret_df = pd.merge(ret_df, rec1_df, on="card_id", how="left")
            rec2_df = self._do_rec_feat(df, -1, "old_rec2")
            ret_df = pd.merge(ret_df, rec2_df, on="card_id", how="left")
        if self.prefix == "new":
            rec1_df = self._do_rec_feat(df, 1, "new_rec1")
            ret_df = pd.merge(ret_df, rec1_df, on="card_id", how="left")
            rec2_df = self._do_rec_feat(df, 2, "new_rec2")
            ret_df = pd.merge(ret_df, rec2_df, on="card_id", how="left")
        return ret_df

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        df["installments"] = np.where(df["installments"] == 999, -1, df["installments"])
        df["base_inst"] = np.where(df["installments"] <= 1, 1, df["installments"])
        df["pa"] = np.round(df['purchase_amount'] / 0.00150265118 + 497.06, 2)
        df["one_pay"] = df["pa"] / df["base_inst"]
        df["auth_pa"] = np.where(df["authorized_flag"] == 0, df["pa"], 0)
        df["auth_one_pay"] = df["auth_pa"] / df["base_inst"]
        return df

    def do_base_agg(self, df):
        aggs = {
            "one_pay": ["max"],
            "auth_one_pay": ["max"],
        }
        all_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols
        return all_agg

    def _do_monthly(self, df):
        aggs = {
            "pa": ["mean", "sum"],
            "auth_pa": ["mean", "sum"],
        }
        monthly = df.groupby(["card_id", "month_lag"]).agg(aggs).reset_index()
        cols = ["_".join([k, agg]) for k in aggs.keys() for agg in aggs[k]]
        monthly.columns = ["card_id", "month_lag"] + cols

        aggs2 = {
            "pa_mean": ["mean"],
            "pa_sum": ["mean"],
            "auth_pa_mean": ["mean"],
            "auth_pa_sum": ["mean"],
        }
        ret_df = monthly.groupby("card_id").agg(aggs2).reset_index()
        cols = ["_".join([self.prefix, "monthly", k]) for k in aggs2.keys() for aggs2 in aggs2[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df

    @staticmethod
    def _do_rec_feat(df, lag, name):
        recent_df = df[df["month_lag"] == lag]
        aggs = {
            "pa": ["sum", "max"],
            "auth_pa": ["sum", "max"],
        }
        rec_df = recent_df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([name, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        rec_df.columns = ["card_id"] + cols
        return rec_df






