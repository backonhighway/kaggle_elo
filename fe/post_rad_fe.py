import pandas as pd
import numpy as np
import datetime


class PostRadFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        ret_df = self.do_base_agg(df)
        # mode_feats = self.do_mode_feats(df)
        # ret_df = pd.merge(ret_df, mode_feats, on="card_id", how="left")
        nan_feats = self._do_nan_feats(df)
        ret_df = pd.merge(ret_df, nan_feats, on="card_id", how="left")
        time_feats = self.do_time_feats(df)
        ret_df = pd.merge(ret_df, time_feats, on="card_id", how="left")
        # mon_df = self._do_monthly_count(df)
        # ret_df = pd.merge(ret_df, mon_df, on="card_id", how="left")
        if self.prefix == "old":
            cond_df = self.do_conditional(df)
            ret_df = pd.merge(ret_df, cond_df, on="card_id", how="left")
            recent = self._do_recent_feats(df)  # with/without rec1 makes difference between folds
            ret_df = pd.merge(ret_df, recent, on="card_id", how="left")
            repurchase = self.do_repurchase_rate(df)
            ret_df = pd.merge(ret_df, repurchase, on="card_id", how="left")
            mer_df = self._do_mer_cnt(df)
            ret_df = pd.merge(ret_df, mer_df, on="card_id", how="left")
        # if self.prefix == "new":
        #     recent = self._do_rec_feat(df, 2, "rec_new")
        #     ret_df = pd.merge(ret_df, recent, on="card_id", how="left")
        return ret_df

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        df["installments"] = np.where(df["installments"] == 999, -1, df["installments"])
        df["base_inst"] = np.where(df["installments"] <= 1, 1, df["installments"])
        df["pa"] = df["purchase_amount"]
        df["one_pay"] = df["pa"] / df["base_inst"]
        df["auth_pa"] = np.where(df["authorized_flag"] == 0, df["pa"], 0)
        df["auth_one_par"] = df["auth_pa"] / df["base_inst"]
        return df

    def do_base_agg(self, df):
        aggs = {
            "one_pay": ["max"],
            "auth_one_par": ["max"],
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

    def _do_recent_feats(self, df):
        rec3_df = self._do_rec_feat(df, -2, "rec3")
        rec1_df = self._do_rec_feat(df, 0, "rec1")

        ret_df = pd.merge(rec1_df, rec3_df, on="card_id", how="left")
        return ret_df

    @staticmethod
    def _do_rec_feat(df, lag, name):
        recent_df = df[df["month_lag"] >= lag]
        aggs = {
            "installments": ["sum", ],
            "purchase_amount": ["count"],  # max is not bad
        }
        rec_df = recent_df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([name, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        rec_df.columns = ["card_id"] + cols
        return rec_df

    def _do_monthly_count(self, df):
        aggs = {}
        if self.prefix == "old":
            range_obj = range(-13, 1)
        else:
            range_obj = range(1, 3)
        for i in range_obj:
            col_name = "month" + str(i) + "_count"
            df[col_name] = np.where(df["month_lag"] == i, 1, 0)
            add_agg = {col_name: ["sum", "mean"]}
            aggs.update(add_agg)

        ret_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_agg.columns = ["card_id"] + cols
        return ret_agg







