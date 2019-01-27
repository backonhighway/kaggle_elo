import pandas as pd
import numpy as np
import datetime


class ReAggFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        ret_df = self.do_base_agg(df, self.prefix)
        # mode_feats = self.do_mode_feats(df)
        # ret_df = pd.merge(ret_df, mode_feats, on="card_id", how="left")
        nan_feats = self._do_nan_feats(df)
        ret_df = pd.merge(ret_df, nan_feats, on="card_id", how="left")
        time_feats = self.do_time_feats(df)
        ret_df = pd.merge(ret_df, time_feats, on="card_id", how="left")
        if self.prefix == "old":
            cond_df = self.do_conditional(df)
            ret_df = pd.merge(ret_df, cond_df, on="card_id", how="left")
            recent = self._do_recent_feats(df)  # with/without rec1 makes difference between folds
            ret_df = pd.merge(ret_df, recent, on="card_id", how="left")
            repurchase = self.do_repurchase_rate(df)
            ret_df = pd.merge(ret_df, repurchase, on="card_id", how="left")
        # if self.prefix == "new":
        #     recent = self._do_rec_feat(df, 2, "rec_new")
        #     ret_df = pd.merge(ret_df, recent, on="card_id", how="left")
        return ret_df

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        df['dow'] = df['purchase_date'].dt.dayofweek
        df["weekend"] = np.where(df["dow"] >= 5, 1, 0)
        df["trans_elapsed_days"] = (datetime.date(2018, 6, 1) - df['purchase_date'].dt.date).dt.days
        df['hour'] = df['purchase_date'].dt.hour
        df["hour"] = np.where(df["hour"] <= 5, df["hour"]+24, df["hour"])
        df['woy'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df["day"] = df['purchase_date'].dt.day
        df["installments"] = np.where(df["installments"] == 999, -1, df["installments"])
        # df["inst_pur"] = df["installments"] + 1.0
        # df["inst_pur"] = (df["purchase_amount"]+1) * np.log1p(df["inst_pur"])
        # df["inst_pur2"] = (df["purchase_amount"]+1) * df["category_3"]
        df["no_city"] = np.where(df["city_id"] == -1, 1, 0)
        df["pa2"] = np.where(df["purchase_amount"] <= 0.8, 0.8, df["purchase_amount"])
        df['month_diff'] = (datetime.datetime.today() - df['purchase_date']).dt.days // 30
        df['month_diff'] += df['month_lag']
        df["pa2_month_diff"] = df["pa2"] * df["month_diff"]

        # low_days = ["2017-04-08", "2017-05-12", "2017-05-24", "2017-06-19", "2017-06-30"]
        # df["str_date"] = df["purchase_date"].dt.date.astype(str)
        # df["low_day_flag"] = np.where(df["str_date"].isin(low_days), 1, 0)
        return df

    @staticmethod
    def do_base_agg(df, prefix):
        aggs = {
            "city_id": ["nunique"],  # maybe the most frequent one?
            "category_1": ["mean"],
            "installments": ["mean", "sum"],
            "category_3": ["mean"],
            "merchant_id": ["nunique"],
            "merchant_category_id": ["nunique"],  # maybe target encode or purchase encode?
            "month_lag": ["mean", "std", "max", "min", "skew", "nunique"],
            "purchase_amount": ["max", "min", "mean", "std", "count", "sum"],  # "skew"],
            "category_2": ["nunique"],
            "state_id": ["nunique"],
            "subsector_id": ["nunique"],
            "trans_elapsed_days": ["mean", "std", "max", "min", "skew", "nunique"],
            "no_city": ["mean", "count"],
            # "inst_pur": ["mean"],
            # "inst_pur2": ["mean"],
            "pa2": ["mean", "sum"],  # min, std, max
        }
        old_aggs = {
            "authorized_flag": ["mean", "sum"],
            "month": ["nunique"],
            "woy": ["nunique"],
            "hour": ["nunique"],
            "day": ["nunique"],
            "dow": ["nunique"],
            # "low_day_flag": ["sum", "mean"]
            "month_diff": ["mean", "std"],
            "pa2_month_diff": ["mean", "min"]
        }
        if prefix == "old":
            aggs.update(old_aggs)

        all_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols

        # make ptp col
        month_ptp = "_".join([prefix, "month_lag", "ptp"])
        max_col = "_".join([prefix, "month_lag", "max"])
        min_col = "_".join([prefix, "month_lag", "min"])
        all_agg[month_ptp] = all_agg[max_col] - all_agg[min_col]

        month_ptp = "_".join([prefix, "trans_elapsed_days", "ptp"])
        max_col = "_".join([prefix, "trans_elapsed_days", "max"])
        min_col = "_".join([prefix, "trans_elapsed_days", "min"])
        all_agg[month_ptp] = all_agg[max_col] - all_agg[min_col]

        return all_agg

    def do_mode_feats(self, df):
        aggs = [
            "city_id", "merchant_category_id", "subsector_id"
        ]
        ret_df = None

        def most_freq(series):
            mode_series = series.mode()
            try:
                ret_value = mode_series.iat[0]
            except IndexError:
                ret_value = np.NaN
            return ret_value

        for agg in aggs:
            temp_df = df.groupby("card_id")[agg].apply(lambda x: most_freq(x)).reset_index()
            if ret_df is None:
                ret_df = temp_df
            else:
                ret_df = pd.merge(ret_df, temp_df, on="card_id", how="left")

        ret_df.columns = ["card_id"] + ["_".join([self.prefix, agg]) for agg in aggs]
        return ret_df

    def _do_nan_feats(self, df):
        ret_df = df.groupby("card_id")[["merchant_id", "category_2", "category_3"]]\
            .apply(lambda s: s.isna().sum()).reset_index()
        cols = ["null_merchant", "null_state", "null_install"]
        cols = ["_".join([self.prefix, c]) for c in cols]
        ret_df.columns = ["card_id"] + cols
        return ret_df

    def do_conditional(self, df):
        aggs = {
            "purchase_amount": ["max", "min", "mean"]
        }
        no_city_agg = {
            "purchase_amount": ["max", "min", "mean"],
            "installments": ["mean", "sum"],
        }
        no_city = df[df["city_id"] == -1].groupby("card_id").agg(no_city_agg).reset_index()
        no_city.columns = ["card_id"] + ["_".join([self.prefix, "no_city", k, agg]) for k in no_city_agg.keys() for agg in no_city_agg[k]]
        no_install = df[df["installments"] == -1].groupby("card_id").agg(aggs).reset_index()
        no_install.columns = ["card_id"] + ["_".join([self.prefix, "no_install", k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df = pd.merge(no_city, no_install, on="card_id", how="outer")

        if self.prefix == "old":
            not_auth = df[df["authorized_flag"] == 1].groupby("card_id").agg(aggs).reset_index()
            not_auth.columns = ["card_id"] + ["_".join([self.prefix, "not_auth", k, agg]) for k in aggs.keys() for agg in
                                              aggs[k]]
            auth = df[df["authorized_flag"] == 0].groupby("card_id").agg(aggs).reset_index()
            auth.columns = ["card_id"] + ["_".join([self.prefix, "auth", k, agg]) for k in aggs.keys() for agg in aggs[k]]
            ret_df = pd.merge(ret_df, not_auth, on="card_id", how="outer")
            ret_df = pd.merge(ret_df, auth, on="card_id", how="outer")
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

    def do_repurchase_rate(self, df):

        def get_rate(series):
            freq_counts = series.value_counts()
            repeated = (freq_counts > 1).sum()
            chances = freq_counts.count()
            return repeated / chances

        ret_df = df.groupby("card_id")["merchant_id"].apply(lambda x: get_rate(x)).reset_index()
        ret_df.columns = ["card_id", self.prefix + "_repurchase"]
        return ret_df

    def do_time_feats(self, df):
        counter = df.groupby("card_id")["purchase_amount"].agg("count").reset_index()
        counter.columns = ["card_id", "cnt"]

        df['dow'] = df['purchase_date'].dt.dayofweek
        dow = pd.crosstab(df["card_id"], df["dow"]).reset_index()
        dow_cols = ["_".join([self.prefix, "dow", str(i), "count"]) for i in range(7)]
        dow.columns = ["card_id"] + dow_cols

        df['hour'] = df['purchase_date'].dt.hour
        df["hour"] = np.where(df["hour"] <= 5, df["hour"]+24, df["hour"])
        bins = [-1, 12, 17, 21, 32]
        labels = [1, 2, 3, 4]
        df['binned_hour'] = (pd.cut(df['hour'], bins=bins, labels=labels)).astype(int)
        hour = pd.crosstab(df["card_id"], df["binned_hour"]).reset_index()
        hour_cols = ["_".join([self.prefix, "hour", str(i), "count"]) for i in range(4)]
        hour.columns = ["card_id"] + hour_cols

        dow = pd.merge(counter, dow, on="card_id", how="inner")
        for c in dow_cols:
            dow[c] = dow[c] / dow["cnt"]
        dow.drop(columns="cnt", inplace=True)

        hour = pd.merge(counter, hour, on="card_id", how="inner")
        for c in hour_cols:
            hour[c] = hour[c] / hour["cnt"]
        hour.drop(columns="cnt", inplace=True)

        return hour
        # ret_df = pd.merge(dow, hour, on="card_id", how="inner")
        # return ret_df














