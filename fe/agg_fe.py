import pandas as pd
import numpy as np
import datetime

class AggFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self.do_prep(df)
        grouped = self.do_base_agg(df)
        #pivoted = self.do_time_feats(df)
        #ret_df = pd.merge(grouped, pivoted, on="card_id", how="inner")
        return grouped
        #return ret_df

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        df['dow'] = df['purchase_date'].dt.dayofweek
        df["weekend"] = np.where(df["dow"] >= 5, 1, 0)
        df["trans_elapsed_days"] = (datetime.date(2018, 6, 1) - df['purchase_date'].dt.date).dt.days
        # df['hour'] = df['purchase_date'].dt.hour
        # df["hour"] = np.where(df["hour"] <= 4, df["hour"]+24, df["hour"])
        return df

    def do_base_agg(self, df):
        aggs = {
            "authorized_flag": ["mean"],
            "city_id": ["nunique"],  # maybe the most frequent one?
            "category_1": ["mean"],
            "installments": ["mean", "sum"],
            "category_3": ["mean"],
            "merchant_id": ["nunique"],
            "merchant_category_id": ["nunique"],  # maybe target encode or purchase encode?
            "month_lag": ["mean", "std", "max", "min"],
            "purchase_amount": ["max", "min", "mean", "std", "count", "skew", "sum"],
            # "category_2": ["nunique", "top"],
            "state_id": ["nunique"],
            "subsector_id": ["nunique"],
            "trans_elapsed_days": ["mean", "std", "max", "min"],
            "weekend": ["mean"]
            # "dow": ["mean"],
            # "hour": ["mean"]
        }
        all_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols

        # make ptp col
        month_ptp = "_".join([self.prefix, "month_lag", "ptp"])
        max_col = "_".join([self.prefix, "month_lag", "max"])
        min_col = "_".join([self.prefix, "month_lag", "min"])
        all_agg[month_ptp] = all_agg[max_col] - all_agg[min_col]

        month_ptp = "_".join([self.prefix, "trans_elapsed_days", "ptp"])
        max_col = "_".join([self.prefix, "trans_elapsed_days", "max"])
        min_col = "_".join([self.prefix, "trans_elapsed_days", "min"])
        all_agg[month_ptp] = all_agg[max_col] - all_agg[min_col]
        # all_agg.drop(columns=[max_col, min_col], inplace=True)

        return all_agg

    def do_time_feats(self, df):
        counter = df.groupby("card_id")["purchase_amount"].agg("count").reset_index()
        counter.columns = ["card_id", "cnt"]

        df['dow'] = df['purchase_date'].dt.dayofweek
        dow = pd.crosstab(df["card_id"], df["dow"]).reset_index()
        dow_cols = ["_".join([self.prefix, "dow", str(i), "count"]) for i in range(7)]
        dow.columns = ["card_id"] + dow_cols

        df['hour'] = df['purchase_date'].dt.hour
        bins = [-1, 5, 10, 14, 18, 23, 25]
        labels = [1, 2, 3, 4, 5, 6]
        df['binned_hour'] = pd.cut(df['hour'], bins=bins, labels=labels)
        df["binned_hour"] = np.where(df["binned_hour"] == 6, 1, df["binned_hour"])
        hour = pd.crosstab(df["card_id"], df["binned_hour"]).reset_index()
        hour_cols = ["_".join([self.prefix, "hour", str(i), "count"]) for i in range(5)]
        hour.columns = ["card_id"] + hour_cols

        dow = pd.merge(counter, dow, on="card_id", how="inner")
        for c in dow_cols:
            dow[c] = dow[c] / dow["cnt"]
        dow.drop(columns="cnt", inplace=True)

        # hour = pd.merge(counter, hour, on="card_id", how="inner")
        # for c in hour_cols:
        #     hour[c] = hour[c] / hour["cnt"]
        # hour.drop(columns="cnt", inplace=True)

        return hour

        # ret_df = pd.merge(dow, hour, on="card_id", how="inner")
        # return ret_df













