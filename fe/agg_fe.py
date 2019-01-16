import pandas as pd
from elo.common import pocket_timer


class AggFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        timer = pocket_timer.GoldenTimer()
        timer.time("start prep")
        df = self.do_prep(df)
        timer.time("start agg")
        grouped = self.do_base_agg(df)
        timer.time("done agg")
        return grouped

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        # df['dow'] = df['purchase_date'].dt.dayofweek
        # df['hour'] = df['purchase_date'].dt.hour
        # bins = [0, 5, 10, 14, 18, 23, 24]
        # labels = [1, 2, 3, 4, 5, 1]
        # df['binned_hour'] = pd.cut(df['hour'], bins=bins, labels=labels)
        return df

    def do_base_agg(self, df):
        aggs = {
            "authorized_flag": ["mean"],
            # "city_id": ["nunique"], maybe the most frequent one?
            "category_1": ["mean"],
            "installments": ["mean", "max"],
            "category_3": ["mean"],
            # "merchant_category_id": ["nunique"] maybe target encode or purchase encode?
            "month_lag": ["mean", "std"],
            "purchase_amount": ["max", "min", "mean", "std", "count"], # and skew
            # "category_2": ["nunique", "top"],
            # "state_id": ["nunique", "top"],
            # "subsector_id": ["nunique", "top"],
        }
        all_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols
        return all_agg

