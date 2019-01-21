import pandas as pd

class AggFe2:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        df = self._do_prep(df)
        ret_df = self._do_base_agg(df)
        # time_feats = self._do_time_feats(df)
        # ret_df = pd.merge(ret_df, time_feats, on="card_id", how="left")
        return ret_df

    @staticmethod
    def _do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 1, 'N': 0})
        df['month'] = df['purchase_date'].dt.month
        df['woy'] = df['purchase_date'].dt.weekofyear
        df["day"] = df['purchase_date'].dt.day

        df["hour"] = df['purchase_date'].dt.hour
        df["minute"] = df['purchase_date'].dt.minute
        df["second"] = df['purchase_date'].dt.second
        df["zero_time"] = ((df["hour"] == 0) & (df["minute"] == 0) & (df["second"] == 0)).astype(int)
        return df

    def _do_base_agg(self, df):
        first_agg = {
            "month": ["nunique"],
            "woy": ["nunique"],
            "day": ["count", "nunique"],
            "authorized_flag": ["mean"],
        }
        fa = df.groupby(["card_id", "merchant_id"]).agg(first_agg).reset_index()
        fa.columns = ["card_id", "merchant_id"] + ["mon_nunique", "woy_nunique", "day_count", "day_nunique", "auth_mean"]
        fa["rush_buy_flag"] = ((fa["day_count"] > 2) & (fa["day_nunique"] == 1) & (fa["mon_nunique"] == 1)).astype(int)
        fa["periodic_flag"] = ((fa["day_count"] > 2) & (fa["day_nunique"] == 1)
                               & (fa["mon_nunique"] > 1) & (fa["auth_mean"] >= 0.9)).astype(int)

        second_agg = {
            "mon_nunique": ["mean"],
            "woy_nunique": ["mean"],
            "day_count": ["mean"],
            "rush_buy_flag": ["mean", "sum"],
            "periodic_flag": ["sum"],
        }
        ret_df = fa.groupby("card_id").agg(second_agg).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in second_agg.keys() for agg in second_agg[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df

    def _do_time_feats(self, df):
        aggs = {
            "zero_time": ["sum", "mean"]
        }
        ret_df = df.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df




















