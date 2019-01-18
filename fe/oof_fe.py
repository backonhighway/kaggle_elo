import pandas as pd
import numpy as np
import datetime
from sklearn import model_selection


class OofFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, base, ts):
        _ts = self._do_prep(base, ts)
        ret_df = self._do_split_fe(base, _ts)
        return ret_df

    @staticmethod
    def _do_prep(base, ts):
        for_merge = base[["card_id", "target"]]
        ts = pd.merge(ts, for_merge, on="card_id", how="left")
        return ts

    def _do_split_fe(self, base, ts):
        skf = model_selection.StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
        outliers = (base["target"] < -30).astype(int).values
        encoded_ts_list = []
        for train_index, test_index in skf.split(base, outliers):
            train_id = base.iloc[train_index]["card_id"]
            test_id = base.iloc[test_index]["card_id"]
            train_ts = ts[ts["card_id"].isin(train_id)]
            test_ts = ts[ts["card_id"].isin(test_id)]
            encoded = self._do_encode(train_ts, test_ts)
            encoded_ts_list.append(encoded)
        encoded_ts = pd.concat(encoded_ts_list, axis=0)
        print(encoded_ts.shape)
        print(encoded_ts.head())
        ret_df = self._do_agg(encoded_ts)
        return ret_df

    @staticmethod
    def _do_encode(from_df, to_df):
        target_df = from_df.groupby("merchant_id")["target"].mean().reset_index()
        target_df.columns = ["merchant_id", "target_encode"]
        use_col = ["card_id", "merchant_id"]
        ret_df = pd.merge(to_df[use_col], target_df, on="merchant_id", how="left")
        return ret_df[["card_id", "target_encode"]]

    def _do_agg(self, encoded_ts):
        aggs = {
            "target_encode": ["mean", "max", "min"]
        }
        all_agg = encoded_ts.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols
        return all_agg




