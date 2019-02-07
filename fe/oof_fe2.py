import pandas as pd
import numpy as np
from sklearn import model_selection


class OofFe2:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, train, test, ts):
        _ts = self._do_prep(train, ts)
        ret_df = self._do_oof_fe(train, test, _ts)
        return ret_df

    @staticmethod
    def _do_prep(train, ts):
        for_merge = train[["card_id", "target"]]
        ts = pd.merge(ts, for_merge, on="card_id", how="left")
        ts["merchant_id"] = ts["merchant_id"].fillna(-1)
        ts["subsector_id"] = ts["subsector_id"].fillna(-1)
        ts['hour'] = ts['purchase_date'].dt.hour
        ts['minute'] = ts['purchase_date'].dt.minute
        ts['second'] = ts['purchase_date'].dt.second
        mask = (ts["hour"] == 0) & (ts["minute"] == 0) & (ts["second"] == 0)
        ts["hour"] = np.where(mask, 24, ts["hour"])
        ts["ym"] = ts['purchase_date'].dt.year * 100 + ts['purchase_date'].dt.month
        return ts

    def _do_oof_fe(self, train, test, ts):
        skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
        outliers = (train["target"] < -30).astype(int).values
        encoded_ts_list = []
        for train_index, oof_index in skf.split(train, outliers):
            train_id = train.iloc[train_index]["card_id"]
            oof_id = train.iloc[oof_index]["card_id"]
            train_ts = ts[ts["card_id"].isin(train_id)]
            oof_ts = ts[ts["card_id"].isin(oof_id)]
            encoded = self._do_encode(train_ts, oof_ts)
            encoded_ts_list.append(encoded)
        encoded_train_ts = pd.concat(encoded_ts_list, axis=0)
        print(encoded_train_ts.shape)

        test_id = test["card_id"]
        test_ts = ts[ts["card_id"].isin(test_id)]
        encoded_test_ts = self._do_encode(ts, test_ts)

        ret_train = self._do_agg(encoded_train_ts)
        ret_test = self._do_agg(encoded_test_ts)
        print(ret_train.shape)
        print(ret_test.shape)
        ret_df = pd.concat([ret_train, ret_test], axis=0)
        print(ret_df.shape)
        return ret_df

    @staticmethod
    def _do_encode(from_ts, to_ts):
        ret_df = None
        cat_col = ["merchant_id", "subsector_id"]  # hour, ym good for diversity probably
        # cat_col = ["merchant_id", "city_id", "merchant_category_id", "subsector_id"]
        for col in cat_col:
            ts_source = from_ts.copy()
            count_col = col + "_cnt"
            ts_source[count_col] = ts_source.groupby(col)["target"].transform("count").fillna(0)
            ts_source[count_col] = np.where(ts_source[count_col] >= 7, 1, np.log1p(ts_source[count_col])/2)
            ts_source["weighted_target"] = ts_source[count_col] * ts_source["target"]
            target_df = ts_source.groupby(col)["weighted_target"].mean().reset_index()
            target_df.columns = [col, col + "_target_encode"]
            if ret_df is None:
                for_merge_col = ["card_id"] + cat_col
                ret_df = pd.merge(to_ts[for_merge_col], target_df, on=col, how="left")
            else:
                ret_df = pd.merge(ret_df, target_df, on=col, how="left")
        return ret_df

    def _do_agg(self, encoded_ts):
        aggs = {
            "merchant_id_target_encode": ["mean"],
            # "hour_target_encode": ["mean"],
            # "ym_target_encode": ["mean"],
            # "merchant_category_id_target_encode": ["mean", "min"],
            "subsector_id_target_encode": ["mean"],
        }
        all_agg = encoded_ts.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols
        return all_agg



