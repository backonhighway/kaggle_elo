import pandas as pd
import numpy as np
from concurrent import futures
from elo.common import pocket_scaler


class GoldenReshaper:

    def __init__(self, split_num=16):
        self._SPLIT_NUM = split_num
        self.cat_col = [
            "merchant_category_id", "state_id", "subsector_id", "category_2"
        ]
        self.num_col = [
            "authorized_flag", "no_city", "category_1", "installments",
            "month_lag", "purchase_amount", "day"
        ]

    def do_para_reshape(self, ts):
        ts_num, ts_cat, ts_key = self._do_para_reshape(ts)
        return ts_num, ts_cat, ts_key

    def _do_para_reshape(self, trans):
        split_trans = self._split_series(trans)
        future_list = list()
        with futures.ProcessPoolExecutor(max_workers=self._SPLIT_NUM) as executor:
            for s in split_trans:
                future_list.append(executor.submit(self._do_reshape, s))
        future_results = [f.result() for f in future_list]
        num_list = [f[0]for f in future_results]
        cat_list = [f[1]for f in future_results]
        key_list = [f[2]for f in future_results]
        ret_num_arr = np.concatenate(num_list, axis=0)
        ret_cat_arr = np.concatenate(cat_list, axis=0)
        ret_key_arr = np.concatenate(key_list, axis=0)
        print(ret_num_arr.shape)
        print(ret_cat_arr.shape)
        print(ret_key_arr.shape)
        return ret_num_arr, ret_cat_arr, ret_key_arr

    def _split_series(self, series):
        series["id_mod"] = series["card_id"].apply(hash)
        series["id_mod"] = series["id_mod"] % self._SPLIT_NUM

        split_series = list()
        for i in range(0, self._SPLIT_NUM):
            one_split = series[series["id_mod"] == i]
            split_series.append(one_split)
        return split_series

    def _do_reshape(self, trans):
        trans = self._do_prep(trans)
        trans = self._do_scale(trans)
        trans_num, num_keys = self.reshape(trans, self.num_col)
        trans_cat, cat_keys = self.reshape(trans, self.cat_col)
        for kn, kc in zip(num_keys, cat_keys):
            assert kn == kc
            if kn != kc:
                print("omg")
        return trans_num, trans_cat, np.array(num_keys)

    @staticmethod
    def _do_prep(df):
        df = df.sort_values(["card_id", "purchase_date"])
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df["no_city"] = np.where(df["city_id"] == -1, 1, 0)
        df["day"] = df['purchase_date'].dt.day
        return df

    def _do_scale(self, df):
        df = pocket_scaler.rank_gauss(df, self.num_col)
        return df

    @staticmethod
    def reshape(df, use_col):
        sample_size = df["card_id"].nunique()
        time_step = 120  # maximum length
        features = len(use_col)
        ret_shape = (sample_size, time_step, features)
        ret_array = np.zeros(ret_shape)  # sample_size, time_step, features

        single_shape = (time_step, features)
        keys = df["card_id"].unique()
        max_len = 0
        for i, the_key in enumerate(keys):
            df_ = df[df["card_id"] == the_key]
            res = df_[use_col].values
            desired = np.zeros(single_shape)
            if res.shape[0] > time_step:
                max_len = max(res.shape[0], max_len)
                desired = res[:time_step, :res.shape[1]]
            else:
                desired[:res.shape[0], :res.shape[1]] = res
            # print(desired)
            ret_array[i] = desired
        print(max_len)

        return ret_array, keys


