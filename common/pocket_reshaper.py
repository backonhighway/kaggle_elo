import pandas as pd
import numpy as np
import math
from concurrent import futures


class GoldenReshaper:

    def __init__(self, split_num=16, cat_col=None, num_col=None):
        self._SPLIT_NUM = split_num
        self.cat_col = cat_col
        self.num_col = num_col

    def do_para_reshape(self, train_trans, hold_trans):
        train_num, train_cat = self._do_para_reshape(train_trans)
        hold_num, hold_cat = self._do_para_reshape(hold_trans)
        return train_num, train_cat, hold_num, hold_cat

    def _do_para_reshape(self, trans):
        split_trans = self._split_series(trans)
        future_list = list()
        with futures.ProcessPoolExecutor(max_workers=self._SPLIT_NUM) as executor:
            for s in split_trans:
                future_list.append(executor.submit(self._do_reshape, s))
        future_results = [f.result() for f in future_list]
        num_list, cat_list = [f[0]for f in future_results], [f[1]for f in future_results]
        ret_num_arr = np.concatenate(num_list, axis=0)
        ret_cat_arr = np.concatenate(cat_list, axis=0)
        print(ret_num_arr.shape)
        print(ret_cat_arr.shape)
        return ret_num_arr, ret_cat_arr

    def _split_series(self, series):
        series["id_mod"] = series["card_id"].apply(hash)
        series["id_mod"] = series["id_mod"] % self._SPLIT_NUM

        split_series = list()
        for i in range(0, self._SPLIT_NUM):
            one_split = series[series["id_mod"] == i]
            split_series.append(one_split)
        return split_series

    def _do_reshape(self, trans):
        trans_num = self.reshape(trans, self.num_col)
        trans_cat = self.reshape(trans, self.cat_col)
        return trans_num, trans_cat

    @staticmethod
    def reshape(df, use_col):
        sample_size = df["card_id"].nunique()
        time_step = 120  # maximum length
        features = len(use_col)
        ret_shape = (sample_size, time_step, features)
        ret_array = np.zeros(ret_shape)  # sample_size, time_step, features

        single_shape = (time_step, features)
        keys = df["card_id"].unique()
        for i, the_key in enumerate(keys):
            df_ = df[df["card_id"] == the_key]
            res = df_[use_col].values
            desired = np.zeros(single_shape)
            if res.shape[0] > time_step:
                desired = res[:time_step, :res.shape[1]]
            else:
                desired[:res.shape[0], :res.shape[1]] = res
            # print(desired)
            ret_array[i] = desired

        return ret_array

    @staticmethod
    def reshape_crop(df):
        sample_size = df["object_id"].nunique()
        time_step = 352  # maximum length
        features = 1
        ret_shape = (sample_size, time_step, features)
        ret_array = np.zeros(ret_shape)  # sample_size, time_step, features

        keys = df["object_id"].unique()
        use_col = ["flux"]
        for i, the_key in enumerate(keys):
            df_ = df[df["object_id"] == the_key]
            res = df_[use_col].values
            length = res.shape[0]
            if length >= 120:
                max_length = min(length+1, 250)
                random_length = np.random.randint(120, max_length)
                start_point = np.random.randint(0, length-random_length+1)
                res = res[start_point:start_point+random_length+1, :]
                length = res.shape[0]
            repeats = math.ceil(time_step / length)
            res = np.tile(res, (repeats, 1))  # repeat the array
            desired = res[:time_step, :features]  # cut it to the desired shape
            # print(desired)
            ret_array[i] = desired
        return ret_array
