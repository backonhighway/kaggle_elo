import pandas as pd
import numpy as np
import math


class GoldenReshaper:

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
