import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from dask import dataframe as dd
import json


class GoldenCsv:

    @staticmethod
    def read_file(path, nrows=None, parse_date=False):
        return pd.read_csv(path, nrows=nrows, parse_dates=parse_date)

    @staticmethod
    def read_dask(path):
        return dd.read_csv(path).compute()

    @staticmethod
    def read_hdf(path):
        return pd.read_hdf(path)

    @staticmethod
    def read_npy(path):
        return np.load(path)

    @staticmethod
    def output_npy(arr, path):
        np.save(path, arr)

    @staticmethod
    def output_csv(df, path):
        df.to_csv(path, index=False)

    @staticmethod
    def output_feather(df, path):
        df.to_feather(path)

    @staticmethod
    def read_json_file(path, nrows=None):
        json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

        df = pd.read_csv(path,
                         converters={column: json.loads for column in json_cols},
                         dtype={'fullVisitorId': 'str'},  # Important!!
                         nrows=nrows)

        for column in json_cols:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}.{sub_col}" for sub_col in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
        return df


