import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing


class TsFe:
    def __init__(self):
        pass

    def do_fe(self, old, new, train, test):
        old = self.do_prep(old)
        new = self.do_prep(new)

        old_train = self.get_merged_df(old, train)
        old_test = self.get_merged_df(old, test)
        new_train = self.get_merged_df(new, train)
        new_test = self.get_merged_df(new, test)

        return old_train, old_test, new_train, new_test

    @staticmethod
    def do_prep(df):
        df['authorized_flag'] = df['authorized_flag'].map({'Y': 0, 'N': 1})
        df['category_1'] = df['category_1'].map({'Y': 1, 'N': 0})
        df['category_3'] = df['category_3'].map({'A': 0, 'B': 1, 'C': 2})
        df['dow'] = df['purchase_date'].dt.dayofweek
        df["trans_elapsed_days"] = (datetime.date(2018, 6, 1) - df['purchase_date'].dt.date).dt.days
        df['hour'] = df['purchase_date'].dt.hour
        df['woy'] = df['purchase_date'].dt.weekofyear
        df['month'] = df['purchase_date'].dt.month
        df["day"] = df['purchase_date'].dt.day
        df["installments"] = np.where(df["installments"] == 999, -1, df["installments"])
        df["pa"] = np.round(df['purchase_amount'] / 0.00150265118 + 497.06, 2)

        le = preprocessing.LabelEncoder()
        le.fit(list(df["merchant_id"].values.astype('str')))
        df["merchant_id"] = le.transform(df["merchant_id"].values.astype('str'))

        return df

    @staticmethod
    def get_merged_df(ts, base):
        print(ts.shape)
        merged = pd.merge(ts, base, on="card_id", how="inner")
        print(merged.shape)
        return merged
