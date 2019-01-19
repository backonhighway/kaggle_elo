import pandas as pd
import numpy as np
import datetime


class MerchantFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, trans, mer):
        mer_df = self.do_prep(mer)
        ret_df = self.do_agg(trans, mer_df)
        return ret_df

    @staticmethod
    def do_prep(df):
        ret_df = df.drop_duplicates(["merchant_id"])
        ret_df["mer_rank"] = ret_df.index
        ret_df['category_4'] = ret_df['category_4'].map({'Y': 1, 'N': 0})
        use_col = ["merchant_id", "mer_rank", "merchant_group_id", "category_4"]
        return ret_df[use_col]

    def do_agg(self, trans, mer):
        print(trans.shape)
        merged = pd.merge(trans, mer, on="merchant_id", how="left")
        print(merged.shape)
        aggs = {
            "mer_rank": ["mean", "max", "min"],
            "merchant_group_id": ["nunique"],
            "category_4": ["mean"],
        }
        ret_df = merged.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df














