import pandas as pd
import numpy as np
import datetime


# can not parallel
class MerchantFe2:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, ts):
        ret_df = self.do_agg(ts)
        return ret_df

    @staticmethod
    def do_prep(df):
        pass

    def do_agg(self, ts):
        ts["mer_cnt_whole"] = ts.groupby("merchant_id")["merchant_id"].transform("count")
        ts["mer_cnt_in_card"] = ts.groupby(["card_id", "merchant_id"])["merchant_id"].transform("count")
        ts["mer_share"] = ts["mer_cnt_in_card"]/ ts["mer_cnt_whole"]

        aggs = {
            "mer_cnt_whole": ["mean", "max", "min"],
            "mer_share": ["mean", "max", "min", "sum"]
        }
        ret_df = ts.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df















