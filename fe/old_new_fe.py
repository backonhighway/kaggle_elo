import pandas as pd
import numpy as np
import datetime


class OldNewFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, old, new):
        new = self._do_prep(old, new)
        ret_df = self._do_mer(new)
        return ret_df

    @staticmethod
    def _do_prep(old, new):
        old_merchants = set(old["merchant_id"])
        new["is_new_mer"] = np.where(new["merchant_id"].isin(old_merchants), 0, 1)
        return new

    def _do_mer(self, new):
        aggs = {
            "is_new_mer": ["sum", "mean"],
        }
        ret_df = new.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join([self.prefix, k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df














