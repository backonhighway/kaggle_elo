import pandas as pd
import numpy as np
import datetime


class OldFirstMerFe2:
    def __init__(self):
        pass

    def do_fe(self, old):
        old = self._do_prep(old)
        ret_df = self._do_agg(old)
        return ret_df

    @staticmethod
    def _do_prep(old):
        ret_df = old.sort_values(by=["card_id", "purchase_date"])
        ret_df = ret_df.groupby(["card_id", "merchant_id"]).first().reset_index()
        return ret_df

    def _do_agg(self, old):
        rec1_df = self._do_agg_monthly(old, -1)
        rec3_df = self._do_agg_monthly(old, -3)
        return pd.merge(rec1_df, rec3_df, on="card_id", how="left")

    @staticmethod
    def _do_agg_monthly(old, month):
        for_agg = old[old["month_lag"] >= month]
        aggs = {
            "purchase_amount": ["max", "min", "mean", "count"],
            "installments": ["sum"],
        }
        ret_df = for_agg.groupby("card_id").agg(aggs).reset_index()
        cols = ["_".join(["old", str(month), k, agg]) for k in aggs.keys() for agg in aggs[k]]
        ret_df.columns = ["card_id"] + cols
        return ret_df











