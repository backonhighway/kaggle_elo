import pandas as pd
import numpy as np
import datetime
from sklearn import preprocessing


class FromTsFe:
    def __init__(self, prefix):
        self.prefix = prefix

    def do_fe(self, df):
        return self._do_agg(df)

    def _do_agg(self, df):
        agg_name = "pred_from_" + self.prefix + "_ts"
        aggs = {
            agg_name: ["mean", "max", "min"]
        }

        all_agg = df.groupby(["card_id"]).agg(aggs).reset_index()
        cols = ["_".join([k, agg]) for k in aggs.keys() for agg in aggs[k]]
        all_agg.columns = ["card_id"] + cols
        return all_agg

