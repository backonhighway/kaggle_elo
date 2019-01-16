import pandas as pd
from concurrent import futures
from elo.fe import agg_fe
from elo.common import pocket_timer


class ParaFe:

    def __init__(self, prefix="", split_num=32):
        self._SPLIT_NUM = split_num
        self.fer = agg_fe.AggFe(prefix)
        self.timer = pocket_timer.GoldenTimer()

    def do_para_fe(self, trans):
        self.timer.time("start para fe")
        split_trans = self._split_series(trans)
        future_list = list()
        with futures.ProcessPoolExecutor(max_workers=self._SPLIT_NUM) as executor:
            for s in split_trans:
                future_list.append(executor.submit(self.fer.do_fe, s))
        future_results = [f.result() for f in future_list]
        ret_df = pd.concat(future_results)
        print("ret_df_shape=", ret_df.shape)
        self.timer.time("done para agg")
        return ret_df

    def _split_series(self, series):
        series["id_mod"] = series["card_id"].apply(hash)
        series["id_mod"] = series["id_mod"] % self._SPLIT_NUM

        split_series = list()
        for i in range(0, self._SPLIT_NUM):
            one_split = series[series["id_mod"] == i]
            split_series.append(one_split)
        print("split_shape=", split_series[0].shape)
        return split_series


