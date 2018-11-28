from plastic.utils import dict_util


PRED_COL1 = [
    # 'object_id',
    'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz',
    'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv',
    #"weight"
]

SUB_MODEL_COL = [c + "_pred_from_test" for c in ["hostgal_specz", "hostgal_photoz", "distmod"]]


def get_agg_cols():
    _aggs = {
        "flux": ["std", "min", "max", "mean", "median", "skew"],
        "flux_err": ["std", "min", "max", "mean", "median", "skew"],
        "flux_diff": ["std", "min", "max", "mean", "median", "skew"],
        "detected": ["mean"],
    }
    _agg_cols = [k + "_" + agg for k in _aggs.keys() for agg in _aggs[k]]
    _passband_cols = list()
    for passband in range(0, 6):
        _passband_col_names = ["pass" + str(passband) + "_" + c for c in _agg_cols]
        _passband_cols.extend(_passband_col_names)
    return _passband_cols


AGG_COLS = get_agg_cols()


class GoldenCols:
    def get_short_cols(self):
        predict_col = self._get_pure_pass_agg() + self._get_pure_det_agg()\
               + self._get_pure_all_agg() + self._get_other_cols()
        predict_col = [c for c in predict_col if "median" not in c]
        predict_col = [c for c in predict_col if "det_0" not in c]
        predict_col = [c for c in predict_col if "mjd_diff" not in c]
        predict_col = [c for c in predict_col if "flux_slope" not in c]
        predict_col = [c for c in predict_col if "flux_diff" not in c]
        return predict_col

    def get_agg2_cols(self):
        return self._get_pure_pass_agg() + self._get_pure_det_agg()\
               + self._get_pure_all_agg() + self._get_other_cols()

    def get_all_agg_cols(self):
        return ["object_id"] + self._get_pure_all_agg()

    def get_pass_agg_cols(self):
        return ["object_id"] + self._get_pure_pass_agg()

    def get_det_agg_cols(self):
        return ["object_id"] + self._get_pure_det_agg()

    @staticmethod
    def get_sub_cols():
        sub_cols = ["hostgal_specz", "hostgal_photoz", "distmod"]
        return [c + "_pred_from_test" for c in sub_cols] + [c + "_pred_from_train" for c in sub_cols]

    @staticmethod
    def get_umap_cols():
        return ["umap_" + str(i) for i in range(0, 14)]

    @staticmethod
    def _get_other_cols():
        ret_col = ["kernel_mjd",]# "det_flux_width"]
        ret_col.extend(["top_q_mjd_pass" + str(i) for i in range(0, 6)])
        ret_col.extend(["low_q_mjd_pass" + str(i) for i in range(0, 6)])
        ret_col.extend(["90_q_mjd", "10_q_mjd"])
        ret_col.extend(["q90", "q80", "q20", "q10"])
        for i in range(0, 6):
            next_col = "pass_" + str(i) + "_flux_width"
            ret_col.append(next_col)
        ret_col.extend(["_".join(["pass", str(i), "normal_max_mjd"]) for i in range(0, 6)])
        return ret_col

    @staticmethod
    def _get_pure_all_agg():
        aggs = {
            "flux": ["std", "min", "max", "mean", "median", "skew"],
            "flux_err": ["std", "min", "max", "mean", "median", "skew"],
            "flux_diff": ["std", "min", "max", "mean", "median", "skew"],
            "flux_slope": ["std", "min", "max", "mean", "median", "skew"],
            "detected": ["mean"],
        }
        return ["_".join([k, agg]) for k in aggs.keys() for agg in aggs[k]]

    @staticmethod
    def _get_pure_pass_agg():
        aggs = {
            "flux": ["std", "min", "max", "mean", "median", "skew"],
            "flux_err": ["std", "min", "max", "mean", "median", "skew"],
            "flux_diff": ["std", "min", "max", "mean", "median", "skew"],
            "flux_slope": ["std", "min", "max", "mean", "median", "skew"],
            "detected": ["mean"],
            "change_slope": ["sum", "count"],
        }
        aggs = dict_util.to_sorted_dict(aggs)
        return ["_".join(["pass", str(i), k, agg])
                for k in aggs.keys() for agg in aggs[k] for i in range(0, 6)]

    @staticmethod
    def _get_pure_det_agg():
        aggs = {
            "flux": ["std", "min", "max", "mean", "median", "skew"],
            "flux_err": ["std", "min", "max", "mean", "median", "skew"],
            "det_flux_diff": ["std", "min", "max", "mean", "median", "skew"],
            "det_mjd_diff": ["std", "min", "max", "mean", "median", "skew"],
            "det_flux_slope": ["std", "min", "max", "mean", "median", "skew"],
        }
        aggs = dict_util.to_sorted_dict(aggs)
        return ["_".join(["det", str(d), "pass", str(p), k, agg])
                for k in aggs.keys() for agg in aggs[k] for p in range(0, 6) for d in range(0, 2)]

    @staticmethod
    def get_batch_cols():
        ret_col = list()
        _agg_cols = ["detected", "flux", "flux_diff", "flux_err", "count"]
        for col in _agg_cols:
            for i in range(0, 6):
                add_col = col + "_" + str(i)
                ret_col.append(add_col)
        return ["object_id", "batch_num"] + ret_col

    @staticmethod
    def get_batch_agg_cols():
        aggs = dict()
        flux_agg_list = ["std", "min", "max", "mean", "median", "skew"]
        flux_cols = ["flux_0", "flux_1", "flux_2", "flux_3", "flux_4", "flux_5",
                     "flux_01", "flux_12", "flux_23", "flux_34", "flux_45", ]
        for flux_col in flux_cols:
            aggs[flux_col] = flux_agg_list
        for detected_col in ["detected_" + str(i) for i in range(0, 6)]:
            aggs[detected_col] = ["mean"]

        return [k + "_" + agg for k in aggs.keys() for agg in aggs[k]]


CAT_COLS = [
    # "cluster_id"
]

HAS_NAN_COLS = [
]
PRED_NAN_COLS = [ "is_nan_" + col for col in HAS_NAN_COLS]

TRAIN_DROP_COL = [
]
TEST_DROP_COL = [

]


TRAIN_ALL_COLS = PRED_COL1 + AGG_COLS
