from plastic.utils import dict_util


PRED_COL1 = [
    # 'object_id',
    # 'ra', 'decl', 'gal_l', 'gal_b', #'ddf', #'hostgal_specz',
    'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv',
    "galactic_flag"
]



class GoldenCols:
    def get_short_cols(self):
        predict_col = self._get_pure_pass_agg() + self._get_pure_det_agg()\
               + self._get_pure_all_agg() + self._get_other_cols()
        return predict_col

    def get_agg2_cols(self):
        return self._get_other_cols() + self._get_pure_pass_agg() + self._get_pure_det_agg()

    def get_all_agg_cols(self):
        return ["object_id"] + self._get_pure_all_agg()

    def get_pass_agg_cols(self):
        return ["object_id"] + self._get_pure_pass_agg()

    def get_det_all_cols(self):
        return ["object_id"] + self._get_pure_det_all()

    def get_det_agg_cols(self):
        return ["object_id"] + self._get_pure_det_agg()

    @staticmethod
    def _get_other_cols():
        ret_col = ["kernel_mjd",]# "det_flux_width"]
        # ret_col.extend(["top_q_mjd_pass" + str(i) for i in range(0, 6)])
        # ret_col.extend(["low_q_mjd_pass" + str(i) for i in range(0, 6)])
        # for i in range(0, 6):
        #     next_col = "pass_" + str(i) + "_flux_width"
        #     ret_col.append(next_col)
        return ret_col

    @staticmethod
    def _get_pure_all_agg():
        aggs = {
            # "flux": ["std", "mean", "skew"],
            "flux_diff": ["min", "max"],
            "detected": ["mean"],
        }
        return ["_".join([k, agg]) for k in aggs.keys() for agg in aggs[k]]

    @staticmethod
    def _get_pure_pass_agg():
        aggs = {
            "flux": ["skew"],
            "detected": ["mean"],
            # "change_slope": ["sum"]
        }
        aggs = dict_util.to_sorted_dict(aggs)
        return ["_".join(["pass", str(i), k, agg])
                for k in aggs.keys() for agg in aggs[k] for i in range(0, 6)]

    @staticmethod
    def _get_pure_det_agg():
        aggs = {
            "flux": ["std", "min", "max"],
            # "det_flux_slope": ["min", "max", "skew"],
        }
        aggs = dict_util.to_sorted_dict(aggs)
        return ["_".join(["det", "pass", str(p), k, agg])
                for k in aggs.keys() for agg in aggs[k] for p in range(0, 6)]

    @staticmethod
    def _get_pure_det_all():
        aggs = {
            "flux": ["std", "min", "max", "mean", "skew"],
            "det_flux_diff":  ["min", "max", "skew"],
            "det_flux_slope": ["min", "max", "skew"],
        }
        return ["_".join(["det_all", k, agg]) for k in aggs.keys() for agg in aggs[k]]

