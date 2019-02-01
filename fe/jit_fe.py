class JitFe:
    def __init__(self):
        pass

    @staticmethod
    def do_fe(df):
        df["diff_elapsed_days"] = df["new_trans_elapsed_days_max"] - df["old_trans_elapsed_days_min"]
        df["div_cat1"] = df["new_category_1_mean"] / df["old_category_1_mean"]
        df["diff_new_purchase_amount"] = df["new_purchase_amount_max"] - df["new_purchase_amount_min"]
        df["diff_old_purchase_amount"] = df["old_not_auth_purchase_amount_max"] - df["old_not_auth_purchase_amount_min"]
        df["div_city_nunique"] = df["new_city_id_nunique"] / df["old_city_id_nunique"]

        df["new_purchase_amount_count"] = df["new_purchase_amount_count"].fillna(0)
        df["div_new_count_by_month"] = df["new_purchase_amount_count"] / df["new_month_lag_ptp"]
        df["div_old_count_by_month"] = df["old_purchase_amount_count"] / df["old_month_lag_ptp"]
        df["proper_new_purchase_amount_sum"] = df["new_purchase_amount_sum"] + df["new_purchase_amount_count"]
        df["proper_old_purchase_amount_sum"] = df["old_purchase_amount_sum"] + df["old_purchase_amount_count"]

        # df["proper_auth_mean"] = df["old_authorized_flag_mean"] - df["old_seq_non_auth_mean"]
        # df["proper_auth_sum"] = df["old_authorized_flag_sum"] - df["old_seq_non_auth_sum"]

        # seems good without target-encodes
        df["div_rec1_cnt"] = df["rec1_purchase_amount_count"] / df["old_purchase_amount_count"]
        df["div_rec1_inst"] = df["rec1_installments_sum"] / df["old_installments_sum"]
        df["div_rec3_cnt"] = df["rec3_purchase_amount_count"] / df["old_purchase_amount_count"]
        df["div_rec3_inst"] = df["rec3_installments_sum"] / df["old_installments_sum"]
        df["div_rec13_cnt"] = df["rec1_purchase_amount_count"] / df["rec3_purchase_amount_count"]
        df["div_rec13_inst"] = df["rec1_installments_sum"] / df["rec3_installments_sum"]

        # df["new_last_day"] = df["new_last_day"].fillna(df["old_last_day"])
        df['old_CLV'] = df['old_purchase_amount_count'] * df['old_pa2_sum'] / df['old_month_diff_mean']

        df["old_to_last_day"] = (df["old_month_lag_max"] * -30) + (31 - df["old_last_day"])
        df["new_to_last_day"] = ((df["new_month_lag_max"] - 2) * -30) + (31 - df["new_last_day"])

        if "pred_new" in df.columns:
            df["pred_diff"] = df["pred_new"] - df["new_to_last_day"]

        return df
