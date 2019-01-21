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
        return df
