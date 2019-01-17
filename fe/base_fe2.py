import datetime
import pandas as pd


class BaseFe:

    def do_all(self, train, test, new, old):
        train = self.do_meta(train)
        test = self.do_meta(test)
        # train = self.do_with_trans(train, new, "new")
        train = self.do_with_trans(train, old, "old")
        train = self.drop_unwanted(train)
        # test = self.do_with_trans(test, new, "new")
        test = self.do_with_trans(test, old, "old")
        test = self.drop_unwanted(test)
        return train, test

    @staticmethod
    def do_meta(df):
        df["elapsed_days"] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
        return df

    @staticmethod
    def do_with_trans(base, trans, prefix):
        temp_df = trans.groupby("card_id")["purchase_date"].agg(["max", "min"]).reset_index()
        temp_df.columns = ["card_id", "pd_max", "pd_min"]

        df = pd.merge(base, temp_df, on="card_id", how="left")

        df[prefix + "_first_buy"] = (df["pd_min"] - df["first_active_month"]).dt.days
        df[prefix + "_last_buy"] = (df["pd_max"] - df["first_active_month"]).dt.days
        df.drop(columns=["pd_min", "pd_max"], inplace=True)
        return df

    @staticmethod
    def drop_unwanted(df):
        df.drop(columns=["first_active_month"], inplace=True)
        return df

