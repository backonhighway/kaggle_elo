from sklearn import preprocessing
import datetime
import pandas as pd


class PrepFe:

    def do_all(self, train, test, new, old, mer):
        new["is_new"] = 1
        old["is_new"] = 0
        train = self.do_for_both(train)
        test = self.do_for_both(test)
        new = self.do_transaction(new)
        old = self.do_transaction(old)
        mer = self.do_merchants(mer)
        trans = pd.concat([new, old], axis=0, ignore_index=True)
        trans = pd.merge(trans, mer, on="merchant_id", how="left")
        return train, test, trans

    @staticmethod
    def do_for_both(df):
        df["elapsed_days"] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
        return df

    @staticmethod
    def do_merchants(df):
        # dropper = [
        #     "numerical_1", "numerical_2", "category_1",
        #     "active_months_lag3", "average_purchases_lag3", "average_sales_lag3"
        #     "active_months_lag6", "average_purchases_lag6", "average_sales_lag6"
        # ]
        # large_col = [
        #     "merchant_group_id", "most_recent_sales_range", "most_recent_purchases_range",
        #     "category_4", "category_2"
        # ]
        cat_col = [
            "category_1", "category_2", "most_recent_sales_range", "most_recent_purchases_range",
        ]
        for col in cat_col:
            print(col)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[col].values.astype('str')))
            df[col] = lbl.transform(list(df[col].values.astype('str')))
        use_col = [
            "merchant_id", "merchant_category_id", "subsector_id",
            "active_months_lag12", "avg_purchases_lag12", "avg_sales_lag12",
            "most_recent_sales_range", "most_recent_purchases_range",
        ]
        return df[use_col]

    @staticmethod
    def do_transaction(df):
        # cat_cols = [
        #     "city_id", "state_id"
        # ]
        # num_cols = [
        #     "installments", "month_lag", "purchase_amount", "purchase_date"
        # ]
        # key_cols = [
        #     "card_id", "merchant_id"
        # ]
        # df["hour_diff"] = df["purchase_date"].shift()
        # df["hour_diff"] = (df["hour_diff"] - df["purchase_date"]).dt.hour
        df["purchase_date"] =\
            (datetime.date(2018, 2, 1) - df['purchase_date'].dt.date).dt.days

        use_col = [
            "card_id", "merchant_id",
            "city_id", "state_id",  # "subsector_id"
            "installments", "month_lag", "purchase_amount", "purchase_date", # "hour_diff"
        ]
        return df[use_col]

