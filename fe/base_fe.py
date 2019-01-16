import datetime


class BaseFe:

    def do_all(self, train, test):
        train = self.do_meta(train)
        test = self.do_meta(test)
        return train, test

    @staticmethod
    def do_meta(df):
        df["elapsed_days"] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
        df.drop(columns=["first_active_month"], inplace=True)
        return df
