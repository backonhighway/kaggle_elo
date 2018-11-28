import pandas as pd
from sklearn import model_selection
import numpy as np


class GoldenSplitter:

    @staticmethod
    def get_holdout_split(df: pd.DataFrame, target_col="target"):
        df_x = df.drop(target_col, axis=1)
        df_y = df[target_col]
        print(df_x.columns)
        return model_selection.train_test_split(df_x, df_y, test_size=0.2, random_state=99)

    @staticmethod
    def just_get_split(df: pd.DataFrame):
        return model_selection.train_test_split(df, test_size=0.2, random_state=99)

    def get_split_and_weight(self, df, target_col="target"):
        train_x, holdout_x, train_y, holdout_y = self.get_holdout_split(df, target_col)
        w = train_y.value_counts()
        weights = {i: np.sum(w) / w[i] for i in w.index}
        train_w, holdout_w = train_y.map(weights), holdout_y.map(weights)

        return train_x, holdout_x, train_y, holdout_y, train_w, holdout_w
