import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, np.NaN, 6, np.NaN],
})
df["col2"] = df["col1"] * 3 - 1
df["target"] = df["col1"] + df["col2"]

x = df.groupby("key")[["col1", "col2"]].apply(lambda s: s.isna().sum()).reset_index()

print(x)
