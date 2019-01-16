import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6],
})
df["col2"] = df["col1"] * 3 - 1
df["target"] = df["col1"] + df["col2"]


x = pd.crosstab(df["key"], df["col1"]).reset_index()
print(x)
print(x.shape)
print(x.columns)

counter = df.groupby("key")["col1"].agg("count").reset_index()
counter.columns = ["key", "cnt"]
print(counter)

x = pd.merge(x, counter, on="key", how="left")
print(x)