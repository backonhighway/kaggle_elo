import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6],
})
df["col2"] = df["col1"] * 3 - 1
df["target"] = df["col1"] + df["col2"]

x = df.groupby("key").agg({"col1": "mode"}).reset_index()

print(x)
