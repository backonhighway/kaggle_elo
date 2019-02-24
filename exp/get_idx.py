import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6],
})
df["col2"] = df["col1"] * 3 - 1
df["target"] = df["col1"] + df["col2"]


df2 = df[df["col1"] < 8]
print(df2)
print(df2.index)
print("-----")
idx_list = [0, 3, 7]
idx_list = [i for i in idx_list if i in df2.index]
x = df2.iloc[idx_list]
print(x)

