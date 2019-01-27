import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6, 4],
})
df["col2"] = df["col1"] * 3 - 1
df["target"] = df["col1"] + df["col2"]

df["one"] = 1
x = df.groupby("key")["one"].cumsum()
print(x)
