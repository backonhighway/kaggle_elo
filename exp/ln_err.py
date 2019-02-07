import numpy as np
import pandas as pd


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6, np.NaN],
})
df2 = pd.DataFrame({
    "key": [1, 4],
    "col2": [-2, 22],
})
df = pd.merge(df, df2, on="key", how="left")
print(df)
print(np.log1p(df["col2"]))
