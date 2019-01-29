import pandas as pd
import numpy as np
from scipy.special import erfinv


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6, np.NaN],
})
df["col2"] = df["col1"] * 3 - 1
df["target"] = df["col1"] + df["col2"]
df["col3"] = df["key"] * df["col1"]

print(df)

for c in df.columns:
    series = df[c].rank()
    M = series.max()
    m = series.min()
    print(c, m, len(series), len(set(df[c].tolist())))
    series = (series - m) / (M - m)
    series = series - series.mean()
    # series = series.apply(erfinv)
    series = erfinv(series)
    df[c] = series

# for col in df.columns:
#     values = sorted(set(df[col]))
#     # Because erfinv(1) is inf, we shrink the range into (-0.9, 0.9)
#     f = pd.Series(np.linspace(-0.9, 0.9, len(values)), index=values)
#     f = np.sqrt(2) * erfinv(f)
#     f -= f.mean()
#     df[col] = df[col].map(f)

print(df)
