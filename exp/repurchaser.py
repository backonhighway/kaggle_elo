import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "col1": [1, 2, 3, 4, 6, 19, 9, 8, 1, 6],
})


def doit(series):
    temp = series.value_counts()
    repeat_cnt = (temp > 1).sum()
    temp_cnt = temp.count()
    print(repeat_cnt)
    print(temp_cnt)
    print(repeat_cnt/temp_cnt)
    return repeat_cnt/temp_cnt


x = df.groupby("key")["col1"].apply(lambda x: doit(x))


print(x)
