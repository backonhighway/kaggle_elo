import pandas as pd
import numpy as np


df = pd.DataFrame({
    "key": ["abcc", "asoij", "xxxx"],
    "col1": [1, 2, 3],
})


df["id_mod"] = df["key"].apply(hash)
df["id_mod_"] = df["col1"] % 7
print(df)
