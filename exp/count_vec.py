import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


df = pd.DataFrame({
    "key": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    "col1": [1, 2, 3, 4, 6, 2, 1, 8, 1, 6],
})

key_list = []
for k in [1, 2]:
    _df = df[df["key"] == k]
    key_list.append(list(_df["col1"]+10))
print(key_list)
key_list = [" ".join(map(str, element)) for element in key_list]
print(key_list)

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(key_list)

print("----")
print(x)
print(type(x))
