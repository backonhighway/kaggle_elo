import numpy as np
X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 1.8], [6, 2]])
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
# k = model.fit(X)
# y = model.inverse_transform(X)
# print(y)

W = model.fit_transform(X)
H = model.components_
print(W)
print(H)
print(np.dot(W, H))

