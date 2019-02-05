import pandas as pd


train = pd.read_pickle("../input/team/marcus_train.pkl")
test = pd.read_pickle("../input/team/marcus_test.pkl")
train_ts = pd.read_pickle("../input/team/time-features-v3-selected_train.pkl")
test_ts = pd.read_pickle("../input/team/time-features-v3-selected_test.pkl")

print(train.shape)
print(test.shape)
print(train_ts.shape)
print(test_ts.shape)
print("------")
print(train.head())
print(test.head())
print(train_ts.head())
print(test_ts.head())

