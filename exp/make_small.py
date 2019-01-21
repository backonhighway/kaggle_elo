import pandas as pd

new = pd.read_csv("../input/new_merchant_transactions.csv", nrows=1000*100)

old = pd.read_csv("../input/historical_transactions.csv", nrows=1000*100)

train = pd.read_csv("../input/train.csv")

new = pd.merge(new, train, on="card_id", how="left")
old = pd.merge(old, train, on="card_id", how="left")
new.to_csv("../input/new_small.csv")
old.to_csv("../input/old_small.csv")
