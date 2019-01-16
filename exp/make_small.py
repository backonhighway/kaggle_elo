import pandas as pd

new = pd.read_csv("../input/new_merchant_transactions.csv", nrows=1000*100)
new.to_csv("../input/new_small.csv")

old = pd.read_csv("../input/historical_transactions.csv", nrows=1000*100)
old.to_csv("../input/old_small.csv")
