import pandas as pd
pocket = pd.read_csv("/Users/shota.okubo/Downloads/second_lgb.csv")
owruby = pd.read_csv("/Users/shota.okubo/Downloads/ensembles.csv")

print(pocket.describe())
print(owruby.describe())

merged = pd.merge(pocket, owruby, on="signal_id", how="inner")
print(merged.describe())

merged["00"] = ((merged["target_x"] == 0) & (merged["target_y"] == 0)).astype(int)
merged["01"] = ((merged["target_x"] == 0) & (merged["target_y"] == 1)).astype(int)
merged["10"] = ((merged["target_x"] == 1) & (merged["target_y"] == 0)).astype(int)
merged["11"] = ((merged["target_x"] == 1) & (merged["target_y"] == 1)).astype(int)

show_col = ["00", "01", "10", "11"]
print(merged[show_col].describe())

