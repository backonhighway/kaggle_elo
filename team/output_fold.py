import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)

import pandas as pd
import numpy as np
from elo.common import pocket_file_io, path_const
from sklearn import model_selection

csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.ORG_TRAIN)
print(train.shape)

outliers = (train["target"] < -30).astype(int).values
split_num = 5
random_state = 4590

skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=random_state)
train_preds = []
for idx, (train_index, test_index) in enumerate(skf.split(train, outliers)):
    train_id = train.iloc[test_index]
    train_cv_prediction = pd.DataFrame()
    train_cv_prediction["card_id"] = train_id["card_id"]
    train_cv_prediction["fold"] = idx
    train_preds.append(train_cv_prediction)

train_output = pd.concat(train_preds, axis=0)
print(train_output.head())
print(train_output.shape)
train_output.to_csv("../output/folds.csv", index=False)

