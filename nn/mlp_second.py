import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io
from elo.loader import input_loader
from elo.trainer import pocket_cv
from elo.common import pocket_scaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train, test = input_loader.GoldenLoader().load_team_input_v63()
print(train.head())
timer.time("load csv")

cat_cols = input_loader.GoldenLoader().get_team_cat_col()
key_cols = ["card_id", "target"]
pred_cols = [c for c in train.columns if c not in key_cols and c not in cat_cols]
# train_ohe = pd.get_dummies(train[cat_cols])
train_x = train[pred_cols].copy()
train_y = train["target"].copy()
test_x = test[pred_cols].copy()

# do scaling
timer.time("start scale")
not_scale_col = cat_cols + key_cols
scale_col = [c for c in train.columns if c not in not_scale_col]

train_x = pocket_scaler.rank_gauss(train_x, scale_col).fillna(0)
test_x = pocket_scaler.rank_gauss(test_x, scale_col).fillna(0)
timer.time("done scale")
data = (train, test, train_x, train_y, test_x)

trainer = pocket_cv.GoldenTrainer(epochs=20, batch_size=1024)
trainer.do_cv(data)
timer.time("done train")
