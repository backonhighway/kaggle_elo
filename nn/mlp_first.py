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

data = input_loader.GoldenLoader.load_small_input()
timer.time("load csv")

# do scaling
train, test, train_x, train_y, test_x = data
# ss = StandardScaler()
# print(train_x.describe())
# cols = train_x.columns
# train_x = train_x.fillna(0)
# test_x = test_x.fillna(0)
# train_x = ss.fit_transform(train_x)
# test_x = ss.fit_transform(test_x)
# train_x = pd.DataFrame(data=train_x, columns=cols)
# test_x = pd.DataFrame(data=test_x, columns=cols)
# print(train_x.describe())
scale_col = [c for c in train_x.columns if c not in ["card_id", "target"]]
train_x = pocket_scaler.rank_gauss(train_x, scale_col).fillna(0)
test_x = pocket_scaler.rank_gauss(test_x, scale_col).fillna(0)

data = (train, test, train_x, train_y, test_x)

trainer = pocket_cv.GoldenTrainer(epochs=100, batch_size=1024)
trainer.do_cv(data)
timer.time("done train")
