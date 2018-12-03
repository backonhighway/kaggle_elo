import numpy as np

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train_ = csv_io.read_file(path_const.TRAIN_)
hold_ = csv_io.read_file(path_const.HOLD_)
train_trans_num = csv_io.read_npy(path_const.TT_NUM)
train_trans_cat = csv_io.read_npy(path_const.TT_CAT)
hold_trans_num = csv_io.read_npy(path_const.HT_NUM)
hold_trans_cat = csv_io.read_npy(path_const.HT_CAT)
timer.time("load csv")

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from elo.common import pocket_network, pocket_eval

K.clear_session()

epochs = 100
batch_size = 512
ev = pocket_eval.GoldenEval()

checkPoint = ModelCheckpoint("./keras.model", monitor='val_loss', mode='min', save_best_only=True, verbose=0)
es = EarlyStopping(monitor="val_loss", patience=50, verbose=1)

_feats = train_.shape[1]
num_feats = train_trans_num.shape[2]
logger.print(meta_feats)
model = pocket_network.build_bestfitting_nn(series_features=series_feats, meta_features=meta_feats)
model.compile(loss="mean_squared_error", optimizer='adam', metrics=['accuracy'])
history = model.fit([train_series_, train_meta_], [y_train],
                    validation_data=[[holdout_series_, holdout_meta_], y_valid],
                    epochs=epochs,
                    batch_size=batch_size, shuffle=True, verbose=0,
                    callbacks=[checkPoint])

# plot_loss_acc(history)

print('Loading Best Model')
model.load_weights('./keras.model')
# # Get predicted probabilities for each class
y_pred = model.predict([holdout_series_, holdout_meta_], batch_size=batch_size)
logger.print(ev.rmse(y_valid, y_pred))

timer.time("done training")
