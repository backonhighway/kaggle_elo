import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
import pandas as pd

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
es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)

network = pocket_network.GoldenNetwork()
model = network.build_bestfitting_nn()
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['acc'])

train_in = [train_trans_cat, train_trans_num, train_[network.base_cat], train_[network.base_num]]
hold_in = [hold_trans_cat, hold_trans_num, hold_[network.base_cat], hold_[network.base_num]]
y_train, y_valid = train_["target"], hold_["target"]
timer.time("start training...")

history = model.fit(train_in, [y_train],
                    validation_data=[hold_in, y_valid],
                    epochs=epochs,
                    batch_size=batch_size, shuffle=True, verbose=0,
                    callbacks=[checkPoint, es])
# plot_loss_acc(history)

print('Loading Best Model')
model.load_weights('./keras.model')
# # Get predicted probabilities for each class
y_pred = model.predict(hold_in, batch_size=batch_size)
logger.print(ev.rmse(y_valid, y_pred))
history_ = pd.DataFrame(history.history)
history_.to_csv("./history.csv")

timer.time("done training")
