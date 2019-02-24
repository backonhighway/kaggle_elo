from keras.layers import Dense, BatchNormalization, Dropout, CuDNNGRU, PReLU
from keras.layers import concatenate, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Lambda, Embedding, GaussianDropout, Reshape
from keras import Model
from keras.engine import Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras import optimizers
import pandas as pd
from elo.common import path_const


class GoldenMlp2:
    def __init__(self, epochs=100, batch_size=512, lr_scheduler=None):
        K.clear_session()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler

    def build_model(self, feature_cnt):
        network = GoldenNetwork()
        model = network.build_single_input(feature_cnt)
        sgd = optimizers.SGD(lr=0.01)
        model.compile(loss="mean_squared_error", optimizer=sgd, metrics=['mse'])
        return model

    def do_train_direct(self, fold, model, train_x, valid_x, train_y, valid_y):
        print(train_x.shape)
        print(valid_x.shape)
        print(train_y.shape)
        print(valid_y.shape)
        check_point = ModelCheckpoint(
            path_const.get_weight_file(str(fold)),
            monitor='val_loss', mode='min', save_best_only=True, verbose=0
        )
        es = EarlyStopping(monitor="val_loss", patience=20, verbose=1)
        callbacks = [check_point, es]
        if self.lr_scheduler is not None:
            callbacks.append(self.lr_scheduler)

        history = model.fit(train_x, train_y,
                            validation_data=[valid_x, valid_y],
                            epochs=self.epochs, batch_size=self.batch_size,
                            shuffle=True, verbose=2,
                            callbacks=callbacks)
        return model, history

    def save_history(self, history, file_name_suffix):
        history_ = pd.DataFrame(history.history)
        history_.to_csv(path_const.get_history_file(file_name_suffix))


class GoldenNetwork:

    def __init__(self):
        pass

    @staticmethod
    def build_single_input(feature_cnt=5, verbose=False):
        meta_features = feature_cnt
        mi = Input(shape=(meta_features,))
        m = Dense(64)(mi)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.2)(m)
        m = Dense(32)(m)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.05)(m)
        op = Dense(1, activation="linear")(m)

        model = Model(inputs=[mi], output=op)
        if verbose:
            print(model.summary())
        return model

