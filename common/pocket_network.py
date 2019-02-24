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


class GoldenMlp:
    def __init__(self, epochs=100, batch_size=512, lr_scheduler=None):
        K.clear_session()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler

    def build_model(self, feature_cnt):
        network = GoldenNetwork()
        model = network.build_single_input(feature_cnt)
        sgd = optimizers.SGD(lr=0.1)
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
        es = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
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
        self.base_num = ["elapsed_days"]
        self.base_cat = ['feature_1', 'feature_2', 'feature_3']
        self.base_cat_num = {
            "feature_1": (5, 3),
            "feature_2": (5, 3),
            "feature_3": (5, 3),
        }

        self.time_steps = 120
        self.trans_num = ["purchase_amount", "installments", "purchase_date"]
        self.trans_cat = ["most_recent_sales_range", "most_recent_purchases_range"]
        self.trans_cat_num = {
            "most_recent_sales_range": (5, 3),
            "most_recent_purchases_range": (5, 3)
        }

    @staticmethod
    def build_single_input(feature_cnt=218, verbose=False):
        meta_features = feature_cnt
        mi = Input(shape=(meta_features,))
        m = Dense(512)(mi)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.50)(m)
        m = Dense(256)(m)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.20)(m)
        m = Dense(128)(m)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.2)(m)
        m = Dense(64)(m)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.05)(m)
        op = Dense(1, activation="linear")(m)

        model = Model(inputs=[mi], output=op)
        if verbose:
            print(model.summary())
        return model


    def build_bestfitting_nn(self):
        cat_in = Input(shape=(len(self.base_cat),))
        cat_embeds = []
        for idx, col in enumerate(self.base_cat):
            x = Lambda(lambda ci: ci[:, idx, None])(cat_in)
            x = Embedding(self.base_cat_num[col][0], self.base_cat_num[col][1], input_length=1)(x)
            cat_embeds.append(x)
        embeds = concatenate(cat_embeds, axis=2)
        embeds = GaussianDropout(0.2)(embeds)
        num_in = Input(shape=(len(self.base_num),))
        num_x = Reshape([1, len(self.base_num)])(num_in)
        b = concatenate([embeds, num_x], axis=2)
        b = Flatten()(b)

        trans_cat_in = Input(shape=(self.time_steps, len(self.trans_cat)))
        cat_embeds = []
        for idx, col in enumerate(self.trans_cat):
            x = Lambda(lambda ci: ci[:, :, idx])(trans_cat_in)
            x = Embedding(self.trans_cat_num[col][0], self.trans_cat_num[col][1], input_length=self.time_steps)(x)
            cat_embeds.append(x)
        embeds = concatenate(cat_embeds, axis=2)
        embeds = GaussianDropout(0.2)(embeds)
        trans_num_in = Input(shape=(self.time_steps, len(self.trans_num)))
        t = concatenate([embeds, trans_num_in], axis=2)

        # xi = Input(shape=series_shape)
        t = CuDNNGRU(256)(t)
        t = BatchNormalization()(t)
        t = Dropout(0.20)(t)
        t = Dense(64)(t)
        t = PReLU()(t)
        t = BatchNormalization()(t)
        t = Dropout(0.20)(t)

        b = Dense(256)(b)
        b = PReLU()(b)
        b = BatchNormalization()(b)
        b = Dropout(0.20)(b)
        # +0.33 for this layer, next one not much difference
        b = Dense(128)(b)
        b = PReLU()(b)
        b = BatchNormalization()(b)
        b = Dropout(0.20)(b)

        c = concatenate([b, t])
        # adding a layer here -0.03
        c = Dense(32)(c)
        c = BatchNormalization()(c)
        c = Dropout(0.05)(c)
        op = Dense(1)(c)

        model = Model(inputs=[trans_cat_in, trans_num_in, cat_in, num_in], output=op)
        print(model.summary())
        return model

#
#
# def build_all_nn(time_steps=352, series_features=1, meta_features=7):
#     series_shape = (time_steps, series_features)
#
#     xi = Input(shape=series_shape)
#     x = CuDNNGRU(256)(xi)
#     x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#     x = Dense(64)(x)
#     x = PReLU()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#
#     mi = Input(shape=(meta_features,))
#     m = Dense(256)(mi)
#     m = PReLU()(m)
#     m = BatchNormalization()(m)
#     m = Dropout(0.20)(m)
#     # +0.33 for this layer, next one not much difference
#     m = Dense(128)(m)
#     m = PReLU()(m)
#     m = BatchNormalization()(m)
#     m = Dropout(0.20)(m)
#
#     c = concatenate([x, m])
#     # adding a layer here -0.03
#     c = Dense(32)(c)
#     c = BatchNormalization()(c)
#     c = Dropout(0.05)(c)
#     op = Dense(14, activation="softmax")(c)
#
#     model = Model(inputs=[xi, mi], output=op)
#     # print(model.summary())
#     return model
#
#
# def build_cnn_double(time_steps=352):
#     series_features = 9  # flux, mjd, detected
#     series_shape = (time_steps, series_features)
#
#     xi = Input(shape=series_shape)
#     x = Conv1D(filters=24, kernel_size=2, activation="relu")(xi)
#     x = MaxPooling1D(pool_size=2)(x)
#     x = Conv1D(filters=48, kernel_size=4, activation="relu")(x)
#     x = MaxPooling1D(pool_size=4)(x)
#     x = Conv1D(filters=96, kernel_size=4, activation="relu")(x)
#     x = MaxPooling1D(pool_size=4)(x)
#     # Adding a layer here is bad
#     x = Dropout(0.20)(x)  # this dropout is +0.1
#     x = GlobalMaxPooling1D()(x)  # Global max or Flatten? -> Global max is +0.3
#     x = Dense(64)(x)
#     # x = PReLU()(x)
#     # x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#
#     meta_features = 8  # hostgal_photoz, mwebv...
#     mi = Input(shape=(meta_features,))
#     m = Dense(128)(mi)  # 256 for +0.02?
#     m = PReLU()(m)
#     m = BatchNormalization()(m)
#     m = Dropout(0.20)(m)
#     # improves 0.02
#     # m = Dense(64)(m)
#     # m = PReLU()(m)
#     # m = BatchNormalization()(m)
#     # m = Dropout(0.20)(m)
#
#     c = concatenate([x, m])
#     c = Dense(32)(c)
#     c = BatchNormalization()(c)
#     c = Dropout(0.05)(c)
#     op = Dense(14, activation="softmax")(c)
#
#     model = Model(inputs=[xi, mi], output=op)
#     print(model.summary())
#     return model
#
#
# def build_double_input(time_steps=352, series_features=1, meta_features=7):
#     series_shape = (time_steps, series_features)
#
#     xi = Input(shape=series_shape)
#     x = CuDNNGRU(256)(xi)
#     x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#     x = Dense(64)(x)
#     x = PReLU()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#
#     mi = Input(shape=(meta_features,))
#     m = Dense(128)(mi)
#     m = PReLU()(m)
#     m = BatchNormalization()(m)
#     m = Dropout(0.20)(m)
#
#     c = concatenate([x, m])
#     c = Dense(32)(c)
#     c = BatchNormalization()(c)
#     c = Dropout(0.05)(c)
#     op = Dense(14, activation="softmax")(c)
#
#     model = Model(inputs=[xi, mi], output=op)
#     print(model.summary())
#     return model
#
#
# def build_only_ts(time_steps=352):
#     series_features = 3  # flux, mjd, passband
#     series_shape = (time_steps, series_features)
#
#     xi = Input(shape=series_shape)
#     x = CuDNNGRU(256)(xi)
#     x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#     x = Dense(64)(x)
#     x = PReLU()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.05)(x)
#
#     op = Dense(14, activation="softmax")(x)
#
#     model = Model(inputs=[xi], output=op)
#     print(model.summary())
#     return model
#
#
# def build_single_input():
#     meta_features = 8  # hostgal_photoz, mwebv
#     mi = Input(shape=(meta_features,))
#     m = Dense(128)(mi)
#     m = BatchNormalization()(m)
#     m = Dropout(0.20)(m)
#     c = Dense(32)(m)
#     c = BatchNormalization()(c)
#     c = Dropout(0.20)(c)
#     op = Dense(14, activation="softmax")(c)
#
#     model = Model(inputs=[mi], output=op)
#     print(model.summary())
#     return model
#
#
# def build_gru(time_steps=352):
#     series_features = 3  # flux, mjd, passband
#     series_shape = (time_steps, series_features)
#
#     x = Input(shape=series_shape)
#     x = CuDNNGRU(128)(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.20)(x)
#     x = Dense(64)(x)
#     x = PReLU()(x)
#     # x = BatchNormalization()(x)
#     # x = Dropout(0.20)(x)
#     # x = Dense(32)(x)
#     # x = PReLU()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.05)(x)
#     op = Dense(14, activation="softmax")(x)
#
#     model = Model(input_shape=series_shape, output=op)
#     print(model.summary())
#     return model
