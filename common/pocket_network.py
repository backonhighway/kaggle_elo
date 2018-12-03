from keras.layers import Dense, BatchNormalization, Dropout, CuDNNGRU, PReLU
from keras.layers import concatenate, Conv1D, Flatten, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Lambda, Embedding, GaussianDropout, Reshape
from keras import Model
from keras.engine import Input


class GoldenNetwork:

    def __init__(self):
        self.base_cat = ['feature_1', 'feature_2', 'feature_3']
        self.base_cat_num = {
            "feature_1": (5, 3),
            "feature_2": (5, 3),
            "feature_3": (5, 3),
        }
        self.base_num = ["elapsed_days"]

    def build_bestfitting_nn(self):
        cat_in = Input(shape=(len(self.base_cat),))
        cat_embeds = []
        for idx, col in enumerate(self.base_cat):
            x = Lambda(lambda x: x[:, idx, None])(cat_in)
            x = Embedding(self.base_cat_num[col][0], self.base_cat_num[col][1], input_length=1)(x)
            cat_embeds.append(x)
        embeds = concatenate(cat_embeds, axis=2)
        embeds = GaussianDropout(0.2)(embeds)
        num_in = Input(shape=(len(self.base_num),))
        num_x = Reshape([1, len(self.base_num)])(num_in)
        b = concatenate([embeds, num_x], axis=2)


        series_shape = (time_steps, series_features)

        cat_in = Input(shape=(time_steps,))
        c = Embedding(6, 4, input_length=time_steps)(cat_in)
        c = GaussianDropout(0.2)(c)
        ser_in = Input(shape=series_shape)
        x = concatenate([ser_in, c], axis=2)

        # xi = Input(shape=series_shape)
        x = CuDNNGRU(256)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.20)(x)
        x = Dense(64)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.20)(x)

        mi = Input(shape=(meta_features,))
        m = Dense(256)(mi)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.20)(m)
        # +0.33 for this layer, next one not much difference
        m = Dense(128)(m)
        m = PReLU()(m)
        m = BatchNormalization()(m)
        m = Dropout(0.20)(m)

        c = concatenate([x, m])
        # adding a layer here -0.03
        c = Dense(32)(c)
        c = BatchNormalization()(c)
        c = Dropout(0.05)(c)
        op = Dense(14, activation="softmax")(c)

        model = Model(inputs=[cat_in, ser_in, mi], output=op)
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