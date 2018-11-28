import pandas as pd
import numpy as np
from . import pocket_logger
from sklearn import metrics


class GoldenEval:
    def __init__(self, pred_cols=None):
        self.pred_cols = pred_cols
        self.logger = pocket_logger.get_my_logger()

    @staticmethod
    def multi_weighted_logloss(y_true, y_preds):
        """
        @author olivier https://www.kaggle.com/ogrellier
        multi logloss for PLAsTiCC challenge
        """
        # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
        # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
        # with Kyle Boone's post https://www.kaggle.com/kyleboone
        classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
        class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
        if len(np.unique(y_true)) > 14:
            classes.append(99)
            class_weight[99] = 2
        y_p = y_preds
        # Transform y_true in dummies
        y_ohe = pd.get_dummies(y_true)
        # Normalize rows and limit y_preds to 1e-15, 1-1e-15
        y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
        # Transform to log
        y_p_log = np.log(y_p)
        # Get the log for ones, .values is used to drop the index of DataFrames
        # Exclude class 99 for now, since there is no class99 in the training set
        # we gave a special process for that class
        y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
        # Get the number of positives for each class
        nb_pos = y_ohe.sum(axis=0).values.astype(float)
        # Weight average and divide by the number of positives
        class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
        y_w = y_log_ones * class_arr / nb_pos

        loss = - np.sum(y_w) / np.sum(class_arr)

        return loss

    @staticmethod
    def lgb_multi_weighted_logloss(y_preds, train_data):
        y_true = train_data.get_label()
        """
        @author olivier https://www.kaggle.com/ogrellier
        multi logloss for PLAsTiCC challenge
        """
        # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
        # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
        # with Kyle Boone's post https://www.kaggle.com/kyleboone
        classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
        class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
        if len(np.unique(y_true)) > 14:
            classes.append(99)
            class_weight[99] = 2
        y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

        # Transform y_true in dummies
        y_ohe = pd.get_dummies(y_true)
        # Normalize rows and limit y_preds to 1e-15, 1-1e-15
        y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
        # Transform to log
        y_p_log = np.log(y_p)
        # Get the log for ones, .values is used to drop the index of DataFrames
        # Exclude class 99 for now, since there is no class99 in the training set
        # we gave a special process for that class
        y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
        # Get the number of positives for each class
        nb_pos = y_ohe.sum(axis=0).values.astype(float)
        # Weight average and divide by the number of positives
        class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
        y_w = y_log_ones * class_arr / nb_pos

        loss = - np.sum(y_w) / np.sum(class_arr)
        return 'wloss', loss, False

