import pandas as pd
import numpy as np
from . import pocket_logger
from sklearn import metrics


class GoldenEval:
    def __init__(self, pred_cols=None):
        self.pred_cols = pred_cols
        self.logger = pocket_logger.get_my_logger()

    def rmse(self, y_true, y_preds):
        score = metrics.mean_squared_error(y_true, y_preds) ** 0.5
        self.logger.print(score)
        return score

