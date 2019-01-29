import pandas as pd
import numpy as np
from keras.callbacks import LearningRateScheduler
import math


class GoldenLearningRate:

    def __init__(self, initial_lr, period_of_cycle):
        self.initial_lr = initial_lr
        self.period_of_cycle = period_of_cycle
        self.min_lr = 0.0001

    def cosine_annealing_scheduler(self):
        return LearningRateScheduler(self._cosine_learning_rate)

    def _cosine_learning_rate(self, epoch):
        epoch_ = epoch % self.period_of_cycle
        epoch_ = epoch_ / self.period_of_cycle
        epoch_ = epoch_ * np.pi * 0.5
        ret_lr = math.cos(epoch_) * self.initial_lr
        ret_lr = max(ret_lr, self.min_lr)
        return ret_lr
