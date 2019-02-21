from elo.common import pocket_timer, pocket_logger, path_const, evaluator
from elo.common import pocket_network, learning_rate
from sklearn import model_selection
import pandas as pd
import numpy as np


class GoldenTrainer:
    def __init__(self, epochs, batch_size):
        self.logger = pocket_logger.get_my_logger()
        self.epochs = epochs
        self.batch_size = batch_size

    def do_cv(self, data):
        for d in data:
            print(d.shape)
        train, test, train_x, train_y, test_x = data
        timer = pocket_timer.GoldenTimer(self.logger)

        submission = pd.DataFrame()
        submission["card_id"] = test["card_id"]
        submission["target"] = 0
        train_cv = pd.DataFrame()
        train_cv["card_id"] = train["card_id"]
        train_cv["cv_pred"] = 0

        outliers = (train["target"] < -30).astype(int).values
        bagging_num = 1
        split_num = 5
        for bagging_index in range(bagging_num):
            skf = model_selection.StratifiedKFold(n_splits=split_num, shuffle=True, random_state=4590)

            total_score = 0
            train_preds = []
            for idx, (train_index, test_index) in enumerate(skf.split(train, outliers)):
                lr_schedule = learning_rate.GoldenLearningRate(0.1, 20).cosine_annealing_scheduler()
                mlp = pocket_network.GoldenMlp(self.epochs, self.batch_size, lr_schedule)
                network = mlp.build_model(train_x.shape[1])
                X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
                y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

                print("start train")
                model, history = mlp.do_train_direct(str(idx), network, X_train, X_test, y_train, y_test)
                mlp.save_history(history, str(idx))
                print('Loading Best Model')
                model.load_weights(path_const.get_weight_file(str(idx)))

                y_pred = model.predict(test_x, batch_size=self.batch_size)
                y_pred = np.reshape(y_pred, -1)
                y_pred = np.clip(y_pred, -33.219281, 18.0)
                valid_set_pred = model.predict(X_test, batch_size=self.batch_size)
                score = evaluator.rmse(y_test, valid_set_pred)
                print(score)
                total_score += score

                submission["target"] = submission["target"] + y_pred
                train_id = train.iloc[test_index]
                train_cv_prediction = pd.DataFrame()
                train_cv_prediction["card_id"] = train_id["card_id"]
                train_cv_prediction["cv_pred"] = valid_set_pred
                train_preds.append(train_cv_prediction)
                timer.time("done one set in")

            train_output = pd.concat(train_preds, axis=0)
            train_cv["cv_pred"] += train_output["cv_pred"]

            avg_score = str(total_score / split_num)
            self.logger.print("average score= " + avg_score)
            timer.time("end train in ")

        submission["target"] = submission["target"] / (bagging_num * split_num)
        # submission["target"] = np.clip(submission["target"], -33.219281, 18.0)
        submission.to_csv(path_const.OUTPUT_SUB, index=False)

        train_cv["cv_pred"] = train_cv["cv_pred"] / bagging_num
        train_cv["cv_pred"] = np.clip(train_cv["cv_pred"], -33.219281, 18.0)
        train_cv.to_csv(path_const.OUTPUT_OOF, index=False)

        y_true = train_y
        y_pred = train_cv["cv_pred"]
        rmse_score = evaluator.rmse(y_true, y_pred)
        self.logger.print("evaluator rmse score= " + str(rmse_score))

        print(train["target"].describe())
        self.logger.print(train_cv.describe())
        self.logger.print(submission.describe())
        timer.time("done submission in ")
