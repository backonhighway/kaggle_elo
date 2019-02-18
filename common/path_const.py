import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
APP_ROOT = os.path.join(ROOT, "elo")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUB_DIR = os.path.join(APP_ROOT, "sub")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
ORG_NEW_MERCHANTS = os.path.join(INPUT_DIR, "new_merchant_transactions.csv")
ORG_HISTORICAL = os.path.join(INPUT_DIR, "historical_transactions.csv")
ORG_MERCHANTS = os.path.join(INPUT_DIR, "merchants.csv")

SAMPLE_615 = os.path.join(OUTPUT_DIR, "sample_615.csv")
SAMPLE_730 = os.path.join(OUTPUT_DIR, "sample_730.csv")

TRAIN1 = os.path.join(INPUT_DIR, "train1.csv")
TEST1 = os.path.join(INPUT_DIR, "test1.csv")
TRANSACTIONS1 = os.path.join(INPUT_DIR, "transactions1.csv")

CAT_ARR = os.path.join(INPUT_DIR, "cat_arr.npy")
NUM_ARR = os.path.join(INPUT_DIR, "num_arr.npy")

# For NN
NN_DIR = os.path.join(OUTPUT_DIR, "nn")
TRAIN_ = os.path.join(NN_DIR, "train_.csv")
HOLD_ = os.path.join(NN_DIR, "hold_.csv")
TT_NUM = os.path.join(NN_DIR, "tt_num.npy")
TT_CAT = os.path.join(NN_DIR, "tt_cat.npy")
HT_NUM = os.path.join(NN_DIR, "ht_num.npy")
HT_CAT = os.path.join(NN_DIR, "ht_cat.npy")
NEW_NUM = os.path.join(NN_DIR, "new_num.npy")
NEW_CAT = os.path.join(NN_DIR, "new_cat.npy")
NEW_KEY = os.path.join(NN_DIR, "new_key.npy")
OLD_NUM = os.path.join(NN_DIR, "old_num.npy")
OLD_CAT = os.path.join(NN_DIR, "old_cat.npy")

# For LGBM
OLD_TRANS1 = os.path.join(INPUT_DIR, "old_trans1.csv")
NEW_TRANS1 = os.path.join(INPUT_DIR, "new_trans1.csv")
RE_OLD_TRANS1 = os.path.join(INPUT_DIR, "re_old_trans1.csv")
RE_NEW_TRANS1 = os.path.join(INPUT_DIR, "re_new_trans1.csv")
OLD_TRANS2 = os.path.join(INPUT_DIR, "old_trans2.csv")
NEW_TRANS2 = os.path.join(INPUT_DIR, "new_trans2.csv")
OLD_TRANS3 = os.path.join(INPUT_DIR, "old_trans3.csv")
NEW_TRANS3 = os.path.join(INPUT_DIR, "new_trans3.csv")
OLD_TRANS3_1 = os.path.join(INPUT_DIR, "old_trans3_1.csv")
OLD_TRANS3_2 = os.path.join(INPUT_DIR, "old_trans3_2.csv")
OLD_TRANS4 = os.path.join(INPUT_DIR, "old_trans4.csv")
NEW_TRANS4 = os.path.join(INPUT_DIR, "new_trans4.csv")
OLD_TRANS5 = os.path.join(INPUT_DIR, "old_trans5.csv")
NEW_TRANS5 = os.path.join(INPUT_DIR, "new_trans5.csv")
OLD_TRANS6 = os.path.join(INPUT_DIR, "old_trans6.csv")
NEW_TRANS6 = os.path.join(INPUT_DIR, "new_trans6.csv")
NEW_TRANS7 = os.path.join(INPUT_DIR, "new_trans7.csv")
OLD_TRANS8 = os.path.join(INPUT_DIR, "old_trans8.csv")
NEW_TRANS8 = os.path.join(INPUT_DIR, "new_trans8.csv")
OLD_TRANS9 = os.path.join(INPUT_DIR, "old_trans9.csv")
OLD_TRANS10 = os.path.join(INPUT_DIR, "old_trans10.csv")
OLD_TRANS11 = os.path.join(INPUT_DIR, "old_trans11.csv")
NEW_TRANS11 = os.path.join(INPUT_DIR, "new_trans11.csv")
OLD_TRANS13 = os.path.join(INPUT_DIR, "old_trans13.csv")
NEW_TRANS13 = os.path.join(INPUT_DIR, "new_trans13.csv")

LDA_ALL1 = os.path.join(INPUT_DIR, "lda_all1.csv")
LDA_ALL2 = os.path.join(INPUT_DIR, "lda_all2.csv")

PRED_DIR = os.path.join(INPUT_DIR, "pred_feats")
NEW_DAY_PRED_SUB = os.path.join(PRED_DIR, "new_day_pred_sub.csv")
NEW_DAY_PRED_OOF = os.path.join(PRED_DIR, "new_day_pred_oof.csv")
NEW_PUR_MAX_PRED_SUB = os.path.join(PRED_DIR, "new_pur_max_pred_sub.csv")
NEW_PUR_MAX_PRED_OOF = os.path.join(PRED_DIR, "new_pur_max_pred_oof.csv")

# ts source
TS_OLD_TRAIN = os.path.join(INPUT_DIR, "ts_old_train.csv")
TS_OLD_TEST = os.path.join(INPUT_DIR, "ts_old_test.csv")
TS_NEW_TRAIN = os.path.join(INPUT_DIR, "ts_new_train.csv")
TS_NEW_TEST = os.path.join(INPUT_DIR, "ts_new_test.csv")
NEW_TS_PRED_SUB = os.path.join(PRED_DIR, "new_ts_pred_sub.csv")
NEW_TS_PRED_OOF = os.path.join(PRED_DIR, "new_ts_pred_oof.csv")
OLD_TS_PRED_SUB = os.path.join(PRED_DIR, "old_ts_pred_sub.csv")
OLD_TS_PRED_OOF = os.path.join(PRED_DIR, "old_ts_pred_oof.csv")
FEAT_FROM_TS_OLD = os.path.join(INPUT_DIR, "feat_from_ts_old.csv")
FEAT_FROM_TS_NEW = os.path.join(INPUT_DIR, "feat_from_ts_new.csv")

OUTPUT_SUB = os.path.join(SUB_DIR, "temp_sub.csv")
OUTPUT_OOF = os.path.join(SUB_DIR, "temp_oof.csv")
OUTPUT_ENS = os.path.join(SUB_DIR, "temp_ens.csv")

FEATURE_GAIN = os.path.join(OUTPUT_DIR, "feature_gain.csv")


# NN
def get_weight_file(suffix):
    filename = "keras_" + suffix + ".model"
    return os.path.join(OUTPUT_DIR, filename)


def get_history_file(suffix):
    filename = "history_" + suffix + ".csv"
    return os.path.join(OUTPUT_DIR, filename)





