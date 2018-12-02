import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
APP_ROOT = os.path.join(ROOT, "elo")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
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
