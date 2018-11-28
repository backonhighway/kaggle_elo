import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
APP_ROOT = os.path.join(ROOT, "plastic")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN_BASE = os.path.join(INPUT_DIR, "training_set_metadata.csv")
ORG_TRAIN_SERIES = os.path.join(INPUT_DIR, "training_set.csv")
ORG_TEST_BASE = os.path.join(INPUT_DIR, "test_set_metadata.csv")
ORG_TEST_SERIES = os.path.join(INPUT_DIR, "test_set.csv")
SAMPLE_615 = os.path.join(OUTPUT_DIR, "sample_615.csv")
SAMPLE_730 = os.path.join(OUTPUT_DIR, "sample_730.csv")

TRAIN1 = os.path.join(INPUT_DIR, "train1.csv")
TEST1 = os.path.join(INPUT_DIR, "test1.csv")
TRAIN2 = os.path.join(INPUT_DIR, "train2.csv")
TEST2 = os.path.join(INPUT_DIR, "test2.csv")
TRAIN3 = os.path.join(INPUT_DIR, "train3.csv")
TEST3 = os.path.join(INPUT_DIR, "test3.csv")

TRAIN_BATCH = os.path.join(INPUT_DIR, "train_batch.csv")
TEST_BATCH = os.path.join(INPUT_DIR, "test_batch.csv")
TRAIN_BATCH_AGG = os.path.join(INPUT_DIR, "train_batch_agg.csv")
TEST_BATCH_AGG = os.path.join(INPUT_DIR, "test_batch_agg.csv")

TRAIN_WITH_SUB_FROM_TEST = os.path.join(INPUT_DIR, "train_with_sub_from_test.csv")
TEST_WITH_SUB_FROM_TEST = os.path.join(INPUT_DIR, "test_with_sub_from_test.csv")
TRAIN_WITH_SUB_FROM_TRAIN = os.path.join(INPUT_DIR, "train_with_sub_from_train.csv")
TEST_WITH_SUB_FROM_TRAIN = os.path.join(INPUT_DIR, "test_with_sub_from_train.csv")

SUB = os.path.join(OUTPUT_DIR, "sub.csv")
SUB_MAX1 = os.path.join(OUTPUT_DIR, "sub_1-max.csv")
SUB_PROB1 = os.path.join(OUTPUT_DIR, "sub_1-p_prod.csv")
SUB_GAL_ADJUST = os.path.join(OUTPUT_DIR, "sub_gal_adjust.csv")
SUB_99_ADJUST = os.path.join(OUTPUT_DIR, "sub_99_adjust.csv")

SUB_LGB = os.path.join(OUTPUT_DIR, "sub_lgb.csv")
SUB_NN = os.path.join(OUTPUT_DIR, "sub_nn.csv")
SUB_AVG = os.path.join(OUTPUT_DIR, "sub_avg.csv")

TRAIN_SMALL = os.path.join(INPUT_DIR, "train_small.csv")
TEST_SMALL = os.path.join(INPUT_DIR, "test_small.csv")
TRAIN_SMALL2 = os.path.join(INPUT_DIR, "train_small2.csv")
TEST_SMALL2 = os.path.join(INPUT_DIR, "test_small2.csv")

FEATURE_IMPORTANCE_LOG = os.path.join(OUTPUT_DIR, "feature_importance_log.csv")

UMAP1 = os.path.join(INPUT_DIR, "umap1.csv")
KMEANS1 = os.path.join(INPUT_DIR, "kmeans1.csv")

