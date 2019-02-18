import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import ts_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

# newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, nrows=1000*100, parse_date=["purchase_date"])
# older = csv_io.read_file(path_const.ORG_HISTORICAL, nrows=1000*100, parse_date=["purchase_date"])
newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, parse_date=["purchase_date"])
older = csv_io.read_file(path_const.ORG_HISTORICAL, parse_date=["purchase_date"])
train = csv_io.read_file(path_const.ORG_TRAIN, parse_date=["first_active_month"])
test = csv_io.read_file(path_const.ORG_TEST, parse_date=["first_active_month"])
timer.time("read csv")
print("-"*40)

old_train, old_test, new_train, new_test = ts_fe.TsFe().do_fe(older, newer, train, test)
timer.time("done fe")


print(old_train.head())
print(old_test.head())
print(new_train.head())
print(new_test.head())

csv_io.output_csv(old_train, path_const.TS_OLD_TRAIN)
csv_io.output_csv(old_test, path_const.TS_OLD_TEST)
csv_io.output_csv(new_train, path_const.TS_NEW_TRAIN)
csv_io.output_csv(new_test, path_const.TS_NEW_TEST)
timer.time("done output")


