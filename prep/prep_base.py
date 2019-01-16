import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import base_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.ORG_TRAIN, parse_date=["first_active_month"])
test = csv_io.read_file(path_const.ORG_TEST, parse_date=["first_active_month"])
timer.time("read csv")
print("-"*40)

train, test = base_fe.BaseFe().do_all(train, test)
timer.time("done fe")

csv_io.output_csv(train, path_const.TRAIN1)
csv_io.output_csv(test, path_const.TEST1)
timer.time("done output")

