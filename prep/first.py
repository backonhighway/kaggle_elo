import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import prep_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

train = csv_io.read_file(path_const.ORG_TRAIN)
test = csv_io.read_file(path_const.ORG_TEST)
newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, nrows=1000*100)
older = csv_io.read_file(path_const.ORG_HISTORICAL, nrows=1000*100)
# newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS)
# older = csv_io.read_file(path_const.ORG_HISTORICAL)
merchants = csv_io.read_file(path_const.ORG_MERCHANTS)
timer.time("read csv")
print("-"*40)

merchants = merchants.dropduplicates(subset=['merchant_id'])
fer = prep_fe.PrepFe()
train, test, trans = fer.do_all(train, test, newer, older, merchants)
timer.time("done fe")

csv_io.output_csv(train, path_const.TRAIN1)
csv_io.output_csv(test, path_const.TEST1)
csv_io.output_csv(trans, path_const.TRANSACTIONS1)
