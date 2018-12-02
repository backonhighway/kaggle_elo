import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import prep_fe, reshaper

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()
#
# trans = csv_io.read_file(path_const.TRANSACTIONS1)
trans = csv_io.read_file(path_const.TRANSACTIONS1, nrows=1000*100)
timer.time("read csv")
print("-"*40)

reshape_fe = reshaper.GoldenReshaper()
num_arr = reshape_fe.reshape(trans, ["purchase_amount", "installments", "purchase_date"])
cat_arr = reshape_fe.reshape(trans, ["most_recent_sales_range", "most_recent_purchases_range"])
timer.time("done fe")

csv_io.output_npy(num_arr, path_const.NUM_ARR)
csv_io.output_npy(cat_arr, path_const.CAT_ARR)
timer.time("done output")
