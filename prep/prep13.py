import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import para_fe2, post_rad_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

# newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, nrows=1000*100, parse_date=["purchase_date"])
# older = csv_io.read_file(path_const.ORG_HISTORICAL, nrows=1000*100, parse_date=["purchase_date"])
newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, parse_date=["purchase_date"])
older = csv_io.read_file(path_const.ORG_HISTORICAL, parse_date=["purchase_date"])
timer.time("read csv")
print("-"*40)

fer = post_rad_fe.PostRadFe("new")
new_trans = para_fe2.ParaFe(fer).do_para_fe(newer)
fer = post_rad_fe.PostRadFe("old")
old_trans = para_fe2.ParaFe(fer).do_para_fe(older)
timer.time("done fe")

print(new_trans.head())
print(old_trans.head())
print(new_trans.describe())
print(old_trans.describe())

csv_io.output_csv(old_trans, path_const.OLD_TRANS13)
csv_io.output_csv(new_trans, path_const.NEW_TRANS13)
timer.time("done output")


print(old_trans.describe())

