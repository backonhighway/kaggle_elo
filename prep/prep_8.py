import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import first_last_fe, para_fe2
import pandas as pd

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

# newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, nrows=1000*100, parse_date=["purchase_date"])
# older = csv_io.read_file(path_const.ORG_HISTORICAL, nrows=1000*100, parse_date=["purchase_date"])
newer = csv_io.read_file(path_const.ORG_NEW_MERCHANTS, parse_date=["purchase_date"])
older = csv_io.read_file(path_const.ORG_HISTORICAL, parse_date=["purchase_date"])
timer.time("read csv")
print("-"*40)

fer = first_last_fe.FirstLastFe("new")
new_trans = para_fe2.ParaFe(fer).do_para_fe(newer)
fer = first_last_fe.FirstLastFe("old")
old_trans = para_fe2.ParaFe(fer).do_para_fe(older)
timer.time("done fe")

pd.options.display.max_columns = 20
print(new_trans.head())
print(old_trans.head())
print(new_trans.describe())
print(old_trans.describe())

csv_io.output_csv(old_trans, path_const.OLD_TRANS8)
csv_io.output_csv(new_trans, path_const.NEW_TRANS8)
timer.time("done output")


print(old_trans[old_trans["card_id"] == "C_ID_0382b662f4"])