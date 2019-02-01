import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io, path_const
from elo.fe import merchant_fe2
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

old_trans = merchant_fe2.MerchantFe2("old").do_fe(older)
new_trans = merchant_fe2.MerchantFe2("new").do_fe(newer)
timer.time("done fe")

pd.options.display.max_columns = 20
print(old_trans.columns)
print(old_trans.describe())
print(new_trans.columns)
print(new_trans.describe())

csv_io.output_csv(old_trans, path_const.OLD_TRANS11)
csv_io.output_csv(new_trans, path_const.NEW_TRANS11)
timer.time("done output")

print(old_trans[old_trans["card_id"] == "C_ID_0382b662f4"])

