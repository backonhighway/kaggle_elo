import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
from elo.common import pocket_timer, pocket_logger, pocket_file_io
from elo.loader import input_loader
from elo.trainer import pocket_cv

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
csv_io = pocket_file_io.GoldenCsv()

data = input_loader.GoldenLoader.load_small_input()
timer.time("load csv")

trainer = pocket_cv.GoldenTrainer(epochs=20, batch_size=512)
trainer.do_cv(data)
timer.time("done train")
