import logging
import os

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
LOGS_DIR = os.path.join(APP_ROOT, "logs")
LOG_FILE = os.path.join(LOGS_DIR, "temp.log")


def get_my_logger():
    return GoldenLogger()


# Wrapped logger
class GoldenLogger:
    def __init__(self):
        log_fmt = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s '
        file_name = LOG_FILE
        logging.basicConfig(format=log_fmt,
                            datefmt='%Y-%m-%d/%H:%M:%S',
                            filename='../logs/temp.log',
                            level='INFO')
        self.logger = logging.getLogger(__name__)

    def info(self, text):
        self.logger.info(text)

    def print(self, text):
        self.logger.info(text)
        print(text)
