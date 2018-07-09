# -*- coding:utf-8 -*-
import logging
import time
from logging.handlers import WatchedFileHandler

from scripts.settings import LOG_FILE, LOG_LEVEL


formatter = logging.Formatter('[%(filename)-12s]: [%(levelname)-6s] [%(asctime)s]: %(message)s')

watched_file_handler = WatchedFileHandler(LOG_FILE, encoding="utf-8")
watched_file_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)

if LOG_LEVEL == "INFO":
    logger.setLevel(logging.INFO)
    watched_file_handler.setLevel(logging.INFO)
else:
    logger.setLevel(logging.DEBUG)
    watched_file_handler.setLevel(logging.DEBUG)


logger.addHandler(watched_file_handler)

if __name__ == "__main__":

    for i in range(100):
        logger.debug("This is debug infomation")
        logger.info("This is info infomation")
        logger.error("This is error infomation")
        time.sleep(.1)

