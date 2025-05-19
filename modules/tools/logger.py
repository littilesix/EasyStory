import logging
import os,sys
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 4. 重定向标准输出/错误
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass  # 兼容要求

    def fileno(self):
        pass


import os
import sys
import shutil
import logging
from datetime import datetime

LOG_DIR = "logs"
LOG_FILENAME = "app.log"

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            for line in message.rstrip().splitlines():
                self.logger.log(self.level, line)

    def flush(self):
        pass

def rotate_log_file():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, LOG_FILENAME)
    if os.path.exists(log_file):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rotated_name = os.path.join(LOG_DIR, f"{timestamp}.log")
        shutil.move(log_file, rotated_name)

def setup_logging():
    rotate_log_file()

    log_file = os.path.join(LOG_DIR, LOG_FILENAME)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # 文件日志输出（不再使用 RotatingFileHandler）
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    root_logger.propagate = False

    # 屏蔽其他模块的重复日志
    logging.getLogger("urllib3.connectionpool").propagate = False
    logging.getLogger("httpcore.http11").propagate = False
    logging.getLogger("httpcore.connection").propagate = False


"""
def setup_logging():
    log_file = os.path.join(LOG_DIR, "app.log")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # 文件日志输出
    fh = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3,encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    root_logger.propagate = False

    logging.getLogger("urllib3.connectionpool").propagate = False
    logging.getLogger("httpcore.http11").propagate = False
    logging.getLogger("httpcore.connection").propagate = False
"""



