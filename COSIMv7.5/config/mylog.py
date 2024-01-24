import os
import logging
import logging.handlers
def get_logger(module_name=None,logDir=None):
    logging.basicConfig() 
    logger = logging.getLogger(module_name) 
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # 每30M一个文件 最多保留10个历史文件
    logPath = logDir +"/runs.log"
    filehandler = logging.handlers.RotatingFileHandler(logPath, mode='w', maxBytes=1024*1024*30, backupCount=10)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger