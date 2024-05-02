import logging
import os
import shutil
from env import Env
import RLBrain_CAPQL


def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = os.path.join(log_dir, 'run.log')

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 如果还需要在控制台打印日志
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(message)s')
    # console_handler.setFormatter(formatter)
    # logging.getLogger('').addHandler(console_handler)


if __name__ == '__main__':
    algorithmDir = '../result/rl_capql'
    if os.path.exists(algorithmDir):
        shutil.rmtree(path=algorithmDir)
    imageDir = algorithmDir + '/images'
    metricDir = algorithmDir + '/metrics'
    os.makedirs(imageDir, exist_ok=True)
    os.makedirs(metricDir, exist_ok=True)
    setup_logging(log_dir='../result/rl_capql/log')
    env = Env(1, "rl_capql")
    capql = RLBrain_CAPQL.CAPQL(env=env)
    capql.run()
