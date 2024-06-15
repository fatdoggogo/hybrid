from env import Env
import numpy as np
from archive import analysis


# 由算法自己决定如何调用device的计算卸载方法
# 如本随机算法 根据随机数是否大于0.5决定是否进行计算卸载
# 但不管什么算法,device这个循环不可以少一定要对每个device的调用offload方法，否则设备后续计算无法正确触发
def makeOffloadDecision(env, actions, con_actions, f_actions):
    for device in env.devices:
        # r = np.random.rand()
        # if r > 0.5:
        #     device.offload(1, np.random.rand(), np.random.rand())
        # else:
        #     device.offload(0, 0, 0)
        # device.offload(0, 0, 0)
        device.offload(1,0.2,1)

def run():
    env = Env(1,"random")
    env.logger.info("random algorithm run")
    for episode in range(env.episodes):
        env.logger.info('episode:%d - start' % episode)
        env.episode = episode
        # 环境重置
        env.reset()
        # 为环境第一次运行生成任务，并初始化相应的变量
        # env.setUp()
        # 不管什么算法 timeSlot都从1开始
        for timeSlot in range(1, 2):
            env.clock = timeSlot
            state = env.getEnvState()
            state = np.array(state)
            actions = None
            con_actions = None
            f_actions = None
            # 将如何对device进行卸载决策 通过makeOffloadDecision函数传递
            env.offload(makeOffloadDecision, actions, con_actions, f_actions)
            env.getEnvReward()
            env.stepIntoNextState()
            # collect metrics
    env.outputMetric()
    # failures表示由于处理超时导致的任务失败数,如果failure rate过大，说明需要进行调参
    # errors表示完全本地处理的延时 < 进行计算卸载的延时 或者 本地处理能耗 < 进行计算卸载的能耗
    # error rate过大,且reward基本全为负数，也需要进行调参
    total = env.episodes*env.T*env.numberOfDevice
    print("finished!!! failure rate = %f,and error rate = %f" % (env.failures/total,env.errors/(total*2)))
    analysis.draw(env.envDir, env.algorithmDir)

if __name__=='__main__':
    run()