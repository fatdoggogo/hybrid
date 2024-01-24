from env import Env
import analysis
from RLBrain_DQN import DeepQNetwork as DQN

def randombin(numberOfDevice, action):
    """

    :param numberOfDevice: 设备数量
    :param action: 动作
    :return: 1001这样的字符串
    """
    userlist = list(bin(action).replace('0b', ''))
    zeros = numberOfDevice - len(userlist)
    ll = [0 for i in range(zeros)]
    for i in userlist:
        ll.append(int(i))
    return ll

def makeOffloadDecision(env,actions):
    """
    当前用于进行device中offload的特殊函数，传入env中进行操作
    :param env: 环境
    :param actions: 动作
    :return:
    """
    userlist = randombin(env.numberOfDevice,actions)
    for device in env.devices:
        if userlist[device.id - 1] == 1:
            device.offload(1,1)
        else:
            device.offload(0,0)
def run():
    env = Env(1,"rl_dqn")
    n_actions = 2 ** env.numberOfDevice
    n_features = env.numberOfDevice * 4 + env.numberOfServer * 2
    RL = DQN(n_actions, n_features,
              learning_rate=0.01,
              reward_decay=0.9,
              e_greedy=0.9,
              replace_target_iter=200,
              memory_size=2000,
              )

    step = 0
    for episode in range(env.episodes):
        env.logger.info('episode:%d - start' % episode)
        env.episode = episode
        # reset
        env.reset()
        # set up
        # env.setUp()
        state = env.getEnvState()
        for timeSlot in range(1, env.T + 1):
            env.clock = timeSlot
            # 获取环境的当前状态
            # RL choose action based on observation
            actions = RL.choose_action(state)
            # 对每个device执行计算卸载动作
            env.offload(makeOffloadDecision,actions)
            # 获取环境奖励
            reward = env.getEnvReward()
            # 更新到下一个状态
            env.stepIntoNextState()
            # 获取新状态
            state_ = env.getEnvState()
            RL.store_transition(state, actions, reward, state_)
            if step > 50 and (step % 10 == 0):
                RL.learn()
            state = state_
        step += 1
    # collect metrics
    env.outputMetric()
    total = env.episodes*env.T*env.numberOfDevice
    print("finished!!! failure rate = %f,and error rate = %f" % (env.failures/total,env.errors/(total*2)))
    analysis.draw(env.envDir,env.algorithmDir)
if __name__=='__main__':
    run()