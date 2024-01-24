import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import MultivariateNormal

import analysis


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


def makeOffloadDecision(env, dis_action, con_action, f_action):
    """
    当前用于进行device中offload的特殊函数，传入env中进行操作
    :param env: 环境
    :param actions: 动作
    :return:
    """
    userlist = randombin(env.numberOfDevice, dis_action)
    i = 0
    for device in env.devices:
        if userlist[device.id - 1] == 1:
            device.offload(1, con_action[dis_action*6+i], f_action[dis_action*6+i])
        else:
            device.offload(0, 0, 0)
        i += 1


class Actor(nn.Module):

    def __init__(self, s_dim, dis_a_dim, con_a_dim, f_a_dim):
        """

        :param s_dim: 输入的环境维度
        :param dis_a_dim: 输出的离散动作维度
        :param con_a_dim: 输出的连续动作维度
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.disfc = nn.Linear(64, dis_a_dim)
        self.confc = nn.Linear(64, con_a_dim)
        self.ffc = nn.Linear(64, f_a_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs):
        """
        1.离散的动作需要转换成概率？（已转）
        2.连续的动作有没有范围？（使用sigmoid变成了0-1之间）
        3.我这里先设定共享层有3层，如果效果不好再改
        4.在训练时，应该把离散的损失函数与连续的损失函数加起来作为总的损失函数（根据论文猜测）。如果是这样的话，前面的共享层能否得到良好的拟合会是个迷，效果不好再问
        :param obs:输入环境
        :return:离散动作，连续动作
        """
        # 如果输入的是数组，就将其转换成tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        # 编辑前面的共享层
        share1 = F.relu(self.fc1(obs))
        share2 = F.relu(self.fc2(share1))
        # 编辑离散动作层
        dis1 = self.sigmoid(self.disfc(share2))
        # 编辑连续动作层
        con1 = self.sigmoid(self.confc(share2))
        # 编辑计算速率层
        ffc1 = self.sigmoid(self.ffc(share2))

        return dis1, con1, ffc1


class Critic(nn.Module):
    def __init__(self, s_dim):
        """
        论文中提到，这是一个state-value方法，所以只输入s，返回值为1个Q值
        我们这里和actor的层数一致，效果不好再改
        :param s_dim:
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, obs):
        # 如果输入的是数组，就将其转换成tensor
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        tem1 = F.relu(self.fc1(obs))
        tem2 = F.relu(self.fc2(tem1))
        res = self.out(tem2)

        return res

class DDPG():
    def __init__(self, actorNet, criticNet, env, **hyperparameters):
        """

        :param actorNet: actor网络架构
        :param criticNet: critic网络架构
        :param env: 环境
        :param hyperparameters: 参数
        """
        # 4个网络搭建
        self.env = env
        # 初始化参数
        self._init_hyperparameters(hyperparameters)
        self.obs_dim = env.numberOfDevice * 4 + env.numberOfServer * 2
        self.dis_act_dim = 2 ** env.numberOfDevice
        self.con_act_dim = env.numberOfDevice * (2 ** env.numberOfDevice)  # 每个设备的卸载率+计算效率

        # 网络搭建
        self.actor_eval = actorNet(self.obs_dim, self.dis_act_dim, self.con_act_dim, self.con_act_dim)
        self.actor_target = actorNet(self.obs_dim, self.dis_act_dim, self.con_act_dim, self.con_act_dim)
        self.critic_eval = criticNet(self.obs_dim)
        self.critic_target = criticNet(self.obs_dim)

        # 存储池搭建
        self.memory = np.zeros((self.memory_size, self.obs_dim * 2 + 1), dtype=np.float32)
        self.point = 0  # 存储池中存储的当前位置

        # optimizer搭建
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=self.LR_A)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=self.LR_C)

        # 损失函数设计
        self.critic_loss = nn.MSELoss()

        # 离散与连续分布参数设计
        self.cov_var = torch.full(size=(self.con_act_dim,), fill_value=0.1)  # 原本是0.5
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self):
        print(f"Learning... Running {self.ep_steps} timesteps per episode, ", end='')
        all_dis_act = []
        for ep in range(self.episodes):
            self.env.episode = ep
            self.env.reset()
            for step in range(self.ep_steps):
                self.env.clock = step + 1
                s = self.env.getEnvState()
                # 先将动作存储进memory中
                dis_pars, con_pars, f_pars = self.actor_eval(s)

                # 随机选择离散动作及其log概率(这里使用torch.multinomial为网上的方法)
                dis_dist = torch.distributions.Categorical(dis_pars)
                dis_action = dis_dist.sample().detach()
                # 得到连续动作的分布模型(这里我们将多个模型选择的连续分布合成一个多元正态分布，因为这样好求解)
                dist = MultivariateNormal(con_pars, self.cov_mat)
                # 根据模型选出连续动作（根据概率随机选择）
                con_action = dist.sample()
                con_action = torch.clamp(con_action, 0, 1).detach()
                # 根据模型选出对应连续动作的log概率
                f_dist = MultivariateNormal(f_pars, self.cov_mat)
                f_action = f_dist.sample()
                f_action = torch.clamp(f_action, 0.01, 1).detach()

                all_dis_act.append(dis_action)

                # 得到reward和下一个状态
                self.env.offload(makeOffloadDecision, dis_action, con_action, f_action)
                r = self.env.getEnvReward()
                self.env.stepIntoNextState()
                s_ = self.env.getEnvState()

                self.store(s, r, s_)

                # 如果memory存储满了，则可以正式开始训练
                if self.point > self.memory_size:
                    # 参数转移
                    for x in self.actor_target.state_dict().keys():
                        eval('self.actor_target.' + x + '.data.mul_((1-self.TAU))')
                        eval('self.actor_target.' + x + '.data.add_(self.TAU*self.actor_eval.' + x + '.data)')
                    for x in self.critic_target.state_dict().keys():
                        eval('self.critic_target.' + x + '.data.mul_((1-self.TAU))')
                        eval('self.critic_target.' + x + '.data.add_(self.TAU*self.critic_eval.' + x + '.data)')
                    # 随机拿取一个batch_size的数据
                    indices = np.random.choice(self.memory_size, size=self.batch_size)
                    batch_trans = self.memory[indices, :]
                    # 对数据进行分类
                    batch_s = torch.FloatTensor(batch_trans[:, :self.obs_dim])

                    batch_r = torch.FloatTensor(batch_trans[:, -self.obs_dim - 1: -self.obs_dim])
                    batch_s_ = torch.FloatTensor(batch_trans[:, -self.obs_dim:])
                    # 训练actor
                    q = self.critic_eval(batch_s)
                    actor_loss = -torch.mean(q)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    # 训练critic
                    q_tmp = self.critic_target(batch_s_)
                    q_target = batch_r+self.GAMMA*q_tmp
                    q_eval = self.critic_eval(batch_s)
                    td_error = self.critic_loss(q_target, q_eval)
                    self.critic_optimizer.zero_grad()
                    td_error.backward()
                    self.critic_optimizer.step()

        self.env.outputMetric()
        total = self.env.episodes * self.env.T * self.env.numberOfDevice
        print(
            "finished!!! failure rate = %f,and error rate = %f" % (
            self.env.failures / total, self.env.errors / (total * 2)))
        analysis.draw(self.env.envDir, self.env.algorithmDir)
        np.savetxt('out_dis.txt', all_dis_act, fmt="%f", delimiter=',')


    def store(self, s, r, s_):
        """

        :param s: 状态
        :param a: 动作
        :param r: 回报
        :param s_: 下一个状态
        :return: none
        """
        transition = np.hstack((s, [r], s_))  # 将这个多维数组按顺序合并成1为数组
        index = self.point % self.memory_size
        self.memory[index, :] = transition
        self.point += 1

    def _init_hyperparameters(self, hyperparameters):
        """

        :param hyperparameters: 参数字典
        :return: none
        """
        self.batch_size = 8
        self.episodes = self.env.episodes
        self.ep_steps = 100  # 每次训练100个batch_size
        self.memory_size = 10000
        self.LR_A = 0.001
        self.LR_C = 0.002
        self.TAU = 0.01  # 每次两个网络间参数转移的衰减程度
        self.GAMMA = 0.9  # 未来的r的比例

        for param, value in hyperparameters.items():
            exec('self.' + param + '=' + str(value))
