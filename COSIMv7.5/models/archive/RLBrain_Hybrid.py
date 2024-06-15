import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
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

    userlist = randombin(env.numberOfDevice, dis_action)
    i = 0
    for device in env.devices:
        if userlist[device.id - 1] == 1:
            device.offload(1, con_action[dis_action*6+i], f_action[dis_action*6+i])
        else:
            device.offload(0, 0, 0)
        i += 1


class ActorNet(nn.Module):
    def __init__(self, s_dim, dis_a_dim, con_a_dim, f_a_dim):
        """

        :param s_dim: 输入的环境维度
        :param dis_a_dim: 输出的离散动作维度
        :param con_a_dim: 输出的连续动作维度
        """
        super(ActorNet, self).__init__()
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


class CriticNet(nn.Module):
    def __init__(self, s_dim):
        """
        论文中提到，这是一个state-value方法，所以只输入s，返回值为1个Q值
        我们这里和actor的层数一致，效果不好再改
        :param s_dim:
        """
        super(CriticNet, self).__init__()
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


class Hybrid():
    def __init__(self, actor, critic, env):
        # 根据传入的hyperparameters来确定PPO框架中的一些参数
        self._init_hyperparameters()

        # 配置环境，状态空间和动作空间
        self.env = env
        self.obs_dim = env.numberOfDevice * 4 + env.numberOfServer * 2
        self.dis_act_dim = 2 ** env.numberOfDevice
        # self.con_act_dim = env.numberOfDevice
        self.con_act_dim = env.numberOfDevice * (2 ** env.numberOfDevice)  # 每个设备的卸载率+计算效率

        # 配置actor与critic
        self.actor = actor(self.obs_dim, self.dis_act_dim, self.con_act_dim, self.con_act_dim)
        self.critic = critic(self.obs_dim)

        # 配置optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # 配置查询con-actor中con-actions的协方差矩阵 (在计算速率中也是用的是这个)
        self.cov_var = torch.full(size=(self.con_act_dim,), fill_value=0.1)  # 原本是0.5
        self.cov_mat = torch.diag(self.cov_var)

        # 存储loss，查看是否收敛
        self.act_losses = []
        self.cri_losses = []

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        # 一共训练total_timesteps轮
        all_dis_act = []

        for episode in range(self.env.episodes):
            self.env.logger.info('episode:%d - start' % episode)
            self.env.episode = episode
            # 首先在老的actor上跑一个batchsize，得到对应的obs，acts，log_prob以及我们训练critic所需要的的Q现实
            batch_obs, batch_dis_acts, batch_con_acts, batch_f_acts, batch_dis_log_probs, \
            batch_con_log_probs, batch_f_log_probs, batch_rtgs = self.rollout()

            all_dis_act.append(batch_dis_acts.numpy().tolist())

            # 真正开始学习
            # 首先要求得actor损失函数中的A_k，A_k = Q现实-Q估计
            V = self.critic(batch_obs).squeeze()
            A_k = batch_rtgs - V.detach()
            # 这里我们将A_k进行一次规范化，是为了稳定与加快收敛，并且使其在解决某些问题上更加稳定
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # 每一轮要训练n_updates_per_iteration次
            for i in range(self.n_updates_per_iteration):
                # 计算最新的V
                V = self.critic(batch_obs).squeeze()
                # 计算新的log_probs
                current_dis_pars, con_pars, f_pars = self.actor(batch_obs)

                # 把old-act带入新的离散动作分布中，得到最新的概率
                dis_dist = torch.distributions.Categorical(current_dis_pars)
                current_dis_log_prob = dis_dist.log_prob(batch_dis_acts)
                # 计算actor损失函数中的ratios
                # 这里之所以用log代替概率进行计算，最后在进行e的指数运算还原概率，详见笔记
                dis_ratios = torch.exp(current_dis_log_prob - batch_dis_log_probs)
                # 计算dis的损失函数
                dis_surr1 = dis_ratios * A_k
                dis_surr2 = torch.clamp(dis_ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_dis_loss = (-torch.min(dis_surr1, dis_surr2)).mean()

                # 计算连续参数
                dist = MultivariateNormal(con_pars, self.cov_mat)
                current_con_log_prob = dist.log_prob(batch_con_acts)
                con_ratios = torch.exp(current_con_log_prob - batch_con_log_probs)
                con_surr1 = con_ratios * A_k
                con_surr2 = torch.clamp(con_ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_con_loss = (-torch.min(con_surr1, con_surr2)).mean()

                # 计算计算效率参数
                f_dist = MultivariateNormal(f_pars, self.cov_mat)
                current_f_log_prob = f_dist.log_prob(batch_f_acts)
                f_ratios = torch.exp(current_f_log_prob - batch_f_log_probs)
                f_surr1 = f_ratios * A_k
                f_surr2 = torch.clamp(f_ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_f_loss = (-torch.min(f_surr1, f_surr2)).mean()

                # 计算总的损失函数
                actor_loss = actor_dis_loss + actor_con_loss + actor_f_loss
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()
                self.act_losses.append(actor_loss.detach())

                # 计算critic的损失函数
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                self.cri_losses.append(critic_loss.detach())
            # 每一个batchsize存储一次权重
            torch.save(self.actor.state_dict(), '../ppo_actor_f.pth')
            torch.save(self.critic.state_dict(), '../ppo_critic_f.pth')

        # collect metrics
        self.env.outputMetric()
        total = self.env.episodes * self.env.T * self.env.numberOfDevice
        print(
            "finished!!! failure rate = %f,and error rate = %f" % (self.env.failures / total, self.env.errors / (total * 2)))
        analysis.draw(self.env.envDir, self.env.algorithmDir)
        self.show()
        np.savetxt('out_dis.txt', all_dis_act, fmt="%f", delimiter=',')


    def rollout(self):
        batch_obs = []
        batch_dis_acts = []
        batch_con_acts = []
        batch_f_acts = []
        batch_dis_log_probs = []
        batch_con_log_probs = []
        batch_f_log_probs = []
        batch_rews = []

        t = 0

        while t < self.timesteps_per_batch:
            self.env.reset()
            tem_rew = []  # 这里之所以要每个episode的reward存成一个数组是因为在计算r的时候需要考虑后来的r，所以每一个episode的reward要放在一起计算，所以单独村一个数组。
            for i in range(self.max_timesteps_per_episode):
                self.env.clock = i+1
                t += 1
                # 开始
                obs = self.env.getEnvState()
                batch_obs.append(obs)
                dis_action, con_action, f_action, dis_log_prob, con_log_prob, f_log_prob = self.get_action(obs)
                self.env.offload(makeOffloadDecision, dis_action, con_action, f_action)
                rew = self.env.getEnvReward()
                self.env.stepIntoNextState()
                batch_dis_acts.append(dis_action)
                batch_con_acts.append(con_action.numpy())
                batch_f_acts.append(f_action.numpy())
                batch_dis_log_probs.append(dis_log_prob)
                batch_con_log_probs.append(con_log_prob)
                batch_f_log_probs.append(f_log_prob)

                tem_rew.append(rew)
            batch_rews.append(tem_rew)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)  # 这里batch_obs和rtgs需要参与后面的batch运算所以需要转换成tensor
        batch_rtgs = self.compute_rtgs(batch_rews)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        batch_dis_acts = torch.tensor(batch_dis_acts, dtype=torch.int64)
        batch_con_acts = torch.tensor(batch_con_acts, dtype=torch.float)
        batch_f_acts = torch.tensor(batch_f_acts, dtype=torch.float)
        batch_dis_log_probs = torch.tensor(batch_dis_log_probs, dtype=torch.float)
        batch_con_log_probs = torch.tensor(batch_con_log_probs, dtype=torch.float)
        batch_f_log_probs = torch.tensor(batch_f_log_probs, dtype=torch.float)

        return batch_obs, batch_dis_acts, batch_con_acts, batch_f_acts, batch_dis_log_probs, batch_con_log_probs, batch_f_log_probs, batch_rtgs

    def compute_rtgs(self, batch_rews):
        """

        :param batch_rews:输入的关于batchsize中每个episode中的reward  Shape: (number of episodes, number of timesteps per episode)
        :return: batch_rtgs：计算完毕的reward Shape: (number of timesteps in batch)
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        return batch_rtgs

    def get_action(self, obs):
        # 得到动作
        dis_pars, con_pars, f_pars = self.actor(obs)
        # 随机选择离散动作及其log概率(这里使用torch.multinomial为网上的方法)
        dis_dist = torch.distributions.Categorical(dis_pars)
        dis_action = dis_dist.sample().detach()
        dis_log_prob = dis_dist.log_prob(dis_action).detach()
        # 得到连续动作的分布模型(这里我们将多个模型选择的连续分布合成一个多元正态分布，因为这样好求解)
        dist = MultivariateNormal(con_pars, self.cov_mat)
        # 根据模型选出连续动作（根据概率随机选择）
        con_action = dist.sample()
        con_action = torch.clamp(con_action, 0, 1).detach()
        # 根据模型选出对应连续动作的log概率
        con_log_prob = dist.log_prob(con_action).detach()

        f_dist = MultivariateNormal(f_pars, self.cov_mat)
        f_action = f_dist.sample()
        f_action = torch.clamp(f_action, 0.01, 1).detach()
        f_log_prob = f_dist.log_prob(f_action).detach()

        return dis_action, con_action, f_action, dis_log_prob, con_log_prob, f_log_prob

    def _init_hyperparameters(self):
        # 定义与网络有关的参数
        self.timesteps_per_batch = 800  # 一个batchsize中含有多少个timestep
        self.max_timesteps_per_episode = 100  # 一个episode中含有多少个timestep
        self.n_updates_per_iteration = 5  # 每次迭代中newNetwork更新几次
        self.lr = 0.00005
        self.gamma = 0.95
        self.clip = 0.2

        # 定义与网络无关的参数
        self.save_freq = 10  # 10次迭代存储一次
        self.seed = None  # 保存seed，最后用于生成随机数来增强模型的抖动

        # 设置seed
        if self.seed != None:
            # 检查seed是否有效
            assert (type(self.seed) == int)

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")


    def show(self):
        print('act_loss:')
        for act_loss in self.act_losses:
            print(act_loss)
        print('cri_loss:')
        for cri_loss in self.cri_losses:
            print(cri_loss)