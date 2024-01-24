import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
import copy
import numpy as np
from torch.optim import Adam
import analysis
import random


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


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
            device.offload(1, con_action[dis_action * 6 + i], f_action[dis_action * 6 + i])
        else:
            device.offload(0, 0, 0)
        i += 1


class Actor(nn.Module):

    def __init__(self, s_dim, out_c, out_d, wt_dim):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(s_dim + wt_dim, 128)
        self.linear2 = nn.Linear(128, 128)

        self.pi_d = nn.Linear(128, out_d)
        self.mean_linear = nn.Linear(128, out_c)
        self.log_std_linear = nn.Linear(128, out_c)

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init_)

    def forward(self, state, w):
        # 如果输入的是数组，就将其转换成tensor
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        state_comp = torch.cat((state, w), dim=1)
        x = F.relu(self.linear1(state_comp))
        x = F.relu(self.linear2(x))

        pi_d = self.pi_d(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=-20, max=2)
        return pi_d, mean, log_std

    def sample(self, state, w):
        # for each state in the mini-batch, get its mean and std
        pi_d, mean, log_std = self.forward(state, w)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        # restrict the outputs
        action_c = torch.sigmoid(x_t)
        log_prob_c = normal.log_prob(x_t)
        log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)

        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

    def to(self, device):
        return super(Actor, self).to(device)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, out_c, out_d, wt_dim):
        super(QNetwork, self).__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(num_inputs + out_c + wt_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_d)
        )

        self.Q2 = copy.deepcopy(self.Q1)
        self.rwd_dim = wt_dim
        self.apply(weights_init_)

    def forward(self, state, action, w):
        x = torch.cat([state, action, w], 1)
        x1 = self.Q1(x)
        x2 = self.Q2(x)
        return x1, x2


class CAPQL():
    def __init__(self, env, all_weights):

        self.env = env

        self.batch_size = 32
        self.episode_number = 2000
        self.ep_steps = 100  # 每次训练100个batch_size
        self.memory_size = 10000
        self.lr = 0.002
        self.tau = 0.01  # 每次两个网络间参数转移的衰减程度
        self.gamma = 0.9  # 未来的r的比例

        self.state_dim = env.numberOfDevice * 4 + env.numberOfServer * 2  # bandwidth + computing capacity
        self.dis_act_dim = 2 ** env.numberOfDevice
        self.con_act_dim = env.numberOfDevice * (2 ** env.numberOfDevice)  # 每个设备的卸载率+计算效率

        # define TWO Q networks for training
        self.critic = QNetwork(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
        self.actor = Actor(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.actor.parameters(), lr=self.lr)

        self.memory = DiverseExperienceReplay(int(self.memory_size / env.taskNumber))  # 队列，最大值是5000
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.all_weights = all_weights
        self.current_weight = None  # 当前权重
        self.weight_history = []  # 过往权重集
        self.weight_encounter = []
        self.encountered_weight = None  # 过往权重
        self.encountered_weight_num = 1  # 从过往.的数量

        self.each_episode_weight = []
        self.each_episode_DR = []  # 每个回合的累计折扣奖励
        self.each_episode_history_DR = []  # 保存每个回合的累计折扣奖励
        self.each_episode_DR_actual_01 = []
        self.each_episode_loss = []  # 保存每个回合的loss
        self.weight_opt_DR = {}  # 保存每个权重下的历史最优累计折扣奖励， 索引是权重向量元组，即(0.1, 0.5, 0.4)
        for weight in all_weights:
            self.weight_opt_DR[tuple(weight)] = []

        self.weight_opt_fitness = {}  # 保存所有回合下的适应度向量，索引是权重, 其值为[适应度向量，权重*适应度向量]
        for weight in all_weights:
            self.weight_opt_fitness[tuple(weight)] = np.zeros(4)

        self.weight_opt_actions_fitness = {}
        for weight in all_weights:
            self.weight_opt_actions_fitness[tuple(weight)] = []

        self.weight_opt_actions_DR = {}
        for weight in all_weights:
            self.weight_opt_actions_DR[tuple(weight)] = []

    def select_action(self, state, w):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        w = torch.FloatTensor(w).to(self.device).unsqueeze(0)
        action_c, action_d, log_prob_c, log_prob_d, prob_d = self.actor.sample(state, w)

        action_c = torch.clamp(action_c, 0, 1).detach()
        action_d = action_d.detach()
        log_prob_c = log_prob_c.detach()
        log_prob_d = log_prob_d.detach()
        prob_d = prob_d.detach()

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

    def to_offload_action(self, action_c, action_d):
        action_c_list = action_c.squeeze().tolist()  # Squeeze 用于移除多余的维度
        ac1 = action_c_list[0]
        ac2 = action_c_list[1]
        ad = action_d.squeeze().item()  # 假设 action_d 是一个标量或形状为 [1] 的张量
        return [ad, ac1, ac2]

    def remember(self, trace, time_step, current_weight, state, action, next_state, reward):
        data = (state, action, next_state, reward)
        trace.transitions.append(data)  # 任务调度完成之前，每次只在trace中记录本条轨迹中的各个状态转移
        if time_step == self.env.taskNumber:  # 任务调度完后，记录本次轨迹，包括本次的所有状态转移、本轨迹的权重、适合度等
            self.memory.update_buffer(trace, current_weight)

    def run(self):
        step = 0
        one_episode_rewards = []
        running_reward = None
        for eps_idx in range(self.episode_number):
            total_loss = []
            # 每个回合 变换权重
            self.current_weight = self.set_weight_encoun_sque(eps_idx)
            current_state = self.env.reset(self.current_weight)
            one_episode_dis_actions = []
            trace = Trace(eps_idx, self.current_weight)  # yy 实例化trace
            for time_step in range(1, self.env.taskNumber + 1):
                step += 1
                action_c, action_d, log_prob_c, log_prob_d, prob_d = self.select_action(current_state,
                                                                                        self.current_weight)
                one_episode_dis_actions.append(action_c)
                [ad, ac1, ac2] = self.to_offload_action(action_c, action_d)
                self.env.offload(makeOffloadDecision, ad, ac1, ac2)
                reward = self.env.getEnvReward()
                self.env.stepIntoNextState()
                next_state = self.env.getEnvState()

                self.remember(trace, time_step, self.current_weight, current_state, [ad, ac1, ac2], next_state, reward)
                one_episode_rewards.append(np.array(reward))

                # training
                if len(self.memory) > self.batch_size:
                    state_wt, action_wt, next_state_wt, wt_batch, reward_wt, \
                        state_wh, action_wh, next_state_wh, wh_batch, reward_wh = self.memory_sample(eps_idx, self.current_weight)

                    # compute next_q_value target
                    qf_loss_wt = self.get_critic_loss(state_wt, action_wt, next_state_wt, wt_batch, reward_wt)
                    qf_loss_wh = self.get_critic_loss(state_wh, action_wh, next_state_wh, wh_batch, reward_wh)
                    qf_loss = (qf_loss_wt + qf_loss_wh) / 2
                    self.critic_optim.zero_grad()
                    qf_loss.backward()
                    self.critic_optim.step()

                    # train the policy network
                    actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.actor.sample(state_wt, wt_batch)
                    qf1_pi_wt, qf2_pi_wt = self.critic(state_wt, actions_c, wt_batch)
                    qf1_pi_wh, qf2_pi_wh = self.critic(state_wh, actions_c, wh_batch)

                    min_qf_pi_wt = torch.min(qf1_pi_wt, qf2_pi_wt)
                    min_qf_pi_wt = (min_qf_pi_wt * wt_batch).sum(dim=-1, keepdim=True)
                    min_qf_pi_wh = torch.min(qf1_pi_wh, qf2_pi_wh)
                    min_qf_pi_wh = (min_qf_pi_wh * wh_batch).sum(dim=-1, keepdim=True)

                    min_qf_pi = (min_qf_pi_wt + min_qf_pi_wh) / 2

                    policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                    policy_loss_c = (prob_d * (self.alpha_c * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                    policy_loss = policy_loss_d + policy_loss_c

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    # sync the Q networks
                    if eps_idx % self.target_update_interval == 0:
                        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

            disc_actual = np.sum(np.array([(self.discount ** i) * r for i, r in enumerate(one_episode_rewards)]),
                                 axis=0)
            disc_actual_01 = np.sum(np.array([r for i, r in enumerate(one_episode_rewards)]),
                             axis=0)
            total_reward_01 = np.dot(disc_actual_01, self.current_weight)
            total_reward = np.dot(disc_actual, self.current_weight)
            running_reward = total_reward if running_reward == None else running_reward * 0.99 + total_reward * 0.01

            self.each_episode_DR.append(total_reward)  # 索引为回合数，其元素为当前回合的权重和temp
            self.each_episode_history_DR.append(running_reward)  #历史回合回报+当前回合总回报更新历史回合回报
            self.each_episode_DR_actual_01.append(total_reward_01)
            self.each_episode_loss.append(sum(total_loss))
            self.weight_opt_DR[tuple(self.current_weight)].append(total_reward)
            self.weight_opt_actions_DR[tuple(self.current_weight)].append(one_episode_dis_actions)

            get_w_opt_fitness(tuple(self.current_weight), self.weight_opt_fitness, self.env.current_schedule.fitness)
            one_episode_rewards = []
            print('Episode: ', eps_idx, ' | reward: ', disc_actual,' | total_reward: ', total_reward, ' | one_episode_actions: ', one_episode_dis_actions,
              ' | weight: ', self.current_weight)



        self.env.outputMetric()
        total = self.env.episodes * self.env.T * self.env.numberOfDevice
        print(
            "finished!!! failure rate = %f,and error rate = %f" % (
                self.env.failures / total, self.env.errors / (total * 2)))
        analysis.draw(self.env.envDir, self.env.algorithmDir)
        np.savetxt('out_dis.txt', all_dis_act, fmt="%f", delimiter=',')

    def sample(self, eps_idx, batch_size):
        tmp = []
        for trace in self.main_memory:
            for trans in trace.transitions:
                tmp.append(trans)
        for trace in self.secd_memory:
            for trans in trace.transitions:
                tmp.append(trans)
        return random.sample(tmp, batch_size)

    def to_torch_action(actions, device):
        ad = torch.Tensor(actions[:, 0]).int().to(device)
        ac = torch.Tensor(actions[:, 1:]).to(device)
        return ac, ad

    def get_critic_loss(self, state_batch, action_batch, next_state, w_batch, reward):
        with torch.no_grad():
            next_s_actions_c, next_s_actions_d, next_s_log_pi_c, next_s_log_pi_d, next_s_prob_d = self.actor.sample(next_state, w_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_s_actions_c, w_batch)
            min_qf_next_target_wt = next_s_prob_d * (torch.min(qf1_next_target, qf2_next_target) - self.alpha_c * next_s_prob_d * next_s_log_pi_c - self.alpha_d * next_s_log_pi_d)
            next_q_value = reward + self.gamma * (min_qf_next_target_wt.sum(1)).view(-1)

        s_actions_c, s_actions_d = self.to_torch_action(action_batch)
        qf1, qf2 = self.critic.forward(state_batch, s_actions_c, w_batch)
        qf1 = qf1.gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
        qf2 = qf2.gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = (qf1_loss + qf2_loss) / 2
        return qf_loss


    def memory_sample(self, eps_idx, weight):

        # 随机从队列中取出一个batch大小的数据，data是batchsize个transition
        data = self.memory.sample(eps_idx, self.batch_size)
        state = [d[0] for d in data]
        next_state = [d[2] for d in data]
        state = np.array(state)
        next_state = np.array(next_state)

        '''------------ 针对当前权重 ------------'''
        wt_batch = np.repeat([weight], self.batch_size, axis=0)  # 重复最近一次的权重，32次，形成32*3矩阵
        tt = [state, wt_batch]
        with tf.device('/gpu:0'):
            y_wt = self.Q_network(tt).numpy()  # 针对当前权重的一批state放入网络,Q(s,a)
            Q1_wt = self.target_Q_network([next_state, wt_batch]).numpy()  # TQ(s_,a)

            # double DQN for MO
            Q2_wt = self.Q_network([next_state, wt_batch]).numpy()  # Q(s_,a)
        next_action_wt = np.argmax(np.dot(Q2_wt, weight), axis=1)  # 将每个动作的Q值向量与权重相乘,挑选的a
        '''------------ 针对当前权重 ------------'''

        '''------------ 针对过往权重 ------------'''
        np.random.seed(eps_idx)
        max_index = len(self.weight_history)
        idx = np.random.randint(max_index, size=self.batch_size)
        wj_batch = np.array(self.weight_history)[idx]

        with tf.device('/gpu:0'):
            y_wj = self.Q_network([state, wj_batch]).numpy()
            Q1_wj = self.target_Q_network([next_state, wj_batch]).numpy()

            Q2_wj = self.Q_network([next_state, wj_batch]).numpy()
        next_action_wj = [np.argmax(np.dot(Q2_wj[i], wj_batch[i])) for i in range(self.batch_size)]
        '''------------ 针对过往权重 ------------'''

        for i, (_, a, _, r, done) in enumerate(data):
            if done:
                target_yt = r
                target_yj = r
            else:
                target_yt = np.add(r, self.discount * Q1_wt[i][next_action_wt[i]])  # 计算的r+maxQ(s_,a)
                target_yj = np.add(r, self.discount * Q1_wj[i][next_action_wj[i]])

            target_yt = np.array(target_yt, dtype='float32')
            y_wt[i][a] = target_yt

            target_yj = np.array(target_yj, dtype='float32')
            y_wj[i][a] = target_yj

        return state, wt_batch, y_wt, state, wj_batch, y_wj


class Trace:
    def __init__(self, eps_idx, weight):
        self.eps_idx = eps_idx  # Index of episode
        self.weight = weight  # 该条轨迹对应的权重
        self.transitions = []  # All transitions
        self.fitness = []  # 该条轨迹对应的适应度
        self.weighted_fitness = None  # 加权
        self.signature = []  # 累积折扣奖励向量
        self.crd_distance = None  # 该条轨迹的拥挤距离
        self.temp_signature = None  # 临时标签，用于对缓冲区中的轨迹进行排序
