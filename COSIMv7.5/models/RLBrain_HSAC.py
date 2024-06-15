import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from replay_memory import ReplayMemory
from utils import Weight_Sampler_pos

input_shape = 59
out_c = 5
out_d = 3
LOG_STD_MAX = 0.0
LOG_STD_MIN = -3.0


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, s_dim, out_c, out_d, wt_dim):  # out_c=2*(1+device_num)
        super(Actor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(s_dim + wt_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)

        self.pi_d = nn.Linear(64, out_d)
        self.mean = nn.Linear(64, out_c)
        self.logstd = nn.Linear(64, out_c)

        self.apply(weights_init_)
        self.to(self.device)

    def forward(self, state, w):
        state_comp = torch.cat((state, w), dim=1)
        mask = torch.isnan(state_comp).any(dim=1)
        state_comp = state_comp[~mask]
        x = F.leaky_relu(self.fc1(state_comp), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)

        pi_d = self.pi_d(x)
        mean = torch.tanh(self.mean(x))
        log_std = torch.tanh(self.logstd(x))

        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return pi_d, mean, log_std

    def get_action(self, state, w, num_device, num_server):

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)
        w = w.to(self.device)
        pi_d, mean, log_std = self.forward(state, w)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        action_c = torch.sigmoid(x_t)
        log_prob_c_full = normal.log_prob(x_t)
        log_prob_c_full -= torch.log(1.0 - action_c.pow(2) + 1e-6)

        # log_prob_c = torch.cat(
        #     [log_prob_c_full[:, :2].sum(1, keepdim=True), log_prob_c_full[:, :4].sum(1, keepdim=True),
        #      log_prob_c_full[:, :6].sum(1, keepdim=True)], 1)
        log_prob_c = torch.cat([log_prob_c_full[:, :2 * (i + 1)].sum(dim=1, keepdim=True) for i in range(log_prob_c_full.size(1) // 2)], dim=1)

        actions_d = []
        probs_d = []
        log_probs_d = []
        selected_actions_c = []
        for i in range(num_device):
            start_idx = i * (num_server + 1)
            end_idx = start_idx + num_server + 1

            dist = Categorical(logits=pi_d[:, start_idx:end_idx])
            action_d = dist.sample()
            if action_d.dim() == 1:
                action_d = action_d.unsqueeze(-1)
            prob_d = dist.probs
            log_prob_d = torch.log(prob_d + 1e-6)

            actions_d.append(action_d)
            probs_d.append(prob_d)
            log_probs_d.append(log_prob_d)

            # 从全部的action_c中选择对应的action_d的action_c
            action_c_start_idxs = i * (1 + num_server) * 2 + action_d * 2
            action_c_end_idxs = action_c_start_idxs + 2
            temp_c = torch.stack([action_c[i, start:end] for i, (start, end) in enumerate(zip(action_c_start_idxs, action_c_end_idxs))])
            selected_actions_c.append(temp_c)

        return torch.cat(selected_actions_c, dim=1), torch.cat(actions_d, dim=1), log_prob_c, torch.cat(log_probs_d, dim=1), torch.cat(probs_d, dim=1)

    def to(self, device):
        return super(Actor, self).to(device)


class QNetwork(nn.Module):
    def __init__(self, state_dim, numberOfServer, numberOfDevice, wt_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numberOfServer = numberOfServer
        self.numberOfDevice = numberOfDevice
        # TODO: 不确定out_c应该是全部的还是被选中的out_c; 不确定是否对out_d进行onehot编码
        # 全部的out_c和onehot的out_d
        # out_d = (numberOfServer + 1) * numberOfDevice  # 对于一个device 2个server来说离散动作有0，1，2三种
        # out_c = (numberOfServer + 1) * 2  # 对于三个离散动作来说，有6个连续动作
        # 选中的out_c和不onthot的out_d
        out_d = numberOfDevice
        out_c = numberOfDevice * 2
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + out_c + out_d + wt_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, wt_dim)

        self.rwd_dim = wt_dim
        self.apply(weights_init_)
        self.to(self.device)

    def forward(self, state, action_c, action_d, w):
        state = torch.FloatTensor(state).to(self.device) if not torch.is_tensor(state) else state
        action_c = torch.FloatTensor(action_c).to(self.device) if not torch.is_tensor(action_c) else action_c
        action_d = torch.Tensor(action_d).to(self.device) if not torch.is_tensor(action_d) else action_d.to(torch.int64)
        w = torch.FloatTensor(w).to(self.device) if not torch.is_tensor(w) else w

        x = torch.cat([state, action_d, action_c, w], 1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.leaky_relu(self.fc3(x), 0.01)
        x = self.fc4(x)
        return x


class HSAC:
    def __init__(self, env):
        self.env = env
        self.batch_size = 32
        self.episode_number = self.env.episodes
        self.tau = 0.01  # 每次两个网络间参数转移的衰减程度
        self.gamma = 0.80  # 未来的r的比例
        self.wt_dim = 2
        self.alpha_c = 0.2
        self.alpha_d = 0.2

        self.act_losses = []
        self.cri_losses = []

        self.state_dim = self.env.numberOfDevice * 3 + self.env.numberOfServer * 2  # bandwidth + computing capacity
        self.dis_act_dim = (self.env.numberOfServer + 1) * self.env.numberOfDevice
        self.con_act_dim = self.dis_act_dim * 2  # 每个设备对每个server的卸载率+计算效率

        self.qf1 = QNetwork(self.state_dim, self.env.numberOfServer, self.env.numberOfDevice, self.wt_dim)
        self.qf2 = QNetwork(self.state_dim, self.env.numberOfServer, self.env.numberOfDevice, self.wt_dim)
        self.qf1_target = QNetwork(self.state_dim, self.env.numberOfServer, self.env.numberOfDevice, self.wt_dim)
        self.qf2_target = QNetwork(self.state_dim, self.env.numberOfServer, self.env.numberOfDevice, self.wt_dim)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.actor = Actor(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)

        self.critic_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=2e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=0.0006)

        self.memory = ReplayMemory(1000000000, 123456)
        self.weight_sampler = Weight_Sampler_pos(2)

        self.current_weight = None  # 当前权重
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        for eps_idx in range(self.episode_number):
            self.current_weight = self.weight_sampler.sample(1)  # 每个回合 变换权重
            self.env.episode = eps_idx
            self.env.reset()
            self.env.setUp()
            time_step = 0
            total_reward = 0
            while not self.env.isDAGsDone():
                time_step += 1
                current_state, old_dag_status = self.env.getEnvState()
                action_c, action_d, _, _, _ = self.actor.get_action(current_state, self.current_weight,
                                                                    self.env.numberOfDevice,
                                                                    self.env.numberOfServer)
                self.env.offload(time_step, action_d, action_c)
                reward = self.env.getEnvReward(self.current_weight, old_dag_status, action_d, action_c)
                total_reward = total_reward + reward
                self.env.stepIntoNextState()
                next_state, _ = self.env.getEnvState()
                if not self.env.isDAGsDone():
                    self.memory.push(current_state, action_c, action_d, self.current_weight, reward, next_state, self.env.isDAGsDone())
                if len(self.memory) > 3 * self.batch_size:
                    state_batch, s_actions_c, s_actions_d, w_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
                    with ((torch.no_grad())):
                        next_state_actions_c, next_state_actions_d, next_state_log_pi_c, next_state_log_pi_d, next_state_prob_d = self.actor.get_action(
                            next_state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                        qf1_next_target = self.qf1_target.forward(next_state_batch, next_state_actions_c, next_state_actions_d, w_batch)
                        qf2_next_target = self.qf2_target.forward(next_state_batch, next_state_actions_c, next_state_actions_d, w_batch)

                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                        - self.alpha_c * (next_state_prob_d * next_state_log_pi_c).sum(dim=-1, keepdim=True) / next_state_log_pi_c.shape[1]
                        - self.alpha_d * next_state_log_pi_d.sum(dim=-1, keepdim=True) / next_state_log_pi_c.shape[1]
                        next_q_value = torch.Tensor(reward_batch).to(self.device) + (1 - torch.Tensor(done_batch).to(self.device)) * self.gamma * min_qf_next_target

                    qf1_a_values = self.qf1.forward(state_batch, s_actions_c, s_actions_d, w_batch)
                    qf2_a_values = self.qf2.forward(state_batch, s_actions_c, s_actions_d, w_batch)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = (qf1_loss + qf2_loss) / 2

                    self.critic_optimizer.zero_grad()
                    qf_loss.backward()
                    nn.utils.clip_grad_norm_(list(self.qf1.parameters()) + list(self.qf2.parameters()), 0.5)
                    self.critic_optimizer.step()
                    self.cri_losses.append(qf_loss.detach())

                    if eps_idx % 1 == 0:  # TD 3 Delayed update support
                        for _ in range(1):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                            actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.actor.get_action(state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                            qf1_pi = self.qf1.forward(state_batch, actions_c, actions_d, w_batch)
                            qf2_pi = self.qf2.forward(state_batch, actions_c, actions_d, w_batch)
                            min_qf_pi = (torch.min(qf1_pi, qf2_pi) * torch.FloatTensor(w_batch).to(self.device)).sum(dim=-1, keepdim=True)

                            policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi)).sum(1).mean()
                            policy_loss_c = (prob_d * (self.alpha_c * prob_d * log_pi_c - min_qf_pi)).sum(1).mean()
                            policy_loss = policy_loss_d + policy_loss_c

                            self.actor_optimizer.zero_grad()
                            policy_loss.backward()
                            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                            self.actor_optimizer.step()
                            self.act_losses.append(policy_loss.detach())

                # update the target network
                    if eps_idx % 1 == 0:
                        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                if self.env.isDAGsDone():
                    logging.info("dags are done")
                    break

            for device in self.env.devices:
                self.env.totalTimeCosts[eps_idx][device.id - 1] = device.dag.t_dag
                self.env.totalEnergyCosts[eps_idx][device.id - 1] = device.dag.e_dag
                self.env.totalWeightCosts[eps_idx][device.id - 1] = device.dag.t_dag + device.dag.e_dag
            self.env.rewards[eps_idx][0] = total_reward.item()
            self.env.outputMetric()

            if eps_idx % 10 == 0 or eps_idx == self.episode_number - 1:
                print(f'Episode: {eps_idx}, Recent Actor Losses: {self.act_losses[-1:]}, Recent Critic Losses: {self.cri_losses[-1:]}\n')
                with open('../result/rl_hsac/metrics/loss.txt', 'a') as file:
                    file.write(f'Episode: {eps_idx}, Recent Actor Losses: {self.act_losses[-1:]}, Recent Critic Losses: {self.cri_losses[-1:]}\n')

            logging.info('Episode: %s | total_reward: %s | weight: %s', eps_idx, total_reward.item(), self.current_weight.tolist())
