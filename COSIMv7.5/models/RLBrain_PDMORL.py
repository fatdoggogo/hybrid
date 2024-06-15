import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor
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
    def __init__(self, s_dim, out_c, out_d, wt_dim):
        super(Actor, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
        self.fc1 = nn.Linear(s_dim + wt_dim, 400)
        self.fc2 = nn.Linear(400, 300)

        self.cont_action_head = nn.Linear(300, out_c)
        self.disc_action_head = nn.Linear(300, out_d)

        self.apply(weights_init_)
        self.to(self.device)

    def forward(self, state, w):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)
        w = w.to(self.device)

        state_comp = torch.cat((state, w), dim=1)
        mask = torch.isnan(state_comp).any(dim=1)
        state_comp = state_comp[~mask]

        x = F.leaky_relu(self.fc1(state_comp), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)

        action_c = torch.sigmoid(self.cont_action_head(x))
        return action_c, self.disc_action_head(x)

    def get_action(self, state, w, num_device, num_server):

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        if not isinstance(w, torch.Tensor):
            w = torch.tensor(w, dtype=torch.float32)
        w = w.to(self.device)

        with torch.no_grad():
            action_c, disc_action_head = self.forward(state, w)

            disc_action_logits_chunks = torch.chunk(disc_action_head, num_device, dim=1)
            action_d_probs = [F.softmax(logits, dim=1) for logits in disc_action_logits_chunks]

            actions_d = []
            actions_c = []
            for i in range(num_device):
                if np.random.rand() < 0.1:
                    action_d = torch.randint(0, action_d_probs[i].size(1), (action_c.size(0), 1)).to(self.device)
                else:
                    action_d = action_d_probs[i].argmax(dim=1, keepdim=True).to(self.device)
                actions_d.append(action_d)

                action_c_start_idxs = i * (1 + num_server) * 2 + action_d * 2
                action_c_end_idxs = action_c_start_idxs + 2
                temp_c = torch.stack([action_c[i, start:end] for i, (start, end) in enumerate(zip(action_c_start_idxs, action_c_end_idxs))])
                actions_c.append(temp_c)

            return torch.cat(actions_c, dim=1), torch.cat(actions_d, dim=1)

    def to(self, device):
        return super(Actor, self).to(device)


class Critic(nn.Module):
    def __init__(self, state_dim, numberOfServer, numberOfDevice, wt_dim):
        super(Critic, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numberOfServer = numberOfServer
        self.numberOfDevice = numberOfDevice
        out_d = numberOfDevice
        out_c = numberOfDevice * 2

        self.fc1 = nn.Linear(state_dim + out_c + out_d + wt_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.q_head = nn.Linear(300, 2)

        self.fc3 = nn.Linear(state_dim + out_c + out_d + wt_dim, 400)
        self.fc4 = nn.Linear(400, 300)
        self.q2_head = nn.Linear(300, 2)

        self.apply(weights_init_)
        self.to(self.device)

    def forward(self, state, action_c, action_d, w):
        state = torch.FloatTensor(state).to(self.device) if not torch.is_tensor(state) else state
        action_c = torch.FloatTensor(action_c).to(self.device) if not torch.is_tensor(action_c) else action_c
        action_d = torch.Tensor(action_d).to(self.device) if not torch.is_tensor(action_d) else action_d.to(torch.int64)
        w = torch.FloatTensor(w).to(self.device) if not torch.is_tensor(w) else w

        x = torch.cat([state, action_d, action_c, w], 1)
        q1 = F.leaky_relu(self.fc1(x), 0.01)
        q1 = F.leaky_relu(self.fc2(q1), 0.01)
        q1 = self.q_head(q1)

        q2 = F.leaky_relu(self.fc3(x), 0.01)
        q2 = F.leaky_relu(self.fc4(q2), 0.01)
        q2 = self.q2_head(q2)

        return q1, q2

    def Q1(self, state, action_c, action_d, w):
        state = torch.FloatTensor(state).to(self.device) if not torch.is_tensor(state) else state
        action_c = torch.FloatTensor(action_c).to(self.device) if not torch.is_tensor(action_c) else action_c
        action_d = torch.Tensor(action_d).to(self.device) if not torch.is_tensor(action_d) else action_d.to(torch.int64)
        w = torch.FloatTensor(w).to(self.device) if not torch.is_tensor(w) else w

        x = torch.cat([state, action_d, action_c, w], 1)
        q1 = F.leaky_relu(self.fc1(x), 0.01)
        q1 = F.leaky_relu(self.fc2(q1), 0.01)
        q1 = self.q_head(q1)
        return q1


class PDMORLAgent:
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

        self.critic = Critic(self.state_dim, self.env.numberOfServer, self.env.numberOfDevice, self.wt_dim)
        self.actor = Actor(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

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
                action_c, action_d = self.actor.get_action(current_state, self.current_weight, self.env.numberOfDevice, self.env.numberOfServer)
                self.env.offload(time_step, action_d, action_c)
                reward = self.env.getEnvReward(self.current_weight, old_dag_status, action_d, action_c)
                total_reward = total_reward + reward
                self.env.stepIntoNextState()
                next_state, _ = self.env.getEnvState()
                if not self.env.isDAGsDone():
                    self.memory.push(current_state, action_c, action_d, self.current_weight, reward, next_state, self.env.isDAGsDone())
                if len(self.memory) > 3 * self.batch_size:
                    state_batch, s_actions_c, s_actions_d, w_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

                    w_batch_np_critic = copy.deepcopy(torch.tensor(w_batch).to(self.device).cpu().numpy())
                    w_batch_np_actor = copy.deepcopy(torch.tensor(w_batch).to(self.device).cpu().numpy())
                    with ((torch.no_grad())):
                        next_state_actions_c, next_state_actions_d = self.actor_target.get_action(next_state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                        target_Q1, target_Q2 = self.critic_target.forward(next_state_batch, next_state_actions_c, next_state_actions_d, w_batch)
                        wTauQ1 = torch.bmm(torch.FloatTensor(w_batch).to(self.device).unsqueeze(1), target_Q1.unsqueeze(2)).squeeze()
                        wTauQ2 = torch.bmm(torch.FloatTensor(w_batch).to(self.device).unsqueeze(1), target_Q2.unsqueeze(2)).squeeze()
                        _, wTauQ_min_idx = torch.min(torch.cat((wTauQ1.unsqueeze(-1), wTauQ2.unsqueeze(-1)), dim=-1), 1)
                        Tau_Q = torch.where(wTauQ_min_idx.unsqueeze(-1) == 0, target_Q1, target_Q2)
                        target_Q = torch.Tensor(reward_batch).to(self.device) + (1 - torch.Tensor(done_batch).to(self.device)) * self.gamma * Tau_Q

                    current_Q1, current_Q2 = self.critic.forward(state_batch, s_actions_c, s_actions_d, w_batch)

                    w_batch_critic_loss = torch.tensor(w_batch_np_critic, dtype=torch.float32, device=self.device)
                    angle_term_1 = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_critic_loss, current_Q1), 0, 0.9999)))
                    angle_term_2 = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_critic_loss, current_Q2), 0, 0.9999)))
                    critic_loss = angle_term_1.mean() + F.smooth_l1_loss(current_Q1, target_Q) + angle_term_2.mean() + F.smooth_l1_loss(current_Q2, target_Q)

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=100)
                    self.critic_optimizer.step()
                    self.cri_losses.append(critic_loss.detach())

                    if eps_idx % 1 == 0:  # TD 3 Delayed update support
                        for _ in range(1):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                            actions_c, actions_d = self.actor.get_action(state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                            Q = self.critic.Q1(state_batch, actions_c, actions_d, w_batch)
                            wQ = torch.bmm(torch.FloatTensor(w_batch).to(self.device).unsqueeze(1), Q.unsqueeze(2)).squeeze()
                            actor_loss = -wQ

                            w_batch_actor_loss = torch.tensor(w_batch_np_actor, dtype=torch.float32, device=self.device)
                            angle_term = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_actor_loss, Q), 0, 0.9999)))
                            actor_loss = actor_loss.mean() + 10 * angle_term.mean()

                            self.actor_optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=100)
                            self.actor_optimizer.step()
                            self.act_losses.append(actor_loss.detach())

                # update the target network
                    if eps_idx % 1 == 0:
                        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
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
