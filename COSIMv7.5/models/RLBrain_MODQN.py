import logging

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

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


class MODQNet(nn.Module):
    def __init__(self, input_dim, out_c, out_d, wt_dim):  # out_c=2*(1+device_num)
        super(MODQNet, self).__init__()
        self.con_act_dim = out_c
        self.dis_act_dim = out_d
        hidden_dims = [128, 128]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.q_heads_continuous = nn.ModuleList([nn.Linear(hidden_dims[1], out_c) for _ in range(wt_dim)])
        self.q_heads_discrete = nn.ModuleList([nn.Linear(hidden_dims[1], out_d) for _ in range(wt_dim)])

        self.apply(weights_init_)
        self.to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        shared_out = self.shared_layers(x)
        q_values_continuous = [q_head(shared_out) for q_head in self.q_heads_continuous]
        q_values_discrete = [q_head(shared_out) for q_head in self.q_heads_discrete]
        return q_values_continuous, q_values_discrete

    def get_action(self, state, num_device, num_server):

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)

        with torch.no_grad():
            q_values_c, q_values_d = self.forward(state)

            action_c = (q_values_c[0] + q_values_c[1]) / 2
            action_c = torch.clamp(action_c, 0, 1)
            combined_q = sum(q_values_d)
            actions_d = []
            selected_actions_c = []
            for i in range(num_device):
                start_idx = i * (num_server + 1)
                end_idx = start_idx + num_server + 1
                action_d = combined_q[:, start_idx:end_idx].argmax(dim=1, keepdim=True)
                actions_d.append(action_d)

                action_c_start_idxs = i * (1 + num_server) * 2 + action_d * 2
                action_c_end_idxs = action_c_start_idxs + 2
                temp_c = torch.stack([action_c[i, start:end] for i, (start, end) in
                                      enumerate(zip(action_c_start_idxs, action_c_end_idxs))])
                selected_actions_c.append(temp_c)

            return torch.cat(selected_actions_c, dim=1), torch.cat(actions_d, dim=1)


def select_q_values(action, action_q, num_choices_per_action):

    batch_size, num_actions = action.shape
    selected_q_values = torch.zeros(batch_size, num_actions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    for i in range(num_actions):
        start_idx = i * num_choices_per_action
        end_idx = start_idx + num_choices_per_action
        q_values_for_action = action_q[:, start_idx:end_idx]
        selected_q_values[:, i] = q_values_for_action.gather(1, action[:, i:i + 1]).squeeze(1)

    return selected_q_values


def select_max_q_values(action_q, num_choices_per_action):

    batch_size, total_q_values = action_q.shape
    num_actions = total_q_values // num_choices_per_action
    max_q_values = torch.zeros(batch_size, num_actions, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for i in range(num_actions):
        start_idx = i * num_choices_per_action
        end_idx = start_idx + num_choices_per_action
        q_values_for_action = action_q[:, start_idx:end_idx]
        max_q_values[:, i] = q_values_for_action.max(dim=1)[0]

    return max_q_values

class MODQNAgent:
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

        self.state_dim = self.env.numberOfDevice * 3 + self.env.numberOfServer * 2  # bandwidth + computing capacity
        self.dis_act_dim = (self.env.numberOfServer + 1) * self.env.numberOfDevice
        self.con_act_dim = self.dis_act_dim * 2  # 每个设备对每个server的卸载率+计算效率

        self.policy_net = MODQNet(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.target_net = MODQNet(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayMemory(1000000000, 123456)
        self.weight_sampler = Weight_Sampler_pos(2)

        self.current_weight = None  # 当前权重
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        for eps_idx in range(self.episode_number):
            self.current_weight = self.weight_sampler.sample(1)
            self.env.episode = eps_idx
            self.env.reset()
            self.env.setUp()
            time_step = 0
            total_reward = 0
            while not self.env.isDAGsDone():
                time_step += 1
                current_state, old_dag_status = self.env.getEnvState()
                action_c, action_d, = self.policy_net.get_action(current_state, self.env.numberOfDevice, self.env.numberOfServer)
                self.env.offload(time_step, action_d, action_c)
                reward = self.env.getEnvReward(self.current_weight, old_dag_status, action_d, action_c)
                total_reward = total_reward + reward
                self.env.stepIntoNextState()
                next_state, _ = self.env.getEnvState()
                if not self.env.isDAGsDone():
                    self.memory.push(current_state, action_c, action_d, self.current_weight, reward, next_state, self.env.isDAGsDone())
                if len(self.memory) > 3 * self.batch_size:
                    state_batch, s_actions_c, s_actions_d, w_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)

                    q_values_c, q_values_d = self.policy_net.forward(state_batch)
                    next_q_values_c, next_q_values_d = self.target_net.forward(next_state_batch)
                    q_values_c = (q_values_c[0] + q_values_c[1]) / 2
                    q_values_d = (q_values_d[0] + q_values_d[1]) / 2
                    next_q_values_c = (next_q_values_c[0] + next_q_values_c[1]) / 2
                    next_q_values_d = (next_q_values_d[0] + next_q_values_d[1]) / 2

                    current_q_d = select_q_values(torch.tensor(s_actions_d, dtype=torch.int64).to(self.device), q_values_d, self.env.numberOfServer+1)
                    max_next_q_d = select_max_q_values(next_q_values_d, self.env.numberOfServer+1)

                    expected_q_d = torch.Tensor(reward_batch).to(self.device) + (1 - torch.Tensor(done_batch).to(self.device)) * self.gamma * max_next_q_d

                    loss_d = nn.MSELoss()(current_q_d, expected_q_d)
                    self.act_losses.append(loss_d.detach())

                    self.optimizer.zero_grad()
                    loss_d.backward()
                    self.optimizer.step()

                    if eps_idx % 1 == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

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
                print(f'Episode: {eps_idx}, Recent actor Losses: {self.act_losses[-1:]}\n')
                with open('../result/rl_modqn/metrics/loss.txt', 'a') as file:
                    file.write(
                        f'Episode: {eps_idx}, Recent Actor Losses: {self.act_losses[-1:]}\n')

            logging.info('Episode: %s | total_reward: %s | weight: %s', eps_idx, total_reward.item(),
                         self.current_weight.tolist())
