import copy

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

import utils
from utils import *
from replay_memory import *


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):

    def __init__(self, s_dim, out_c, out_d, wt_dim):
        print(str(s_dim), str(out_c), str(out_d), str(wt_dim))
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(s_dim + wt_dim, 128)
        self.linear2 = nn.Linear(128, 128)

        self.pi_d = nn.Linear(128, out_d)
        self.mean_linear = nn.Linear(128, out_c)
        self.log_std_linear = nn.Linear(128, out_c)

        self.sigmoid = nn.Sigmoid()
        self.apply(weights_init_)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state, w):
        state_comp = torch.cat((state, w), dim=1)
        mask = torch.isnan(state_comp).any(dim=1)
        state_comp = state_comp[~mask]
        x = F.relu(self.linear1(state_comp))
        x = F.relu(self.linear2(x))

        pi_d = self.pi_d(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        log_std = torch.clamp(log_std, min=-20, max=2)
        return pi_d, mean, log_std

    def sample(self, state, w, num_device, num_server):
        state = torch.FloatTensor(state).to(self.device) if not torch.is_tensor(state) else state
        w = torch.FloatTensor(w).to(self.device) if not torch.is_tensor(w) else w
        pi_d, mean, log_std = self.forward(state, w)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))

        # restrict the outputs for continuous actions
        action_c = torch.sigmoid(x_t)
        log_prob_c = normal.log_prob(x_t)
        log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)

        actions_d = []
        probs_d = []
        log_probs_d = []
        selected_action_c = []
        selected_log_prob_c = []

        for i in range(num_device):
            start_idx = i * (num_server+1)
            end_idx = start_idx + num_server + 1

            dist = Categorical(logits=pi_d[:, start_idx:end_idx])
            action_d = dist.sample()
            prob_d = dist.probs
            log_prob_d = torch.log(prob_d + 1e-8)

            actions_d.append(action_d)
            probs_d.append(prob_d)
            log_probs_d.append(log_prob_d)

            # action_c_start_idx = i * (1+num_server) * 2 + action_d.item() * 2
            # action_c_end_idx = action_c_start_idx + 2
            #
            # selected_action_c.append(action_c[:, action_c_start_idx:action_c_end_idx])
            # selected_log_prob_c.append(log_prob_c[:, action_c_start_idx:action_c_end_idx])

            action_c_start_idxs = i * (1 + num_server) * 2 + action_d * 2
            action_c_end_idxs = action_c_start_idxs + 2
            selected_action_c_batch = torch.stack([action_c[i, start:end] for i, (start, end) in enumerate(zip(action_c_start_idxs, action_c_end_idxs))])
            selected_log_prob_c_batch = torch.stack([log_prob_c[i, start:end] for i, (start, end) in enumerate(zip(action_c_start_idxs, action_c_end_idxs))])
            selected_action_c.append(selected_action_c_batch)
            selected_log_prob_c.append(selected_log_prob_c_batch)

        actions_2d = [action.unsqueeze(0) if action.dim() == 1 else action for action in actions_d]
        prob_d_2d = [prob_d.unsqueeze(0) if prob_d.dim() == 1 else prob_d for prob_d in probs_d]
        log_prob_d_2d = [log_prob_d.unsqueeze(0) if log_prob_d.dim() == 1 else log_prob_d for log_prob_d in log_probs_d]
        action_d = torch.cat(actions_2d, dim=1)  # tensor[[server1, server2, local]]
        prob_d = torch.cat(prob_d_2d, dim=1)
        log_prob_d = torch.cat(log_prob_d_2d, dim=1)

        action_c = torch.cat(selected_action_c, dim=1)
        log_prob_c = torch.cat(selected_log_prob_c, dim=1)

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
        w = torch.tensor(w) if not isinstance(w, torch.Tensor) else w
        x = torch.cat([state, action, w], 1)
        x1 = self.Q1(x)
        x2 = self.Q2(x)
        return x1, x2


class CAPQL:
    def __init__(self, env):

        self.env = env
        self.batch_size = 32
        self.episode_number = 2000
        self.ep_steps = 100  # 每次训练100个batch_size
        self.memory_size = 10000
        self.lr = 0.002
        self.tau = 0.01  # 每次两个网络间参数转移的衰减程度
        self.gamma = 0.9  # 未来的r的比例

        self.wt_dim = 2
        self.rwd_dim = 1

        self.alpha_c = 0.2
        self.alpha_d = 0.2

        self.state_dim = self.env.numberOfDevice * 4 + self.env.numberOfServer * 2  # bandwidth + computing capacity
        self.dis_act_dim = (self.env.numberOfServer + 1) * self.env.numberOfDevice
        self.con_act_dim = (self.env.numberOfServer + 1) * 2 * self.env.numberOfDevice  # 每个设备对每个server的卸载率+计算效率

        # define TWO Q networks for training
        self.critic = QNetwork(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)
        self.actor = Actor(self.state_dim, self.con_act_dim, self.dis_act_dim, self.wt_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.policy_optimizer = Adam(self.actor.parameters(), lr=self.lr)

        self.memory = ReplayMemory(100000, 123456)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.weight_sampler = Weight_Sampler_pos(2)

        self.current_weight = None  # 当前权重
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        for eps_idx in range(self.episode_number):
            self.current_weight = self.weight_sampler.sample(1)  # 每个回合 变换权重
            self.env.reset()
            self.env.setUp()
            time_step = 0
            total_reward = 0
            one_episode_dis_actions = []
            while not self.env.isDAGsDone():
                time_step += 1
                current_state = self.env.getEnvState()
                if current_state is None:
                    break
                action_c, action_d, log_prob_c, log_prob_d, prob_d = self.actor.sample(current_state, self.current_weight, self.env.numberOfDevice, self.env.numberOfServer)
                self.env.offload(time_step, action_d, action_c)
                reward = self.env.getEnvReward(self.current_weight)
                total_reward = total_reward + reward
                self.env.stepIntoNextState()
                next_state = self.env.getEnvState()
                self.memory.push(current_state, [action_c, action_d], self.current_weight, reward, next_state, self.env.isDAGsDone())
                one_episode_dis_actions.append(action_d)

                if len(self.memory) > 3 * self.batch_size:
                    state_batch, action_batch, w_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(self.batch_size)
                    with torch.no_grad():
                        next_s_actions_c, next_s_actions_d, next_s_log_pi_c, next_s_log_pi_d, next_s_prob_d = self.actor.sample(next_state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_s_actions_c, w_batch)
                        min_qf_next_target_wt = next_s_prob_d * (torch.min(qf1_next_target,
                                                                           qf2_next_target) - self.alpha_c * next_s_prob_d * next_s_log_pi_c - self.alpha_d * next_s_log_pi_d)
                        mask_batch_tensor = torch.tensor(mask_batch, dtype=torch.float32)
                        next_q_value = reward + mask_batch_tensor * self.gamma * min_qf_next_target_wt

                    s_actions_c, s_actions_d = utils.to_torch_action(action_batch, self.device)
                    qf1, qf2 = self.critic.forward(state_batch, s_actions_c, w_batch)
                    qf1 = qf1.gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
                    qf2 = qf2.gather(1, s_actions_d.long().view(-1, 1).to(self.device)).squeeze().view(-1)
                    qf1_loss = F.mse_loss(qf1, next_q_value)
                    qf2_loss = F.mse_loss(qf2, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self.critic_optim.zero_grad()
                    qf_loss.backward()
                    self.critic_optim.step()

                    # train the policy network
                    actions_c, actions_d, log_pi_c, log_pi_d, prob_d = self.actor.sample(state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                    qf1_pi_w, qf2_pi_w = self.critic(state_batch, actions_c, w_batch)
                    min_qf_pi_w = torch.min(qf1_pi_w, qf2_pi_w)
                    min_qf_pi_w = (min_qf_pi_w * w_batch).sum(dim=-1, keepdim=True)

                    policy_loss_d = (prob_d * (self.alpha_d * log_pi_d - min_qf_pi_w)).sum(1).mean()
                    policy_loss_c = (prob_d * (self.alpha_c * prob_d * log_pi_c - min_qf_pi_w)).sum(1).mean()
                    policy_loss = policy_loss_d + policy_loss_c

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    # sync the Q networks
                    if eps_idx % 1 == 0:
                        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                if self.env.isDAGsDone():
                    break

            print('Episode: ', eps_idx, ' | total_reward: ', total_reward.item(),
                  ' | weight: ', self.current_weight)


