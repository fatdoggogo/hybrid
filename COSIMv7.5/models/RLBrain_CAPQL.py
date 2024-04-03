import copy
import torch.nn.functional as F
from torch.optim import Adam

import utils
from CAPQL import QNetwork, Actor
from utils import *
from replay_memory import *


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
        self.alpha_c = 0.2
        self.alpha_d = 0.2

        # actor
        self.state_dim = self.env.numberOfDevice * 3 + self.env.numberOfServer * 2  # bandwidth + computing capacity
        self.dis_act_dim = (self.env.numberOfServer + 1) * self.env.numberOfDevice
        self.con_act_dim = self.dis_act_dim * 2  # 每个设备对每个server的卸载率+计算效率
        # define TWO Q networks for training
        self.critic = QNetwork(self.state_dim, self.env.numberOfServer, self.env.numberOfDevice, self.wt_dim)
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
            self.env.episode = eps_idx
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
                action_c_full, _, action_c, action_d, _, _, _ = self.actor.sample(current_state, self.current_weight, self.env.numberOfDevice, self.env.numberOfServer)
                self.env.offload(time_step, action_d, action_c)
                reward = self.env.getEnvReward(time_step, self.current_weight)
                total_reward = total_reward + reward
                self.env.stepIntoNextState()
                next_state = self.env.getEnvState()
                if next_state is not None:
                    self.memory.push(action_c_full, current_state, [action_c, action_d], self.current_weight, reward, next_state, self.env.isDAGsDone())
                    one_episode_dis_actions.append(action_d)

                if len(self.memory) > 3 * self.batch_size:
                    _, state_batch, action_batch, w_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(self.batch_size)
                    with torch.no_grad():
                        _, log_prob_c_full, next_s_actions_c, next_s_actions_d, _, next_s_log_d, next_s_prob_d = self.actor.sample(next_state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_s_actions_c, next_s_actions_d, w_batch)
                        next_log_c = (next_s_prob_d * log_prob_c_full).sum(1, keepdim=True).clamp(-1e3, 1e3)
                        next_log_d = next_s_log_d.sum(1, keepdim=True).clamp(-1e3, 1e3)
                        min_qf_next_target_wt = torch.min(qf1_next_target, qf2_next_target) - self.alpha_c * next_log_c - self.alpha_d * next_log_d
                        next_q_value = torch.Tensor(reward_batch).to(self.device) + self.gamma * min_qf_next_target_wt

                    s_actions_c, s_actions_d = utils.to_torch_action(action_batch, self.device, self.env.numberOfDevice)
                    qf1, qf2 = self.critic.forward(state_batch, s_actions_c, s_actions_d, w_batch)
                    qf1_loss = F.mse_loss(qf1, next_q_value)
                    qf2_loss = F.mse_loss(qf2, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    self.critic_optim.zero_grad()
                    qf_loss.backward()
                    self.critic_optim.step()

                    # train the policy network
                    _, log_prob_c_full, actions_c, actions_d, _, log_pi_d, prob_d = self.actor.sample(state_batch, w_batch, self.env.numberOfDevice, self.env.numberOfServer)
                    qf1_pi, qf2_pi = self.critic(state_batch, actions_c, actions_d, w_batch)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    min_qf_pi = (min_qf_pi * torch.FloatTensor(w_batch).to(self.device)).sum(dim=-1, keepdim=True)

                    policy_loss_d = ((self.alpha_d * log_pi_d) - min_qf_pi).mean()
                    policy_loss_c = ((self.alpha_c * prob_d * log_prob_c_full) - min_qf_pi).mean()
                    policy_loss = policy_loss_d + policy_loss_c

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

                    # sync the Q networks
                    if eps_idx % 1 == 0:
                        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                if self.env.isDAGsDone():
                    print("dags are done")
                    break

            print('Episode: ', eps_idx, ' | total_reward: ', total_reward.item(), ' | weight: ', self.current_weight)


