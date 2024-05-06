import torch
from device import Device
from server import Server
from task import *
import logging
import json
import numpy as np
import random

# 设置种子
random.seed(42)


class Env:
    def __init__(self, env_id, algorithm_name):

        self.id = env_id
        self.algorithm = algorithm_name
        configPath = '../config/config.json'
        with open(configPath, encoding='utf-8') as f:
            configStr = f.read()
            config = json.loads(configStr)

        self.algorithmDir = '../result/' + self.algorithm
        self.imageDir = self.algorithmDir + '/images'
        self.metricDir = self.algorithmDir + '/metrics'
        self.episodes = config['episodes']
        self.devices: List[Device] = []
        self.servers: List[Server] = []
        self.time_slot = 100
        deviceConfigs = config['device_configs']
        serverConfigs = config['server_configs']

        for d_config in deviceConfigs:
            d = Device(d_config, self)
            self.devices.append(d)

        for s_config in serverConfigs:
            s = Server(s_config, self)
            self.servers.append(s)

        self.numberOfDevice = len(self.devices)
        self.numberOfServer = len(self.servers)

        self.totalWeightCosts = np.zeros(shape=(self.episodes, len(self.devices)))
        self.totalTimeCosts = np.zeros(shape=(self.episodes, len(self.devices)))
        self.totalEnergyCosts = np.zeros(shape=(self.episodes, len(self.devices)))
        self.rewards = np.zeros(shape=(self.episodes, 1))

    def setUp(self):
        for d in self.devices:
            d.setUp()

    def getEnvState(self):
        d_i_states = []
        q_i_states = []
        cpu_freq_states = []
        for device in self.devices:
            if device.dag.currentTask is not None:
                d_i_states.append(device.dag.currentTask.d_i)
                q_i_states.append(device.dag.currentTask.q_i)
                cpu_freq_states.append(device.availableCpuFreq)
            else:
                d_i_states.append(-1)
                q_i_states.append(-1)
                cpu_freq_states.append(-1)
        bw_states = []
        freq_states = []
        for server in self.servers:
            bw_states.append(server.availableBW)
            freq_states.append(server.availableFreq)
        d_i_tensor = self.normalize(torch.tensor(d_i_states, dtype=torch.float))
        q_i_tensor = self.normalize(torch.tensor(q_i_states, dtype=torch.float))
        cpu_freq_tensor = torch.tensor(cpu_freq_states, dtype=torch.float)
        bw_tensor = self.normalize(torch.tensor(bw_states, dtype=torch.float))
        freq_tensor = self.normalize(torch.tensor(freq_states, dtype=torch.float))
        state = torch.cat((d_i_tensor, q_i_tensor, cpu_freq_tensor, bw_tensor, freq_tensor))[None, :]
        return state

    @staticmethod
    def normalize(x: torch.Tensor, eps=1e-5, special_value=-1) -> torch.Tensor:
        mask = x != special_value
        valid_x = x[mask]

        if len(valid_x) < 2:
            if torch.any(mask):
                x[mask] = 1.0
            return x
        mean = valid_x.mean()
        std = valid_x.std()
        std = std if std > eps else eps
        normalized_x = torch.clone(x)
        normalized_x[mask] = (x[mask] - mean) / std

        return normalized_x

    @staticmethod
    def min_max_normalize(data: torch.Tensor, min_val: float, max_val: float, special_value=-1) -> torch.Tensor:
        if special_value is not None:
            mask = data != special_value
        else:
            mask = torch.ones_like(data, dtype=torch.bool)
        eps = 1e-8
        denom = max_val - min_val if max_val != min_val else eps
        normalized_data = torch.clone(data)
        normalized_data[mask] = (data[mask] - min_val) / denom

        return normalized_data

    def getEnvReward(self, current_weight):
        total_reward = 0
        for device in self.devices:
            if device.dag.currentTask is None:
                continue
            t_local, e_local = device.totalLocalProcess(device.dag.currentTask.d_i)
            device.dag.currentTask.rwd_t = (t_local - device.dag.currentTask.T_reward_i) / t_local
            device.dag.currentTask.rwd_e = (e_local - device.dag.currentTask.E_i) / e_local
            device.dag.currentTask.rwd = current_weight[0][0] * device.dag.currentTask.rwd_t + current_weight[0][
                1] * device.dag.currentTask.rwd_e
            device.dag.currentTask.rwd = torch.clamp(device.dag.currentTask.rwd, min=-1.00)
            total_reward = total_reward + device.dag.currentTask.rwd
            logging.info(
                'dev[%s], T_i: %s, T_rwd_i:%.2f, t_local:%.2f, '
                'E_i:%.2f, e_local:%.2f, '
                't_rwd: %.2f, e_rwd: %.2f,'
                'rwd:%.2f, '
                'w: %.2f %.2f,'
                'status:%s, curr_task_id:%s,'
                'l_finish:%s, server_finish:%s',
                device.id, device.dag.currentTask.T_i,
                round(device.dag.currentTask.T_reward_i, 2), round(t_local, 2),
                round(device.dag.currentTask.E_i, 2), round(e_local, 2),
                round(device.dag.currentTask.rwd_t, 2), round(device.dag.currentTask.rwd_e, 2),
                round(device.dag.currentTask.rwd.item(), 2),
                round(current_weight[0][0].item(), 2), round(current_weight[0][1].item(), 2),
                device.dag.currentTask.status, device.dag.currentTask.id,
                device.dag.currentTask.local_finished, device.dag.currentTask.server_finished
            )
        return total_reward

    def offload(self, time_step, dis_action, con_action):
        for device in self.devices:
            if not device.dag.is_finished:
                server_id = dis_action[0, device.id - 1].item()
                if device.dag.currentTask.status == TaskStatus.NOT_SCHEDULED:
                    if server_id >= 1:
                        device.dag.currentTask.server_id = server_id
                        device.dag.currentTask.offloading_rate = con_action[0, (device.id - 1) * 2].item()
                        device.dag.currentTask.computing_f = con_action[0, (device.id - 1) * 2 + 1].item()
                    else:
                        device.dag.currentTask.server_id = 0
                        device.dag.currentTask.offloading_rate = 0
                        device.dag.currentTask.computing_f = 0
                device.offload(device.dag.currentTask, time_step)
        for server in self.servers:
            server.process(self.time_slot, time_step)

    def stepIntoNextState(self):
        for d in self.devices:
            if not d.dag.is_finished:
                d.updateState()

    def reset(self):
        for d in self.devices:
            d.reset()
        for s in self.servers:
            s.reset()

    def isDAGsDone(self):
        return all(device.dag.is_finished for device in self.devices)

    def outputMetric(self):
        np.savetxt(self.metricDir + '/cost.csv', self.totalWeightCosts, fmt="%f", delimiter=',')
        np.savetxt(self.metricDir + '/time-cost.csv', self.totalTimeCosts, fmt="%f", delimiter=',')
        np.savetxt(self.metricDir + '/energy-cost.csv', self.totalEnergyCosts, fmt="%f", delimiter=',')
        np.savetxt(self.metricDir + '/reward.csv', self.rewards, fmt="%f", delimiter=',')

