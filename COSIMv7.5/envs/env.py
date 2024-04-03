import csv
from typing import List

import torch

from device import Device
from server import Server
from task import *
import logging
import json
import numpy as np
import shutil
import os


class Env:
    def __init__(self, env_id, algorithm_name):

        self.id = env_id
        self.algorithm = algorithm_name
        configPath = '../config/config.json'
        with open(configPath, encoding='utf-8') as f:
            configStr = f.read()
            config = json.loads(configStr)

        self.algorithmDir = '../result/' + self.algorithm
        if os.path.exists(self.algorithmDir):
            shutil.rmtree(path=self.algorithmDir)
        self.imageDir = self.algorithmDir + '/images'
        self.metricDir = self.algorithmDir + '/metrics'
        os.makedirs(self.imageDir, exist_ok=True)
        os.makedirs(self.metricDir, exist_ok=True)

        self.episodes = config['episodes']
        self.episode = 0
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

        self.totalTimeCosts = np.zeros(shape=(self.episodes, 3))
        self.totalEnergyCosts = np.zeros(shape=(self.episodes, 3))
        self.rewards = np.zeros(shape=(self.episodes, 1))

    def setUp(self):
        output_path = self.metricDir + '/task.csv'
        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epi', 'step', 'DevID', 'TaskID', 'status', 'server', 'offRate', 'compf', 'realf', 'realBW',
                             'd', 'q', 'Ttrans', 'Texec_l', 'Texec_ser', 'start', 'finish', 'rwd', 'rwt_t', 'rwt_e'])
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
                cpu_freq_states.append(device.cpuFrequency)
            else:
                state.append(-1)
                state.append(-1)
                state.append(-1)
        bw_states = []
        freq_states = []
        for server in self.servers:
            bw_states.append(server.availableBW)
            freq_states.append(server.availableFreq)
        d_i_tensor = self.normalize(torch.tensor(d_i_states, dtype=torch.float))
        q_i_tensor = self.normalize(torch.tensor(q_i_states, dtype=torch.float))
        cpu_freq_tensor = self.normalize(torch.tensor(cpu_freq_states, dtype=torch.float))
        bw_tensor = self.normalize(torch.tensor(bw_states, dtype=torch.float))
        freq_tensor = self.normalize(torch.tensor(freq_states, dtype=torch.float))
        state = torch.cat((d_i_tensor, q_i_tensor, cpu_freq_tensor, bw_tensor, freq_tensor))[None, :]
        return state

    @staticmethod
    def normalize(x: torch.Tensor, eps=1e-5) -> torch.Tensor:
        return (x - x.mean()) / (x.std() + eps)

    def outputMetric(self, episode, time_step, task):
        output_path = self.metricDir + '/task.csv'
        with open(output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode, time_step, task.device_id, task.id, task.status,
                             task.server_id, task.offloading_rate, task.computing_f, task.computing_f, task.bw,
                             task.d_i, task.q_i, task.T_trans_i, task.T_exec_local_i, task.T_exec_server_i,
                             task.start_step, task.finish_step, task.rwd, task.rwd_t, task.rwd_e])

    def getEnvReward(self, time_step, current_weight):
        total_reward = 0
        for device in self.devices:
            if device.dag.currentTask is None:
                continue
            t_local, e_local = device.totalLocalProcess(device.dag.currentTask.d_i)
            device.dag.currentTask.rwd_t = (t_local - device.dag.currentTask.T_reward_i) / t_local
            device.dag.currentTask.rwd_e = (e_local - device.dag.currentTask.E_i) / e_local
            device.dag.currentTask.rwd = current_weight[0][0] * device.dag.currentTask.rwd_t + current_weight[0][1] * device.dag.currentTask.rwd_e
            total_reward = total_reward + device.dag.currentTask.rwd
            print(f'device[{device.id}], T_i: {device.dag.currentTask.T_i}, T_reward_i:{round(device.dag.currentTask.T_reward_i,2)}, t_local:{round(t_local,2)}, '
                  f'E_i:{round(device.dag.currentTask.E_i,2)}, e_local:{round(e_local,2)}, '
                  f'time_reward: {round(device.dag.currentTask.rwd_t,2)},energy_reward: {round(device.dag.currentTask.rwd_e,2)},'
                  f'reward:{round(device.dag.currentTask.rwd.item(),2)}, '
                  f'weight: {round(current_weight[0][0].item(),2)} {round(current_weight[0][1].item(),2)},'
                  f'status:{device.dag.currentTask.status}, current_task_id:{device.dag.currentTask.id}')
            self.outputMetric(self.episode, time_step, device.dag.currentTask)
        self.rewards[self.episode][0] = self.rewards[self.episode][0] + total_reward
        return total_reward

    def offload(self, time_step, dis_action, con_action):
        print("current timestep: ", time_step, "time_slot:", self.time_slot)
        for device in self.devices:
            if not device.dag.is_finished:
                server_id = dis_action[0, device.id-1].item()
                if device.dag.currentTask.status == TaskStatus.NOT_SCHEDULED:
                    if server_id >= 1:
                        device.dag.currentTask.server_id = server_id
                        device.dag.currentTask.offloading_rate = con_action[0, (device.id-1) * 2].item()
                        device.dag.currentTask.computing_f = con_action[0, (device.id-1) * 2 + 1].item()
                    else:
                        device.dag.currentTask.server_id = 0
                        device.dag.currentTask.offloading_rate = 0
                        device.dag.currentTask.computing_f = 0
                device.offload(device.dag.currentTask, time_step)
            else:
                print("device", device.id, "is finished, break")
        for server in self.servers:
            server.process(self.time_slot, time_step)

    def stepIntoNextState(self):
        for d in self.devices:
            if not d.dag.is_finished:
                d.updateState()

    def reset(self):
        for d in self.devices:
            d.reset()
            d.setUp()
        for s in self.servers:
            s.reset()

    def isDAGsDone(self):
        return all(device.dag.is_finished for device in self.devices)

    # def outputMetric(self):
    #     output = self.metricDir + '/task.csv'
    #     np.savetxt(output, self.devices, fmt="%f", delimiter=',')
        # output = self.metricDir + '/cost.csv'
        # np.savetxt(output, self.totalWeightCosts, fmt="%f", delimiter=',')
        # output = self.metricDir + '/time-cost.csv'
        # np.savetxt(output, self.totalTimeCosts, fmt="%f", delimiter=',')
        # output = self.metricDir + '/energy-cost.csv'
        # np.savetxt(output, self.totalEnergyCosts, fmt="%f", delimiter=',')
        # output = self.metricDir + '/reward.csv'
        # np.savetxt(output, self.rewards, fmt="%f", delimiter=',')


if __name__ == "__main__":
    import os
    print("当前工作目录:", os.getcwd())
    env = Env(1, "CAPQL")
    env.setUp()
