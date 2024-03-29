from typing import List

import torch

from device import Device
from server import Server
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

        self.envDir = '../result/env_' + str(self.id)
        self.algorithmDir = self.envDir + '/' + self.algorithm
        if not os.path.exists("../result"):
            os.mkdir("../result")
        if os.path.exists(self.algorithmDir):
            shutil.rmtree(path=self.algorithmDir)
        os.mkdir(self.algorithmDir)
        self.imageDir = self.algorithmDir + '/images'
        self.metricDir = self.algorithmDir + '/metrics'
        os.mkdir(self.imageDir)
        os.mkdir(self.metricDir)
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
        for d in self.devices:
            d.setUp()

    def getEnvState(self):
        state = []
        for device in self.devices:
            if device.dag.currentTask is not None:
                state.append(device.dag.currentTask.d_i)
                state.append(device.dag.currentTask.q_i)
                state.append(device.cpuFrequency)
            else:
                return None
        for server in self.servers:
            state.append(server.availableBW)
            state.append(server.availableFreq)
        return torch.tensor(state).unsqueeze(0)

    def getEnvReward(self, current_weight):
        total_reward = 0
        for device in self.devices:
            if device.dag.currentTask is None:
                continue
            t_local, e_local = device.totalLocalProcess(device.dag.currentTask.d_i)
            if t_local - device.dag.currentTask.T_i < 0:
                logging.info('device[%d] time error:%f < %f' % (device.id, t_local, device.dag.currentTask.T_i))
            time_r = (t_local - device.dag.currentTask.T_i) / t_local

            if e_local - device.dag.currentTask.E_i < 0:
                logging.info('device[%d] energy error:%f < %f' % (device.id, e_local, device.dag.currentTask.E_i))
            energy_r = (e_local - device.dag.currentTask.E_i) / e_local

            total_reward = total_reward + current_weight[0][0] * time_r + current_weight[0][1] * energy_r
        self.rewards[self.episode][0] = self.rewards[self.episode][0] + total_reward
        return total_reward

    def offload(self, time_step, dis_action, con_action):
        print("current timestep: ", time_step, "time_slot:", self.time_slot)
        for device in self.devices:
            if not device.dag.is_finished:
                server_id = dis_action[0, device.id-1].item()
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
                print("dag is finished, break")
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

    def calculateCost(self):
        """
        分别计算在当前策略，全部本地以及全部卸载三种情况下，按权重得到的总消耗，并将其存储
        :return:
        """
    # def outputMetric(self):
    #     """
    #     存储结果
    #     :return:
    #     """
    #     output = self.metricDir + '/cost.csv'
    #     np.savetxt(output, self.totalWeightCosts, fmt="%f", delimiter=',')
    #     output = self.metricDir + '/time-cost.csv'
    #     np.savetxt(output, self.totalTimeCosts, fmt="%f", delimiter=',')
    #     output = self.metricDir + '/energy-cost.csv'
    #     np.savetxt(output, self.totalEnergyCosts, fmt="%f", delimiter=',')
    #     output = self.metricDir + '/reward.csv'
    #     np.savetxt(output, self.rewards, fmt="%f", delimiter=',')


if __name__ == "__main__":
    import os
    print("当前工作目录:", os.getcwd())
    env = Env(1, "CAPQL")
    env.setUp()
