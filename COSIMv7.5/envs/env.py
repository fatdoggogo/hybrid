from typing import List

from server import *
from device import *
import logging
import json
import numpy as np
import os
import shutil

currentPath = os.path.abspath(__file__)
parentDir = os.path.abspath(os.path.dirname(currentPath) + os.path.sep + ".")


class Env(object):
    def __init__(self, env_id, algorithm_name):
        """
        :param env_id: 环境编号
        :param algorithm_name: 算法名称
        """
        self.id = env_id
        self.algorithm = algorithm_name

        configPath = parentDir + '/config.json'
        with open(configPath, encoding='utf-8') as f:
            configStr = f.read()
            config = json.loads(configStr)
        # 创建对应的地址，用来存储对应结果
        resultDir = parentDir + '/result'
        self.envDir = resultDir + '/env_' + str(self.id)
        self.algorithmDir = self.envDir + '/' + self.algorithm
        if not os.path.exists(resultDir):
            os.mkdir(resultDir)
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

        deviceId = 1
        for config in deviceConfigs:
            cnt = config['cnt']
            for i in range(0, cnt):
                device = Device(deviceId, config, self)
                self.devices.append(device)
                deviceId = deviceId + 1
        serverId = 1
        for config in serverConfigs:
            cnt = config['cnt']
            for i in range(0, cnt):
                server = Server(serverId, config, self)
                self.servers.append(server)
                serverId = serverId + 1

        self.numberOfDevice = len(self.devices)
        self.numberOfServer = len(self.servers)

        self.totalTimeCosts = np.zeros(shape=(self.episodes, 3))
        self.totalEnergyCosts = np.zeros(shape=(self.episodes, 3))
        self.rewards = np.zeros(shape=(self.episodes, 1))

    def getEnvState(self):
        state = []
        for device in self.devices:
            state.append(device.dag.currentTask)
            state.append(device.cpuFrequency)

        for server in self.servers:
            state.append(server.availableBW)
            state.append(server.availableFreq)
        return state

    def getEnvReward(self, current_weight):
        total_reward = 0
        for device in self.devices:
            t_local, e_local = device.totalLocalProcess(device.dag.currentTask.d_i)

            if t_local - device.dag.currentTask.T_i < 0:
                logging.info('device[%d] time error:%f < %f' % (device.id, t_local, device.dag.currentTask.T_i))
            time_r = (t_local - device.dag.currentTask.T_i) / t_local

            if e_local - device.dag.currentTask.E_i < 0:
                logging.info('device[%d] energy error:%f < %f' % (device.id, e_local, device.dag.currentTask.E_i))
            energy_r = (e_local - device.dag.currentTask.E_i) / e_local

            total_reward = total_reward + current_weight[0] * time_r + current_weight[1] * energy_r
        self.rewards[self.episode][0] = self.rewards[self.episode][0] + total_reward
        return total_reward

    def offload(self, time_step, curr_task: Task, dis_action, con_action):
        i = 0
        for device in self.devices:
            server_id = dis_action[i]
            if server_id >= 1:
                curr_task.server_id = server_id
                curr_task.offloading_rate = con_action[i * 2]
                curr_task.computing_f = con_action[i * 2 + 1]
            else:
                curr_task.server_id = 0
                curr_task.offloading_rate = 0
                curr_task.computing_f = 0
            device.offload(curr_task, time_step)
            i += 1
        for server in self.servers:
            server.process(self.time_slot, time_step)
        self.calculateCost()

    def setUp(self, timestep):
        for device in self.devices:
            device.setUp(timestep)

    def stepIntoNextState(self):
        for device in self.devices:
            device.updateState()

    def reset(self, timestep):
        for device in self.devices:
            device.reset()
            device.setUp(timestep)
        for server in self.servers:
            server.reset()

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
