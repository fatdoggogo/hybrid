from typing import List

from server import Server
from device import Device
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
        self.logDir = self.algorithmDir + '/logs'
        os.mkdir(self.imageDir)
        os.mkdir(self.metricDir)
        os.mkdir(self.logDir)

        self.logger = logging.getLogger()

        # 环境当前时钟
        self.clock = 1
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
            state.append(device.cpuFrequency)
            state.append(device.dag.currentTask.d_i)
            state.append(device.dag.currentTask.q_i)

        for server in self.servers:
            state.append(server.availableBW)
            state.append(server.availableFreq)
        return np.array(state)

    def setUp(self):
        self.logger.info('set up start')
        for device in self.devices:
            device.setUp()
        self.logger.info('set up finish')

    def stepIntoNextState(self):
        for device in self.devices:
            device.updateState()

    def getEnvReward(self):
        total_reward = 0
        for device in self.devices:
            t_local, e_local = device.totalLocalProcess(device.dag.currentTask.d_i)

            if t_local - device.dag.currentTask.T_i < 0:
                logging.info('device[%d] time error:%f < %f' % (device.id, t_local, device.dag.currentTask.T_i))
            time_r = (t_local - device.dag.currentTask.T_i) / t_local

            if e_local - device.dag.currentTask.E_i < 0:
                logging.info('device[%d] energy error:%f < %f' % (device.id, e_local, device.dag.currentTask.E_i))
            energy_r = (e_local - device.dag.currentTask.E_i) / e_local

            failed_p = 0 if device.finished == True else 1
            total_reward = currentTask + self.timeWeight * time_r + self.energyWeight * energy_r - 0.5 * failed_p
        self.rewards[self.episode][0] = self.rewards[self.episode][0] + total_reward
        if self.clock == self.T:
            self.logger.info(
                'episode:%d - environment total reward = %f' % (self.episode, self.rewards[self.episode][0]))
        return total_reward

    # def offload(self, makeOffloadDecision, actions):
    #     """
    #     执行计算卸载决策
    #     :param makeOffloadDecision:传入的操作函数
    #     :param actions:传入的行为动作
    #     :return:
    #     """
    #     # 同一个time slot,多个device基于同一状态做出卸载决策
    #     makeOffloadDecision(self, actions)
    #     # server在每个time slot 会处理当前time slot提交到该server的所有任务
    #     for server in self.servers:
    #         server.process()  # server端的计算
    #         server.testProcess()
    #     self.calculateCost()

    # TODO://step() 根据当前所有dag的current_task 决定T_i,更新T_i
    def offload(self, makeOffloadDecision, dis_actions, con_actions, f_action):
        # 同一个time slot,多个device基于同一状态做出卸载决策
        makeOffloadDecision(self, dis_actions, con_actions, f_action)
        # server在每个time slot 会处理当前time slot提交到该server的所有任务
        for server in self.servers:
            server.process()  # server端的计算
        self.calculateCost()

    def outputMetric(self):
        """
        存储结果
        :return:
        """
        output = self.metricDir + '/cost.csv'
        np.savetxt(output, self.totalWeightCosts, fmt="%f", delimiter=',')
        output = self.metricDir + '/time-cost.csv'
        np.savetxt(output, self.totalTimeCosts, fmt="%f", delimiter=',')
        output = self.metricDir + '/energy-cost.csv'
        np.savetxt(output, self.totalEnergyCosts, fmt="%f", delimiter=',')
        output = self.metricDir + '/reward.csv'
        np.savetxt(output, self.rewards, fmt="%f", delimiter=',')

    def hardReset(self):
        """vm
        完全重置
        :return:
        """
        self.reset()
        self.totalWeightCosts = np.zeros(shape=(self.episodes, 3))
        self.totalTimeCosts = np.zeros(shape=(self.episodes, 3))
        self.totalEnergyCosts = np.zeros(shape=(self.episodes, 3))
        self.rewards = np.zeros(shape=(self.episodes, 1))
        self.failures = 0
        self.errors = 0

    def reset(self):
        """
        重置
        :return:
        """
        # device reset
        for device in self.devices:
            device.reset()
            device.setUp()
        # server reset
        for server in self.servers:
            server.updateState()

    def calculateCost(self):
        """
        分别计算在当前策略，全部本地以及全部卸载三种情况下，按权重得到的总消耗，并将其存储
        :return:
        """
        weightSum = self.timeWeight + self.energyWeight
        for device in self.devices:
            # 记录对应权重
            p1 = device.priority * 1.0 / self.prioritySum
            # 部分卸载
            timeCost1 = (self.timeWeight / weightSum) * device.currProcessedDelay * p1
            energyCost1 = (self.energyWeight / weightSum) * device.currProcessedEnergy * p1
            totalCost1 = timeCost1 + energyCost1
            self.totalTimeCosts[self.episode][0] = self.totalTimeCosts[self.episode][0] + timeCost1
            self.totalEnergyCosts[self.episode][0] = self.totalEnergyCosts[self.episode][0] + energyCost1
            # 全部本地处理
            timeCost2 = (self.timeWeight / weightSum) * device.totalLocalProcessTime * p1
            energyCost2 = (self.energyWeight / weightSum) * device.totalLocalProcessEnergy * p1
            totalCost2 = timeCost2 + energyCost2
            self.totalTimeCosts[self.episode][1] = self.totalTimeCosts[self.episode][1] + timeCost2
            self.totalEnergyCosts[self.episode][1] = self.totalEnergyCosts[self.episode][1] + energyCost2
            # 全部卸载
            timeCost3 = (self.timeWeight / weightSum) * device.totalRemoteProcessTime * p1
            energyCost3 = (self.energyWeight / weightSum) * device.totalRemoteProcessEnergy * p1
            totalCost3 = timeCost3 + energyCost3
            self.totalTimeCosts[self.episode][2] = self.totalTimeCosts[self.episode][2] + timeCost3
            self.totalEnergyCosts[self.episode][2] = self.totalEnergyCosts[self.episode][2] + energyCost3

            self.totalWeightCosts[self.episode][0] = self.totalWeightCosts[self.episode][0] + totalCost1
            self.totalWeightCosts[self.episode][1] = self.totalWeightCosts[self.episode][1] + totalCost2
            self.totalWeightCosts[self.episode][2] = self.totalWeightCosts[self.episode][2] + totalCost3