import logging
import math

import pandas as pd

from task import *

np.random.seed(3)

# device一定会处理完这个任务才会处理下一个task
class Device(object):
    def __init__(self, id, config, env):
        self.env = env  # 设备所在的环境
        self.id = id
        self.cpuFrequency = config['cpu_frequency']  # 每个设备的CPU频率  f(n,l)
        self.BW = config['BW']  # server向device传输结果数据时可分配的带宽 unit MHZ
        self.En = config['energy_consume_per_time_slot']  # 设备单位时间传输数据需要的能量 unit mJ
        self.transPower = config['transmission_power']
        self.channelGain = config['channel_gain']
        self.channelNoise = config['channel_noise']
        self.dag = None
        self.global_sequence = []
        self.metrics = np.zeros(shape=(self.env.T, 16))
        logging.info('init [Device-%d] fininsh' % self.id)

    @staticmethod
    def generateDAG(task_num, fat_num):
        # n_list = [15, 25, 35]  # DAG任务数量
        n_list = [task_num]
        fat_list = [fat_num]  # 决定DAG的宽度和高度，较小的fat值可构造较瘦高的图，值越大可构造较矮胖的图
        density_list = [0.7]  # 决定两层之间边的数量，值越小边越少，反之
        regularity = 0.5  # 决定每层任务数的差异性
        jump = 1

        index = 1
        for n in n_list:
            for fat in fat_list:
                for density in density_list:
                    dag = DAG_geneator(index, n, fat, density, regularity, jump)
                    dag.run()
                    index += 1
        instance_name = '1-' + task_num + '-' + fat_num + '-0.7-0.5'
        dagTaskNum = int(instance_name.split('-')[1]) + 2
        generate_server_task(instance_name)
        return instance_name, dagTaskNum

    @staticmethod
    def get_taskSet(instance_name):  # 根据参数从6个App个选择一个，读取文件，获取任务特征
        pathname = '../dag/instance/' + instance_name + '/'
        DAG_path = pathname + 'DAG.txt'
        taskCPUCycleNumber = pd.read_csv(pathname + 'task_CPU_cycles_number.csv')
        taskInputDataSize = pd.read_csv(pathname + 'task_input_data_size.csv')
        taskOutputDataSize = pd.read_csv(pathname + 'task_output_data_size.csv')
        taskSet = []
        taskIndex = 0
        entryTask = None
        exitTask = None

        with open(DAG_path, 'r') as readFile:
            for line in readFile:
                task = Task()  # yy实例化Task，每一行是一个任务
                tem = taskCPUCycleNumber.values[taskIndex][0]
                task.c_i = float(taskCPUCycleNumber.values[taskIndex][0] * 10)
                task.d_i = float(taskInputDataSize.values[taskIndex][0])
                task.q_i = float(taskOutputDataSize.values[taskIndex][0])
                taskIndex += 1
                s = line.splitlines()
                s = s[0].split(':')
                predecessor = s[0]
                id = s[1]
                successor = s[2]
                if predecessor != '':
                    predecessor = predecessor.split(',')
                    for pt in predecessor:
                        task.preTaskSet.append(int(pt))  # TODO： 增加task实例而不是id
                else:
                    entryTask = int(id)
                task.id = int(id)
                if successor != '':
                    successor = successor.split(',')
                    for st in successor:
                        task.sucTaskSet.append(int(st))
                else:
                    task.sucTaskSet = []
                    exitTask = int(id)
                taskSet.append(task)
        return taskSet, entryTask, exitTask

    def setUp(self):  # 在env的reset中进行了调用，用于初始化设备
        instance_name, dagTaskNum = self.generateDAG(10, 0.7)
        taskSet, entryTask, exitTask = self.get_taskSet(instance_name)
        self.dag = DAG(instance_name, entryTask, exitTask, taskSet, dagTaskNum)
        for task in taskSet:
            self.global_sequence.append(task.id)
        logging.info("episode:%d - Time:%d - [Device-%d] -[instance name %s]" % (
            self.env.episode, self.env.clock, self.id, instance_name))

    def localProcess(self, dataSize):
        if dataSize == 0:
            return 0, 0
        cpuCyclePerBit = self.env.constants['avg_cpu_cycle_per_bit']
        effectiveCapacitanceCoefficient = self.env.constants['effective_capacitance_coefficient']
        local_t = dataSize * cpuCyclePerBit * 1.0 / self.cpuFrequency * 1e-9 * 1000
        # calculate z(n) unit J
        energyConsumptionPerCycle = effectiveCapacitanceCoefficient * math.pow(self.cpuFrequency * 1e9, 2)
        local_energy = dataSize * cpuCyclePerBit * energyConsumptionPerCycle * 1000
        return local_t, local_energy

    def offload(self, curr_task: Task):
        curr_task.status = TaskStatus.RUNNING
        if curr_task.offloading_rate < 0 or curr_task.offloading_rate > 1:
            logging.error("offloadingRate must ∈ [0,1]")

        if curr_task.offloading_rate == 0 and curr_task.server_id == 0 and curr_task.computing_f == 0:
            curr_task.T_trans_i = 0
            curr_task.E_trans_i = 0
            (curr_task.T_exec_local_i,
             curr_task.E_exec_local_i) = self.localProcess(curr_task.d_i)
            curr_task.T_exec_server_i = 0
            task_t = curr_task.T_i()
            task_e = curr_task.E_i()
            logging.info("episode:%d - Time:%d - [Device-%d] local process no offloading, T_i %d ms, E_i %d J" %
                         (self.env.episode, self.env.clock, self.id, task_t, task_e))

        if 0 < curr_task.offloading_rate <= 1 and curr_task.server_id is not None and curr_task.computing_f is not None:
            (curr_task.T_exec_local_i,
             curr_task.E_exec_local_i) = self.localProcess((1 - curr_task.offloading_rate) * curr_task.d_i)
            # T_trans_i, E_trans_i, T_exec_server_i需要sever填充
            logging.info("episode:%d - Time:%d - [Device-%d] offload to [Server-%d] with action offloadingRate=%f" % (
                self.env.episode, self.env.clock, self.id, curr_task.server_id, curr_task.offloading_rate))

            curr_task.device_id = self.id
            self.env.servers[curr_task.server_id - 1].acceptTask(curr_task)

    def serverFinishTask(self, curr_task: Task, T_trans_i, T_exec_server_i):
        curr_task.T_trans_i = T_trans_i
        curr_task.E_trans_i = self.En * T_trans_i
        curr_task.T_exec_server_i = T_exec_server_i
        curr_task.status = TaskStatus.FINISHED
        self.dag.update_status(curr_task)
        logging.info("episode:%d - Time:%d - [Device-%d] - [task-%d] - total processing consume %f ms, %f mj" % (
            self.env.episode, self.env.clock, self.id, curr_task.id, curr_task.T_i, curr_task.E_i))

    def reset(self):
        self.dag = None
        self.global_sequence = []
        self.metrics = np.zeros(shape=(self.env.T, 16))

    def updateState(self):
        self.newRequestedTaskLoad = self.generateNewTaskRequest()
        total = self.totalTaskLoad + self.newRequestedTaskLoad
        left = total if self.finished == False else total - self.currProcessedTaskLoad
        # device 下一个 time slot task queue的data size
        self.totalTaskLoad = min(left, self.maxLoad)
        # device 下一个 time slot 实际可以处理的data size
        self.currProcessedTaskLoad = min(self.maxTaskLoadPerTimeSlot, self.totalTaskLoad)
        logging.info("episode:%d - Time:%d - [Device-%d] totally need process %d bits data" % (
            self.env.episode, self.env.clock + 1, self.id, self.currProcessedTaskLoad))

        self.localProcessTime = 0
        self.localProcessEnergy = 0
        self.remoteProcessTime = 0
        self.remoteProcessEnergy = 0
        self.currProcessedDelay = 0
        self.currProcessedEnergy = 0
        self.finished = True

    # def collectMetric(self):
    #     self.metrics[self.env.clock - 1][0] = self.id
    #     self.metrics[self.env.clock - 1][2] = self.cpuFrequency
    #     self.metrics[self.env.clock - 1][3] = self.totalTaskLoad
    #     self.metrics[self.env.clock - 1][4] = self.currProcessedTaskLoad
    #     self.metrics[self.env.clock - 1][5] = self.currProcessedDelay
    #     self.metrics[self.env.clock - 1][6] = self.currProcessedEnergy
    #     self.metrics[self.env.clock - 1][7] = 1 if self.finished == False else 0
    #     self.metrics[self.env.clock - 1][8] = self.localProcessTime
    #     self.metrics[self.env.clock - 1][9] = self.localProcessEnergy
    #     self.metrics[self.env.clock - 1][10] = self.remoteProcessTime
    #     self.metrics[self.env.clock - 1][11] = self.remoteProcessEnergy
    #     self.metrics[self.env.clock - 1][12] = self.totalLocalProcessTime
    #     self.metrics[self.env.clock - 1][13] = self.totalLocalProcessEnergy
    #     self.metrics[self.env.clock - 1][14] = self.totalRemoteProcessTime
    #     self.metrics[self.env.clock - 1][15] = self.totalRemoteProcessEnergy
