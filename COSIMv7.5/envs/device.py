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
        self.cpuCyclePerBit = self.env.constants['avg_cpu_cycle_per_bit']
        self.effectiveCapacitanceCoefficient = self.env.constants['effective_capacitance_coefficient']
        self.dag = None
        self.global_sequence = []
        self.metrics = np.zeros(shape=(self.env.T, 16))
        self.time_slot = 100
        self.time_step = 1
        logging.info('init [Device-%d] finish' % self.id)

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
        self.dag = DAG(instance_name, taskSet, dagTaskNum)
        for task in taskSet:
            self.global_sequence.append(task.id)
        logging.info("episode:%d - Time:%d - [Device-%d] -[instance name %s]" % (
            self.env.episode, self.env.clock, self.id, instance_name))

    def totalLocalProcess(self, datasize):
        # calculate z(n) unit J
        energyConsumptionPerCycle = self.effectiveCapacitanceCoefficient * math.pow(self.cpuFrequency * 1e9, 2)
        t_local = datasize * self.cpuCyclePerBit * 1.0 / self.cpuFrequency * 1e-9 * 1000
        e_local = datasize * self.cpuCyclePerBit * energyConsumptionPerCycle * 1000
        return t_local, e_local

    def localProcess(self, task: Task):
        dataSize = (1 - task.offloading_rate) * task.d_i
        # calculate z(n) unit J
        energyConsumptionPerCycle = self.effectiveCapacitanceCoefficient * math.pow(self.cpuFrequency * 1e9, 2)
        task.T_exec_local_i = dataSize * self.cpuCyclePerBit * 1.0 / self.cpuFrequency * 1e-9 * 1000
        task.E_exec_local_i = dataSize * self.cpuCyclePerBit * energyConsumptionPerCycle * 1000
        increment = 1 if dataSize == 0 else math.ceil(task.T_exec_local_i / self.time_slot)
        task.expected_local_finish_step = task.start_step + increment - 1

    def offload(self, curr_task: Task):
        curr_task.status = TaskStatus.RUNNING
        curr_task.start_step = self.time_step
        if curr_task.offloading_rate < 0 or curr_task.offloading_rate > 1:
            logging.error("offloadingRate must ∈ [0,1]")
        if curr_task.offloading_rate == 0 and curr_task.server_id == 0 and curr_task.computing_f == 0:
            curr_task.T_trans_i = 0
            curr_task.E_trans_i = 0
            self.localProcess(curr_task)
            curr_task.T_exec_server_i = 0
            logging.info("episode:%d - Time:%d - [Device-%d] local process no offloading" % (self.env.episode, self.env.clock, self.id))
        if 0 < curr_task.offloading_rate <= 1 and curr_task.server_id is not None and curr_task.computing_f is not None:
            self.localProcess(curr_task)
            logging.info("episode:%d - Time:%d - [Device-%d] offload to [Server-%d] with action offloadingRate=%f" % (
                self.env.episode, self.env.clock, self.id, curr_task.server_id, curr_task.offloading_rate))
            curr_task.device_id = self.id
            self.env.servers[curr_task.server_id - 1].acceptTask(curr_task)

    def ifFinish(self, curr_task: Task, time_step):
        if curr_task.expected_local_finish_step == time_step:
            curr_task.status = TaskStatus.FINISHED
            logging.info("episode:%d - Time:%d - [Device-%d] , finish_step %d ms, E_i %d J" % (
                self.env.episode, self.env.clock, self.id, curr_task.finish_step(), curr_task.E_i()))
        else:
            curr_task.status = TaskStatus.RUNNING
            logging.info("server finished, local still running ")

    def reset(self):
        self.dag = None
        self.global_sequence = []
        self.metrics = np.zeros(shape=(self.env.T, 16))

    def updateState(self):
        self.dag.update_status()

