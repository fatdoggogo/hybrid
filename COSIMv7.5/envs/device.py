import json
import logging

import torch

from task import *

import random

# 设置种子
random.seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.manual_seed(42)


class Device:
    def __init__(self, config, env):
        self.env = env  # 设备所在的环境
        self.id = config['cnt']
        self.cpuFrequency = np.random.uniform(1.5, 2.5)
        self.availableCpuFreq = self.cpuFrequency
        self.BW = config['BW']  # server向device传输结果数据时可分配的带宽 unit MHZ
        self.En = config['energy_consume_per_time_slot']  # 设备单位时间传输数据需要的能量 unit uJ
        self.transPower = config['transmission_power']
        self.channelGain = config['channel_gain']
        self.channelNoise = config['channel_noise']
        self.density = config['density']
        self.regularity = config['regularity']
        self.cpuCyclePerBit = 750
        self.effectiveCapacitanceCoefficient = 1e-23
        self.dag = None
        self.time_slot = 100

    @staticmethod
    def generateDAG(dag_index):
        # density决定两层之间边的数量，值越小边越少; regularity决定每层任务数的差异性
        task_num = random.randint(8, 12)
        fat_num = random.choice([0.4, 0.5, 0.6])
        dag = DAG_geneator(dag_index, task_num, fat_num, 0.5, 0.5, jump=1)
        instance_name = str(dag_index) + '-' + str(task_num) + '-' + str(fat_num) + '-' + str(0.5) + '-' + str(0.5)
        dag.run(instance_name)
        dagTaskNum = int(instance_name.split('-')[1]) + 2
        generate_server_task(instance_name)
        return instance_name, dagTaskNum

    @staticmethod
    def get_taskSet(instance_name):  # 根据参数从6个App选择一个，读取文件，获取任务特征
        pathname = '../dag/instance/' + instance_name + '/'
        DAG_path = pathname + 'DAG.txt'
        taskInputDataSize = pd.read_csv(pathname + 'task_input_data_size.csv')
        taskOutputDataSize = pd.read_csv(pathname + 'task_output_data_size.csv')
        taskSet = []
        entryTask = None
        exitTask = None

        with open(DAG_path, 'r') as readFile:
            for line in readFile:
                s = line.splitlines()
                s = s[0].split(':')
                if int(s[1]) >= len(taskSet):
                    task = Task()
                    task.id = int(s[1])
                    taskSet.append(task)
                else:
                    task = taskSet[int(s[1])]
                task.d_i = float(taskInputDataSize.values[task.id][0])
                task.q_i = float(taskOutputDataSize.values[task.id][0])
                predecessor = s[0]
                successor = s[2]
                if predecessor != '':
                    predecessor = predecessor.split(',')
                    for pt in predecessor:
                        pre_task = taskSet[int(pt)]
                        task.preTaskSet.append(pre_task)
                else:
                    entryTask = task
                if successor != '':
                    successor = successor.split(',')
                    for st in successor:
                        post_task = Task()
                        post_task.id = int(st)
                        if post_task.id == len(taskSet):
                            taskSet.append(post_task)
                        task.sucTaskSet.append(post_task)
                else:
                    task.sucTaskSet = []
                    exitTask = task
        return taskSet, entryTask, exitTask

    def setUp(self):
        instance_name, dagTaskNum = self.generateDAG(self.id)
        taskSet, entryTask, exitTask = self.get_taskSet(instance_name)
        self.dag = DAG(instance_name, taskSet)

    def totalLocalProcess(self, datasize):
        energyConsumptionPerCycle = self.effectiveCapacitanceCoefficient * math.pow(self.cpuFrequency * 1e9, 2)
        t_local = datasize * self.cpuCyclePerBit * 1.0 / (self.cpuFrequency * 1e9) * 1e5
        e_local = datasize * self.cpuCyclePerBit * energyConsumptionPerCycle
        return t_local, e_local

    def localProcess(self, task: Task):
        dataSize = (1 - task.offloading_rate) * task.d_i
        # calculate z(n) unit uJ
        energyConsumptionPerCycle = self.effectiveCapacitanceCoefficient * math.pow(self.cpuFrequency * 1e9, 2)
        task.T_exec_local_i = dataSize * self.cpuCyclePerBit * 1.0 / (self.cpuFrequency * 1e9) * 1e5
        task.E_exec_local_i = dataSize * self.cpuCyclePerBit * energyConsumptionPerCycle
        increment = 1 if dataSize == 0 else math.ceil(task.T_exec_local_i / self.time_slot)
        task.expected_local_finish_step = task.start_step + increment - 1

    @staticmethod
    def ifTaskFinish(curr_task: Task, time_step):
        if curr_task.expected_local_finish_step <= time_step:
            curr_task.local_finished = 1
            curr_task.status = TaskStatus.FINISHED
        else:
            curr_task.status = TaskStatus.RUNNING

    def offload(self, curr_task: Task, time_step):
        if curr_task.status == TaskStatus.NOT_SCHEDULED:
            curr_task.status = TaskStatus.RUNNING
            curr_task.start_step = time_step
            curr_task.device_id = self.id

            if curr_task.offloading_rate == 0 or curr_task.server_id == 0 or curr_task.computing_f == 0:
                curr_task.T_trans_i = 0
                curr_task.E_trans_i = 0
                curr_task.offloading_rate = 0
                curr_task.server_id = 0
                curr_task.computing_f = 0
                curr_task.bw = 0
                self.localProcess(curr_task)
                curr_task.T_exec_server_i = 0
                curr_task.server_finished = 1
                self.ifTaskFinish(curr_task, time_step)
                logging.info("episode: %d - Time: %d - Device: %d local process no offloading",
                             self.env.episode, time_step, self.id)

            elif 0 < curr_task.offloading_rate <= 1 and curr_task.server_id > 0 and curr_task.computing_f > 0:
                self.localProcess(curr_task)
                logging.info("episode: %d - Time: %d - Device: %d - Server: %d with action offloadingRate: %f",
                             self.env.episode, time_step, self.id, curr_task.server_id, curr_task.offloading_rate)
                selected_server = next((server for server in self.env.servers if server.id == curr_task.server_id),
                                       None)
                selected_server.acceptTask(curr_task)

        else:
            logging.info("episode: %d - Time: %d - Device: %d none new task, still processing",
                         self.env.episode, time_step, self.id)
            if curr_task.offloading_rate == 0 or curr_task.expected_sever_finish_step <= time_step:
                self.ifTaskFinish(curr_task, time_step)

    def reset(self):
        self.dag = None

    def updateState(self):
        if self.dag.currentTask.status == TaskStatus.RUNNING:
            self.availableCpuFreq = np.random.uniform(0.01, 0.05)
        elif self.dag.currentTask.status == TaskStatus.FINISHED:
            self.cpuFrequency = np.random.uniform(1.5, 2.5)
            self.availableCpuFreq = self.cpuFrequency
        self.dag.update_status()


if __name__ == "__main__":
    configPath = '../config/config.json'
    with open(configPath, encoding='utf-8') as f:
        configStr = f.read()
        config = json.loads(configStr)
    deviceConfigs = config['device_configs']
    var = deviceConfigs[0]
    from env import *

    env = Env
    device = Device(deviceConfigs[0], env)
    device.setUp()
