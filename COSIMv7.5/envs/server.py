from device import *
import math
import random

# 设置种子
random.seed(42)


class Server(object):
    def __init__(self, conf, environ):
        self.id = conf['cnt']
        self.env = environ
        self.bandwidth = conf['bandwidth']  # device向server 上传数据时可分配的带宽 unit MHZ
        self.availableBW = self.bandwidth
        self.maxCpuFrequency = conf['max_cpu_frequency']  # f(m,max) unit GHZ
        self.availableFreq = self.maxCpuFrequency
        self.transPower = conf['transmission_power']
        self.channelGain = conf['channel_gain']
        self.channelNoise = conf['gaussian_channel_noise']
        self.cpuCyclePerBit = conf['avg_cpu_cycle_per_bit']
        self.tasks: List[Task] = []

    def acceptTask(self, task: Task):
        self.tasks.append(task)
        task.server_id = self.id

    def process(self, timeslot, time_step):
        if len(self.tasks) > 0:
            new_tasks = [task for task in self.tasks if task.expected_sever_finish_step is None]
            if len(new_tasks) > 0:
                freq_sum = sum(task.computing_f for task in new_tasks)
                for task in new_tasks:
                    dev = self.env.devices[task.device_id - 1]
                    task.bw = self.availableBW * 1.0 / len(new_tasks)
                    uploadRate = task.bw * 1e6 * math.log2(
                        1 + (dev.transPower * dev.channelGain / (task.bw * 1e6 * dev.channelNoise * 1e-9)))  # 25-50Mb/s
                    downloadRate = dev.BW * 1e6 * math.log2(
                        1 + (self.transPower * self.channelGain / (dev.BW * 1e6 * self.channelNoise * 1e-9)))
                    uploadTime = 1e6 * task.upload_data_sum * 1.0 / uploadRate  # 换算成微秒us 100-200us， timeslot为100us
                    # calculate BW
                    task.expected_trans_finish_step = math.ceil(uploadTime / timeslot) + task.start_step - 1
                    task.computing_f = task.computing_f * 1.0 / freq_sum * (self.availableFreq + 0.03)
                    processTime = task.process_data * self.cpuCyclePerBit * 1.0 / (task.computing_f * 1e9) * 1e5
                    downloadTime = task.download_data_sum * 1.0 / downloadRate * 1e6
                    total_t = uploadTime + downloadTime + processTime
                    task.expected_sever_finish_step = math.ceil(total_t / timeslot) + task.start_step - 1

                    task.T_trans_i = uploadTime + downloadTime
                    task.E_trans_i = dev.En * task.T_trans_i
                    task.T_exec_server_i = processTime

            not_finished_trans_tasks = [task for task in self.tasks if task.expected_trans_finish_step > time_step]
            finished_tasks = [task for task in self.tasks if task.expected_sever_finish_step <= time_step]
            for task in finished_tasks:
                task.server_finished = 1
                self.env.devices[task.device_id - 1].ifTaskFinish(task, time_step)

            self.tasks = [task for task in self.tasks if task not in finished_tasks]
            self.availableFreq = self.maxCpuFrequency - sum(task.computing_f for task in self.tasks)
            self.availableBW = self.bandwidth - sum(task.bw for task in not_finished_trans_tasks)

    def updateState(self, time_step):
        """
        用于环境更新',list中所有未完成task current_step加1,当前的cpu_freq
        # TODO://BW STATE update
        :return:
        """

    def reset(self):
        self.tasks: List[Task] = []
        self.availableBW = self.bandwidth
        self.availableFreq = self.maxCpuFrequency
