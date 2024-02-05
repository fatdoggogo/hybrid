from device import *
import math


class Server(object):
    def __init__(self, id, config, env):
        self.id = id
        self.env = env
        self.BW = config['BW']  # device向server 上传数据时可分配的带宽 unit MHZ
        self.availableBW = self.BW
        self.maxCpuFrequency = config['max_cpu_frequency']  # f(m,max) unit GHZ
        self.availableFreq = self.maxCpuFrequency
        self.transPower = config['transmission_power']
        self.channelGain = config['channel_gain']
        self.channelNoise = self.env.constants['gaussian_channel_noise']
        self.cpuCyclePerBit = self.env.constants['avg_cpu_cycle_per_bit']
        self.tasks: List[Task] = []
        logging.info('init [Server-%d] BW:%f MHZ,maxCpuFrequency: %f GHZ' % (
            self.id, self.BW, self.maxCpuFrequency))

    def acceptTask(self, task: Task):
        f_sum = sum(task.computing_f for task in self.tasks)
        if f_sum + task.computing_f > self.maxCpuFrequency:
            logging.info("已经到达最大cpu freq，不能再offload到server%d上", self.id)
            return False
        if self.BW < 100:
            logging.info("已经到达最大带宽，不能再offload到server%d上", self.id)
            return False
        else:
            self.tasks.append(task)
            task.server_id = self.id
            return True

    def process(self, timeslot, time_step):
        logging.info('episode:%d - Time:%d -[Server-%d] start processing' % (self.env.episode, self.env.clock, self.id))
        new_tasks = [task for task in self.tasks if task.expected_sever_finish_step is None]
        freqsum = sum(task.computing_f for task in new_tasks)
        for task in new_tasks:
            device = self.env.devices[task.device_id - 1]
            occupied_bw = self.availableBW * 1.0 / len(new_tasks)
            uploadRate = occupied_bw * 1e6 * math.log2(
                1 + (device.transPower * device.channelGain / (occupied_bw * device.channelNoise)))
            downloadRate = device.BW * 1e6 * math.log2(
                1 + (self.transPower * self.channelGain / (device.BW * self.channelNoise)))
            uploadTime = task.upload_data_sum * 1.0 / uploadRate * 1000
            # calculate BW
            task.expected_trans_finish_step = math.ceil(uploadTime / timeslot) + task.start_step - 1
            cpuFreq = task.computing_f * 1.0 / freqsum * self.availableFreq
            processTime = task.process_data * self.cpuCyclePerBit * 1.0 / cpuFreq * 1e-9 * 1000
            downloadTime = task.download_data_sum * 1.0 / downloadRate * 1000
            total_t = uploadTime + downloadTime + processTime
            task.expected_sever_finish_step = math.ceil(total_t / timeslot) + task.start_step - 1
            task.T_trans_i = uploadTime + downloadTime
            task.E_trans_i = device.En * task.T_trans_i
            task.T_exec_server_i = processTime

        not_finished_trans_tasks = [task for task in self.tasks if task.expected_trans_finish_step > time_step]
        finished_tasks = [task for task in self.tasks if task.expected_sever_finish_step > time_step]
        for task in finished_tasks:
            device = self.env.devices[task.device_id - 1]
            device.ifFinish(task, time_step)

        self.tasks = [task for task in self.tasks if task not in finished_tasks]
        self.availableFreq = self.maxCpuFrequency - sum(task.computing_f for task in self.tasks)
        self.availableBW = self.BW - sum(task.computing_f for task in not_finished_trans_tasks)

    def updateState(self, time_step):
        """
        用于环境更新',list中所有未完成task current_step加1,当前的cpu_freq
        # TODO://BW STATE update
        :return:
        """

    def reset(self):
        self.tasks: List[Task] = []
        self.availableBW = self.BW
        self.availableFreq = self.maxCpuFrequency
