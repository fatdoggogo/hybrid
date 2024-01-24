from device import *


class Server(object):
    def __init__(self, id, config, env):
        self.id = id
        self.env = env
        self.BW = config['BW']  # device向server 上传数据时可分配的带宽 unit MHZ
        self.maxCpuFrequency = config['max_cpu_frequency']  # f(m,max) unit GHZ
        self.transPower = config['transmission_power']
        self.channelGain = config['channel_gain']
        self.channelNoise = self.env.constants['gaussian_channel_noise']
        self.cpuCyclePerBit = self.env.constants['avg_cpu_cycle_per_bit']
        self.tasks: List[Task] = []
        logging.info('init [Server-%d] BW:%f MHZ,maxCpuFrequency: %f GHZ' % (
            self.id, self.BW, self.maxCpuFrequency))

    def updateState(self):
        """
        用于环境更新',list中所有未完成task current_step加1
        :return:
        """
        self.tasks = []

    def acceptTask(self, task: Task):
        self.tasks.append(task)

    def process(self, timeslot, time_step):
        logging.info('episode:%d - Time:%d -[Server-%d] start processing' % (self.env.episode, self.env.clock, self.id))
        f_sum = sum(task.computing_f for task in self.tasks)
        if f_sum > self.maxCpuFrequency:
            logging.error("当前任务已经超过server%d的计算能力", self.id)
        for task in self.tasks:
            device = self.env.devices[task.device_id - 1]
            BW = self.BW * 1.0 / len(self.tasks)
            uploadRate = BW * 1e6 * math.log2(1 + (device.transPower * device.channelGain / (BW * device.channelNoise)))
            downloadRate = device.BW*1e6*math.log2(1+(self.transPower*self.channelGain/(device.BW * self.channelNoise)))
            uploadTime = task.upload_data_sum * 1.0 / uploadRate * 1000
            cpuFrequency = self.maxCpuFrequency * task.computing_f * 1.0 / f_sum
            processTime = task.process_data * self.cpuCyclePerBit * 1.0 / cpuFrequency * 1e-9 * 1000
            downloadTime = task.download_data_sum * 1.0 / downloadRate * 1000
            device.serverFinishTask(task, uploadTime+downloadTime, processTime)
