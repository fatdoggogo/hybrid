from typing import List
from enum import Enum
from graph_generator import *

np.random.seed(3)


class TaskStatus(Enum):
    NOT_SCHEDULED = 0
    RUNNING = 1
    FINISHED = 2


class Task:
    def __init__(self):
        self.id = None  # id 就是优先级1最大
        self.preTaskSet: List[Task] = []  # The set of predecessor task (element is Task class).
        self.sucTaskSet: List[Task] = []  # The set of successor task (element is Task class).

        self.c_i = None  # yy任务所需CPU周期
        self.d_i = None  # yy任务输入数据大小
        self.q_i = None  # yy任务输出数据大小

        # 填充
        self.offloading_rate = None
        self.server_id = None
        self.computing_f = None

        self.T_trans_i = None  # d+qj+qi无线传输时间
        self.E_trans_i = None  # 传输任务的能耗=En * T_trans_i

        self.T_exec_local_i = None  # 本地执行任务的时间
        self.E_exec_local_i = None  # 本地执行任务的能耗 = Pn * (1-offloading_rate) * d_i * local_cycle

        self.T_exec_server_i = None  # 服务器执行的时间
        self.status = TaskStatus.NOT_SCHEDULED
        self.start_step = None  # 1...N
        self.expected_local_finish_step = None
        self.expected_sever_finish_step = None
        self.device_id = None

    @property
    def upload_data_sum(self):
        return sum(task.q_i for task in self.preTaskSet) + self.offloading_rate * self.d_i

    @property
    def process_data(self):
        return self.offloading_rate * self.d_i

    @property
    def download_data_sum(self):
        return self.offloading_rate * self.q_i

    @property
    def T_server_i(self):
        return self.T_exec_server_i + self.T_trans_i

    @property
    def T_i(self):  # 任务总执行的时间=max{T_local_i,T_server_i}
        return max(self.T_exec_local_i, self.T_server_i)

    @property
    def E_i(self):  # 任务总消耗=E_trans_i + E_exec_local_i
        return self.E_trans_i + self.E_exec_local_i


class DAG:
    def __init__(self, instance_name, entryTask, exitTask, tasks, dagTaskNum):
        self.instance_name = instance_name
        self.entryTask = entryTask
        self.exitTask = exitTask
        self.tasks: List[Task] = tasks
        self.dagTaskNum = dagTaskNum
        self.currentTask: Task = self.tasks[0]
        self.is_finished = False
        self.t_dag = 0
        self.e_dag = 0

    def update_status(self):
        if self.currentTask.status == TaskStatus.FINISHED:
            self.t_dag = self.t_dag + self.currentTask.T_i
            self.e_dag = self.e_dag + self.currentTask.E_i
            self.currentTask = self.tasks[self.currentTask.id]  # next task
            if all(task.status == TaskStatus.FINISHED for task in self.tasks):
                self.is_finished = True
