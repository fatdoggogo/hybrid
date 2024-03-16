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
        self.start_step = None
        self.finish_step = None  # 1...N
        self.expected_local_finish_step = None
        self.expected_sever_finish_step = None
        self.device_id = None
        self.expected_trans_finish_step = None
        self.time_slot = 1000

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
        expected_local_finish = 0 if self.expected_local_finish_step is None else self.expected_local_finish_step
        expected_server_finish = 0 if self.expected_sever_finish_step is None else self.expected_sever_finish_step
        return max(expected_local_finish, expected_server_finish) * self.time_slot

    @property
    def E_i(self):  # 任务总消耗=E_trans_i + E_exec_local_i
        self.E_trans_i = 0 if self.E_trans_i is None else self.E_trans_i
        self.E_exec_local_i = 0 if self.E_exec_local_i is None else self.E_exec_local_i
        return self.E_trans_i + self.E_exec_local_i


class DAG:
    def __init__(self, instance_name, tasks):
        self.instance_name = instance_name
        self.tasks: List[Task] = tasks
        self.currentTask: Task = self.tasks[0]
        self.is_finished = False
        self.t_dag = 0
        self.e_dag = 0
        self.finished_timestep = None
        self.time_slot = 1000

    def update_status(self):
        if self.currentTask.status == TaskStatus.FINISHED:
            if not all(task.status == TaskStatus.FINISHED for task in self.tasks):
                if self.currentTask == self.tasks[-1]:
                    self.t_dag = self.t_dag + max(self.currentTask.T_exec_local_i, self.currentTask.T_exec_server_i)
                else:
                    self.t_dag = self.t_dag + self.currentTask.T_i
                self.e_dag = self.e_dag + self.currentTask.E_i
                self.currentTask = self.tasks[self.currentTask.id + 1]

            else:
                self.is_finished = True
                self.finished_timestep = self.tasks[-1].finish_step
                self.currentTask = None
