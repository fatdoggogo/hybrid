import copy
import os
import random
import numpy as np
import pandas as pd

random.seed(2)


class DAG_geneator:
    def __init__(self, index, n, fat, density, regularity, jump):
        self.index = index
        self.n = n  # Number of nodes
        self.fat = fat  # Affect the height and width
        self.density = density  # determine the number of edges between two levels
        self.regularity = regularity  # determine the uniformity of the number of tasks in each level
        self.jump = jump  # indicate that an edge can go from level l to level l + jump

        self.nb_tasks_per_level = []
        self.DAG = {}

    def run(self):
        self.generateTasks()
        self.DAG = self.generateDependencies()
        self.add_start_end_task()
        self.save_to_file()

    def save_to_file(self):
        instance_name = str(self.index) + '-' + str(self.n) + '-' + str(self.fat) + '-' + str(self.density) + '-' + str(
            self.regularity)
        isExists = os.path.exists('../dag/instance/' + instance_name)
        if not isExists:
            os.makedirs('../dag/instance/' + instance_name)

        f_w = open('../dag/instance/' + instance_name + '/DAG.txt', 'w')
        key = sorted(self.DAG.keys())
        for k in key:
            pre = [str(e) for e in self.DAG[k]['pre']]  # 元素转换成字符串
            pre = ','.join(pre)
            suc = [str(e) for e in self.DAG[k]['suc']]  # 元素转换成字符串
            suc = ','.join(suc)
            result = pre + ':' + str(k) + ':' + suc + '\n'
            f_w.write(result)
        f_w.close()

    def add_start_end_task(self):
        # 添加虚拟的开始节点和结束节点
        DAG_temp = copy.deepcopy(self.DAG)
        self.DAG[0] = {}
        self.DAG[self.n + 1] = {}
        self.DAG[0]['pre'] = []
        self.DAG[0]['suc'] = []
        self.DAG[self.n + 1]['pre'] = []
        self.DAG[self.n + 1]['suc'] = []
        for idx in DAG_temp.keys():
            if self.DAG[idx]['pre'][0] == 0:
                self.DAG[0]['suc'].append(idx)
            if len(self.DAG[idx]['suc']) == 0:
                self.DAG[idx]['suc'].append(self.n + 1)
                self.DAG[self.n + 1]['pre'].append(idx)

    def generateDependencies(self):
        task_idx = 1  # 任务索引，从1开始，空出0以便后续添加一个开始任务
        nb_levels = len(self.nb_tasks_per_level)
        for i in range(self.nb_tasks_per_level[0]):  # 第一层
            self.DAG[task_idx] = {}
            self.DAG[task_idx]['pre'] = [0]
            self.DAG[task_idx]['suc'] = []
            task_idx += 1

        for i in range(1, nb_levels):
            crr_level_task_idx = []
            pre_level_task_idx = None
            for j in range(self.nb_tasks_per_level[i]):
                random.seed(i + j)
                nb_parents = min(1 + int(random.uniform(0, self.density * self.nb_tasks_per_level[i - 1])),
                                 self.nb_tasks_per_level[i - 1])
                self.DAG[task_idx] = {}
                self.DAG[task_idx]['pre'] = []
                self.DAG[task_idx]['suc'] = []
                crr_level_task_idx.append(task_idx)

                tmp1 = self.nb_tasks_per_level[:i - 1]
                tmp2 = self.nb_tasks_per_level[:i]
                tmp11 = list(np.arange(1, sum(tmp1) + 1))
                pre_level_task_idx = list(np.arange(1, sum(tmp2) + 1))

                for e in tmp11:
                    pre_level_task_idx.remove(e)

                avl_parents = copy.deepcopy(pre_level_task_idx)
                for k in range(nb_parents):
                    e = random.choice(avl_parents)
                    self.DAG[task_idx]['pre'].append(e)
                    self.DAG[e]['suc'].append(task_idx)
                    avl_parents.remove(e)

                task_idx += 1
            for idx in pre_level_task_idx:
                if len(self.DAG[idx]['suc']) == 0:
                    e = random.choice(crr_level_task_idx)
                    self.DAG[idx]['suc'].append(e)
                    self.DAG[e]['pre'].append(idx)

        return self.DAG

    # 生成每层的任务数
    def generateTasks(self):
        nb_tasks_per_level = int(np.exp(self.fat * np.log(self.n)))
        total_nb_tasks = 0
        while True:
            tmp = self.getIntRandomNumberAround(nb_tasks_per_level, 100. - 100. * self.regularity)
            if total_nb_tasks + tmp > self.n:
                tmp = self.n - total_nb_tasks
            self.nb_tasks_per_level.append(tmp)
            total_nb_tasks += tmp
            if total_nb_tasks >= self.n:
                break

    @staticmethod
    def getIntRandomNumberAround(x, perc):
        r = -perc + (2 * perc * random.random())
        new_int = max(1, int(x * (1.0 + r / 100.00)))
        return new_int


def generate_server_task(instance_name):
    taskNumber = 100
    serverNumber = 10
    fs = (2, 6)  # 服务器计算能力范围
    total_CP = []
    for i in range(taskNumber):
        temp = max(fs) - 1 * np.random.poisson(1.5, serverNumber)
        for j in range(serverNumber):
            if temp[j] == 0:
                temp[j] = 2  # 计算能力控制在（2,6）之间
        total_CP.append(temp)

    serverComputingCapability = pd.DataFrame(total_CP)
    serverComputingCapability.to_csv('../dag/instance/' + instance_name + '/server_computing_capability.csv',
                                     index=False)  # index=False表示不保存行名

    taskCPUCycleNumber = pd.DataFrame([round(random.uniform(0.1, 0.5), 2) for _ in range(taskNumber)])
    taskCPUCycleNumber.to_csv('../dag/instance/' + instance_name + '/task_CPU_cycles_number.csv', index=False)

    taskInputDataSize = pd.DataFrame([round(random.uniform(5000, 6000), 2) for _ in range(taskNumber)])
    taskInputDataSize.to_csv('../dag/instance/' + instance_name + '/task_input_data_size.csv', index=False)

    taskOutputDataSize = pd.DataFrame([round(random.uniform(500, 1000), 2) for _ in range(taskNumber)])
    taskOutputDataSize.to_csv('../dag/instance/' + instance_name + '/task_output_data_size.csv', index=False)


if __name__ == "__main__":
    # n_list = [15, 25, 35]      # DAG任务数量
    n_list = [3, 5]  # DAG任务数量
    fat_list = [0.4, 0.6]  # 决定DAG的宽度和高度，较小的fat值可构造较瘦高的图，值越大可构造较矮胖的图
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


if __name__ == "__main__":
    generate_server_task('1-3-0.4-0.7-0.5')
