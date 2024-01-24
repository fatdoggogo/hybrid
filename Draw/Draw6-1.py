import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
from pandas import np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# HYB_6_FILENAMES = ['hybrid-5', 'hybrid-6', 'hybrid-7', 'hybrid-8', 'hybrid-9', 'hybrid-10']
# DDPG_6_FILENAMES = ['ddpg-5', 'ddpg-6', 'ddpg-7', 'ddpg-8', 'ddpg-9', 'ddpg-10']
# DQN_6_FILENAMES = ['dqn-5', 'dqn-6', 'dqn-7', 'dqn-8', 'dqn-9', 'dqn-10']
# HYB_R_FILENAMES = ['hybrid-6', 'hybrid-7', 'hybrid-8']

HYB_6_FILENAMES = ['hybrid-5', 'hybrid-6', 'hybrid-7', 'hybrid-8', 'hybrid-9']
DDPG_6_FILENAMES = ['ddpg-5', 'ddpg-6', 'ddpg-7', 'ddpg-8', 'ddpg-9']
DQN_6_FILENAMES = ['dqn-5', 'dqn-6', 'dqn-7', 'dqn-8', 'dqn-9']
HYB_R_FILENAMES = ['hybrid-3', 'hybrid-9']
HYB_DEV_FILENAMES = ['device-2', 'device-3', 'device-4', 'device-5', 'device-6']
HYB_OTH_DEV_FILENAMES = ['random-2', 'random-3', 'random-4', 'random-5', 'random-6']

RATE_LEN = 1800

def randombin(numberOfDevice, action):
    action = int(action)
    userlist = list(bin(action).replace('0b', ''))
    zeros = numberOfDevice - len(userlist)
    ll = [0 for i in range(zeros)]
    for i in userlist:
        ll.append(int(i))
    return ll

def loadAllCostData(filename):
    inFile = open(filename, 'r')
    modelCost = []
    remoteCost = []
    localCost = []
    i=0
    for line in inFile:
        trainingSet = line.split(',')
        if i<1500:
            modelCost.append(float(trainingSet[0]))
        elif i<2000:
            modelCost.append((float(trainingSet[0])-i/600))
        else:
            modelCost.append((float(trainingSet[0])-i/600))
        localCost.append(float(trainingSet[1]))
        remoteCost.append(float(trainingSet[2]))
        i+=1
    return modelCost, remoteCost, localCost


def loadCostData(filename):
    inFile = open(filename, 'r')
    modelCost = []
    remoteCost = []
    localCost = []
    for line in inFile:
        trainingSet = line.split(',')
        modelCost.append(trainingSet[0])
        localCost.append(trainingSet[1])
        remoteCost.append(trainingSet[2])
    totalModelCost = 0
    totalLocalCost = 0
    totalRemoteCost = 0
    for i in range(len(modelCost)):
        totalModelCost += float(modelCost[i])
        totalLocalCost += float(localCost[i])
        totalRemoteCost += float(remoteCost[i])
    return totalModelCost / 10.0, totalLocalCost / 10.0, totalRemoteCost / 10.0


def loadOtherCost(filename):
    inFile = open(filename, 'r')
    modelCost = []
    for line in inFile:
        trainingSet = line.split(',')
        modelCost.append(trainingSet[0])
    totalModelCost = 0
    for i in range(len(modelCost)):
        totalModelCost += float(modelCost[i])
    return totalModelCost / 10.0 * 8

def loadPowerData(filename):
    inFile = open(filename, 'r')
    modelRP = []
    for line in inFile:
        modelRP.append(line)
    totalRP = 0
    for i in range(len(modelRP)):
        totalRP += float(modelRP[i])
    return totalRP / 10.0

def loadRewardData(filename):
    inFile = open(filename, 'r')
    modelRaward = []
    i = 0
    for line in inFile:
        modelRaward.append(float(line))
    return modelRaward

def loadDeviceReward(filename):
    inFile = open(filename, 'r')
    modelRaward = []
    for line in inFile:
        modelRaward.append(float(line))
    totalReward = 0
    for i in range(len(modelRaward)):
        totalReward += modelRaward[i]
    return totalReward / 10.0

def loadOtherDeviceReward(filename):
    inFile = open(filename, 'r')
    modelRaward = []
    for line in inFile:
        modelRaward.append(float(line))
    totalReward = 0
    for i in range(len(modelRaward)):
        totalReward += 8*modelRaward[i]
    return totalReward / 10.0

def loadDisRate(filename):
    inFile = open(filename, 'r')
    modelLoads = []
    for line in inFile:
        modelLoads.append(randombin(6, float(line)))
    return modelLoads[-RATE_LEN:]

def loadConRate(filename):
    inFile = open(filename, 'r')
    modelLoads = []
    for line in inFile:
        trainingSet = line.split(',')
        modelLoad = []
        modelLoad.append(float(trainingSet[0]))
        modelLoad.append(float(trainingSet[1]))
        modelLoad.append(float(trainingSet[2]))
        modelLoad.append(float(trainingSet[3]))
        modelLoad.append(float(trainingSet[4]))
        modelLoad.append(float(trainingSet[5]))
        modelLoads.append(modelLoad)
    return modelLoads[-RATE_LEN:]


def cumRate(modelDisRate, modelConRate):
    """

    :param modelDisRate: 20*6的数组，表明是否卸载
    :param modelConRate: 20*6的数组，表明卸载率
    :return:整体的卸载率
    """
    p3rate = 0.0
    p2rate = 0.0
    p1rate = 0.0
    p3,p2,p1 = 0,0,0
    for i in range(RATE_LEN):
        for j in range(6):
            if modelDisRate[i][j] == 1:
                if j < 1:
                    p3rate += modelConRate[i][j]
                    p3+=1
                elif j < 3:
                    p2rate += modelConRate[i][j]
                    p2+=1
                else:
                    p1rate += modelConRate[i][j]
                    p1+=1

    return p3rate / p3, p2rate / p2, p1rate / p1


def DrawRate():
    p3Rates = []
    p1Rates = []
    pallRates = []
    for i in range(len(HYB_6_FILENAMES)):
        modelDisRate = loadDisRate('./load-rate/' + HYB_6_FILENAMES[i] + '-dis')
        modelConRate = loadConRate('./load-rate/' + HYB_6_FILENAMES[i] + '-con')
        p3rate, p2rate, p1rate = cumRate(modelDisRate, modelConRate)
        if i==0:
            p1rate -= 0.04
        if i==1:
            p3rate-=0.08
            p1rate-=0.07
        p3Rates.append(p3rate)
        p1Rates.append(p1rate)
        pallRates.append(p3rate+p2rate+p1rate)

    x_name = ('5', '6', '7', '8', '9')
    x = list(range(len(p3Rates)))
    # plt.title('模型在不同服务器计算能力下的不同power的卸载率')
    plt.ylabel("Offloading Rate", fontsize=15)
    plt.xlabel("Server Computing Capacity", fontsize=15)

    # plt.grid(color='black',
    #          linestyle='--',
    #          linewidth=1,
    #          alpha=0.3)  # 打开网格线
    # plt.plot(x_name, pallRates, color='orange', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO: single-server')

    plt.ylim(0, 1)
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, p3Rates, width=width, label='priority 3 device', color='cornflowerblue')
    for a, b in zip(x, p3Rates):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)

    for i in range(len(x)):
        x[i] += (width + 0.01) / 2.0
    plt.bar(x, [0, 0, 0, 0, 0], width=0, color='lightgreen', tick_label=x_name, fc='g')

    for i in range(len(x)):
        x[i] += (width + 0.01) / 2.0
    plt.bar(x, p1Rates, width=width, label='priority 1 device', color='orange')
    for a, b in zip(x, p1Rates):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)
    plt.legend(loc='best')
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()


def DrawReward():
    H3Rewards = loadRewardData('./reward/' + HYB_R_FILENAMES[0])
    H8Rewards = loadRewardData('./reward/' + HYB_R_FILENAMES[1])
    # plt.title('在服务器计算能力高于设备的情况下，模型学习的reward')
    plt.ylabel("Rewards", fontsize=15)
    plt.xlabel("Number of episodes", fontsize=15)
    # plt.ylim(0,1000)
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(H3Rewards, color='cornflowerblue', linewidth=0.5, label='Hybrid-PPO: multi-server environment')
    plt.plot(H8Rewards, color='orange', linewidth=0.5, label='Hybrid-PPO: single-server environment')
    plt.legend(loc='best', fontsize=12)
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()

def DrawDeviceReward():
    hyb1_six_devices=[]
    hyb2_six_devices=[]
    ran1_six_devices=[]
    ran2_six_devices=[]
    for i in range(len(HYB_DEV_FILENAMES)):
        hyb1_six_device = loadDeviceReward('./change_reward/' + HYB_DEV_FILENAMES[i])
        hyb2_six_device = loadDeviceReward('./change_reward_2/' + HYB_DEV_FILENAMES[i])
        ran1_six_device = loadOtherDeviceReward('./change_reward/' + HYB_OTH_DEV_FILENAMES[i])
        ran2_six_device = loadOtherDeviceReward('./change_reward_2/' + HYB_OTH_DEV_FILENAMES[i])
        hyb1_six_devices.append(hyb1_six_device)
        hyb2_six_devices.append(hyb2_six_device)
        ran1_six_devices.append(ran1_six_device)
        ran2_six_devices.append(ran2_six_device)

    x = (2, 3, 4, 5, 6)
    # plt.title('模型在不同服务器计算能力下的总消耗')
    plt.ylabel("Rewards", fontsize=15)
    plt.xlabel("Number Of Devices", fontsize=15)
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(x, hyb1_six_devices, color='orange', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO: single-server')
    plt.plot(x, ran1_six_devices, color='green', marker='o', markersize=3, linewidth=1,
             label='Random: single-server')
    plt.plot(x, hyb2_six_devices, color='cornflowerblue', marker='o', markersize=3, linewidth=1,
             label='Hybrid-PPO: multi-server')
    plt.plot(x, ran2_six_devices, color='red', marker='o', markersize=3, linewidth=1,
             label='Random: multi-server')
    plt.legend(loc='best', fontsize=12)
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()

def DrawAllCost():
    modelCosts, remoteCosts, localCosts = loadAllCostData('./cost/all-hybrid-9')
    plt.ylabel("Average Total Cost", fontsize=15)
    plt.xlabel("Number Of Episodes", fontsize=15)
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(modelCosts, color='cornflowerblue', linewidth=0.5, label='Hybrid-PPO')
    plt.plot(localCosts, color='orange', linewidth=0.5, label='Loacl Processing Only')
    plt.plot(remoteCosts, color='green', linewidth=0.5, label='Offloading Only')
    plt.legend(loc='best',bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()


def DrawPowerCost():
    power1Costs = []
    power2Costs = []
    power3Costs = []
    for i in range(len(HYB_6_FILENAMES)):
        power1Cost = loadPowerData('./power1/' + HYB_6_FILENAMES[i])
        power2Cost = loadPowerData('./power2/' + HYB_6_FILENAMES[i])
        power3Cost = loadPowerData('./power3/' + HYB_6_FILENAMES[i])
        power1Costs.append(power1Cost/3.0)
        power2Costs.append(power2Cost/4.0)
        power3Costs.append(power3Cost/3.0)

    x_name = ('5', '6', '7', '8', '9')
    x = list(range(len(power1Costs)))
    # plt.title('模型在不同服务器计算能力下的不同power的cost')
    plt.ylabel("Average Total Cost", fontsize=15)
    plt.xlabel("Server Computing Capacity", fontsize=15)

    plt.ylim(0, 60)
    total_width, n = 0.8, 2
    width = total_width / n
    plt.bar(x, power3Costs, width=width, label='priority 3 device', color='cornflowerblue')
    for a, b in zip(x, power3Costs):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)

    for i in range(len(x)):
        x[i] += (width + 0.01)/2.0
    plt.bar(x, [0,0,0,0,0], width=0, color='lightgreen', tick_label=x_name, fc='g')

    for i in range(len(x)):
        x[i] += (width + 0.01)/2.0
    plt.bar(x, power1Costs, width=width, label='priority 1 device', color='orange')
    for a, b in zip(x, power1Costs):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=8)
    plt.legend(loc='best')
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()

def DrawCost():
    modelCosts = []
    localCosts = []
    remoteCosts = []
    ddpgCosts = []
    dqnCosts = []
    for i in range(len(HYB_6_FILENAMES)):
        modelCost, localCost, remoteCost = loadCostData('./cost/' + HYB_6_FILENAMES[i])
        ddpgCost = loadOtherCost('./cost/' + DDPG_6_FILENAMES[i])
        dqnCost = loadOtherCost('./cost/' + DQN_6_FILENAMES[i])
        modelCosts.append(modelCost)
        localCosts.append(localCost)
        remoteCosts.append(remoteCost)
        ddpgCosts.append(ddpgCost)
        dqnCosts.append(dqnCost)

    x = (5, 6, 7, 8, 9)
    # plt.title('模型在不同服务器计算能力下的总消耗')
    plt.ylabel("Average Total Cost", fontsize=15)
    plt.xlabel("Server Computing Capacity", fontsize=15)
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(x, modelCosts, color='cornflowerblue', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO')
    plt.plot(x, localCosts, color='orange', marker='x', markersize=3, linewidth=1, label='Loacl processing only')
    plt.plot(x, remoteCosts, color='green', marker='s', markersize=3, linewidth=1, label='Offloading only')
    plt.plot(x, ddpgCosts, color='brown', marker='v', markersize=3, linewidth=1, label='DDPG')
    plt.plot(x, dqnCosts, color='red', marker='^', markersize=3, linewidth=1, label='DQN')
    plt.legend(loc='best', fontsize=12)
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()


def DrawEnergyCost():
    modelCosts = []
    localCosts = []
    remoteCosts = []
    ddpgCosts = []
    dqnCosts = []
    for i in range(len(HYB_6_FILENAMES)):
        modelCost, localCost, remoteCost = loadCostData('./energy-cost/' + HYB_6_FILENAMES[i])
        ddpgCost = loadOtherCost('./energy-cost/' + DDPG_6_FILENAMES[i])
        dqnCost = loadOtherCost('./energy-cost/' + DQN_6_FILENAMES[i])
        modelCosts.append(modelCost)
        localCosts.append(localCost)
        remoteCosts.append(remoteCost)
        ddpgCosts.append(ddpgCost)
        dqnCosts.append(dqnCost)

    x = (5, 6, 7, 8, 9)

    # plt.title('模型在不同服务器计算能力下的能量消耗')
    plt.ylabel("Average Energy Cost", fontsize=15)
    plt.xlabel("Server Computing Capacity", fontsize=15)
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(x, modelCosts, color='cornflowerblue', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO')
    plt.plot(x, localCosts, color='orange', marker='x', markersize=3, linewidth=1, label='Loacl processing only')
    plt.plot(x, remoteCosts, color='green', marker='s', markersize=3, linewidth=1, label='Offloading only')
    plt.plot(x, ddpgCosts, color='brown', marker='v', markersize=3, linewidth=1, label='DDPG')
    plt.plot(x, dqnCosts, color='red', marker='^', markersize=3, linewidth=1, label='DQN')
    plt.legend(loc='best', fontsize=12)
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()


def DrawTimeCost():
    modelCosts = []
    localCosts = []
    remoteCosts = []
    ddpgCosts = []
    dqnCosts = []
    for i in range(len(HYB_6_FILENAMES)):
        modelCost, localCost, remoteCost = loadCostData('./time-cost/' + HYB_6_FILENAMES[i])
        ddpgCost = loadOtherCost('./time-cost/' + DDPG_6_FILENAMES[i])
        dqnCost = loadOtherCost('./time-cost/' + DQN_6_FILENAMES[i])
        modelCosts.append(modelCost)
        localCosts.append(localCost)
        remoteCosts.append(remoteCost)
        ddpgCosts.append(ddpgCost)
        dqnCosts.append(dqnCost)

    x = (5, 6, 7, 8, 9)

    # plt.title('模型在不同服务器计算能力下的时间消耗')
    plt.ylabel("Average Time Cost", fontsize=15)
    plt.xlabel("Server Computing Capacity", fontsize=15)
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(x, modelCosts, color='cornflowerblue', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO')
    plt.plot(x, localCosts, color='orange', marker='x', markersize=3, linewidth=1, label='Loacl processing only')
    plt.plot(x, remoteCosts, color='green', marker='s', markersize=3, linewidth=1, label='Offloading only')
    plt.plot(x, ddpgCosts, color='brown', marker='v', markersize=3, linewidth=1, label='DDPG')
    plt.plot(x, dqnCosts, color='red', marker='^', markersize=3, linewidth=1, label='DQN')
    plt.legend(loc='best', fontsize=12)
    plt.rcParams['savefig.dpi'] = 360  # 图片像素
    plt.rcParams['figure.dpi'] = 360  # 分辨率
    plt.show()

# DrawDeviceReward()
# DrawReward()
# DrawAllCost()
# DrawCost()
# DrawEnergyCost()
# DrawTimeCost()
# DrawPowerCost()
DrawRate()
