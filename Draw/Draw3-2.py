import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

HYB_3_FILENAMES = ['hybrid-1', 'hybrid-1.5', 'hybrid-2', 'hybrid-2.5', 'hybrid-3']
DDPG_3_FILENAMES = ['ddpg-1', 'ddpg-1.5', 'ddpg-2', 'ddpg-2.5', 'ddpg-3']
DQN_3_FILENAMES = ['dqn-1', 'dqn-1.5', 'dqn-2', 'dqn-2.5', 'dqn-3']

def loadCostData(flieName):
    inFile = open(flieName, 'r')
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
    return totalModelCost / 80.0, totalLocalCost / 80.0, totalRemoteCost / 80.0

def loadOtherCost(filename):
    inFile = open(filename, 'r')
    modelCost = []
    for line in inFile:
        trainingSet = line.split(',')
        modelCost.append(trainingSet[0])
    totalModelCost = 0
    for i in range(len(modelCost)):
        totalModelCost += float(modelCost[i])
    return totalModelCost / 10.0


def loadRPData(filename):
    inFile = open(filename, 'r')
    modelRP = []
    for line in inFile:
        modelRP.append(line)
    totalRP = 0
    for i in range(len(modelRP)):
        totalRP += float(modelRP[i])
    return totalRP / 10.0


def DrawPowerCost():
    power1Costs = []
    power2Costs = []
    power3Costs = []
    for i in range(len(HYB_3_FILENAMES)):
        power1Cost = loadRPData('./2-power1/' + HYB_3_FILENAMES[i])
        power2Cost = loadRPData('./2-power2/' + HYB_3_FILENAMES[i])
        power3Cost = loadRPData('./2-power3/' + HYB_3_FILENAMES[i])
        power1Costs.append(power1Cost /1.0)
        power2Costs.append(power2Cost/2.0)
        power3Costs.append(power3Cost/3.0)
    x = (1, 1.5, 2, 2.5, 3)
    plt.title('模型在不同服务器计算能力下的不同power的cost')
    plt.ylabel("cost")
    plt.xlabel("服务器计算能力")
    plt.grid(color='black',
             linestyle='--',
             linewidth=1,
             alpha=0.3)  # 打开网格线
    plt.plot(x, power3Costs, color='cornflowerblue', marker='o', markersize=2, linewidth=1)
    plt.plot(x, power2Costs, color='green', marker='o', markersize=2, linewidth=1)
    plt.plot(x, power1Costs, color='red', marker='o', markersize=2, linewidth=1)
    plt.show()


def DrawCost():
    modelCosts = []
    localCosts = []
    remoteCosts = []
    ddpgCosts = []
    dqnCosts = []
    for i in range(len(HYB_3_FILENAMES)):
        modelCost, localCost, remoteCost = loadCostData('./2-cost/' + HYB_3_FILENAMES[i])
        ddpgCost = loadOtherCost('./2-cost/' + DDPG_3_FILENAMES[i])
        dqnCost = loadOtherCost('./2-cost/' + DQN_3_FILENAMES[i])
        modelCosts.append(modelCost)
        localCosts.append(localCost)
        remoteCosts.append(remoteCost)
        ddpgCosts.append(ddpgCost)
        dqnCosts.append(dqnCost)

    print(modelCosts, ddpgCosts, dqnCosts, remoteCosts, localCosts)

    # x = (1, 1.5, 2, 2.5, 3)
    # # plt.title('模型在不同服务器计算能力下的总消耗')
    # plt.ylabel("Average Total Cost", fontsize=15)
    # plt.xlabel("Server Computing Capacity", fontsize=15)
    # plt.grid(color='black',
    #          linestyle='--',
    #          linewidth=1,
    #          alpha=0.3)  # 打开网格线
    # plt.plot(x, modelCosts, color='cornflowerblue', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO')
    # plt.plot(x, localCosts, color='orange', marker='x', markersize=3, linewidth=1, label='Loacl processing only')
    # plt.plot(x, remoteCosts, color='green', marker='s', markersize=3, linewidth=1, label='Offloading only')
    # plt.plot(x, ddpgCosts, color='brown', marker='v', markersize=3, linewidth=1, label='DDPG')
    # plt.plot(x, dqnCosts, color='red', marker='^', markersize=3, linewidth=1, label='DQN')
    # plt.legend(loc='best', fontsize=12)
    # plt.rcParams['savefig.dpi'] = 360  # 图片像素
    # plt.rcParams['figure.dpi'] = 360  # 分辨率
    # plt.show()

def DrawEnergyCost():
    modelCosts = []
    localCosts = []
    remoteCosts = []
    ddpgCosts = []
    dqnCosts = []
    for i in range(len(HYB_3_FILENAMES)):
        modelCost, localCost, remoteCost = loadCostData('./2-energy-cost/' + HYB_3_FILENAMES[i])
        ddpgCost = loadOtherCost('./2-energy-cost/' + DDPG_3_FILENAMES[i])
        dqnCost = loadOtherCost('./2-energy-cost/' + DQN_3_FILENAMES[i])
        modelCosts.append(modelCost)
        localCosts.append(localCost)
        remoteCosts.append(remoteCost)
        ddpgCosts.append(ddpgCost)
        dqnCosts.append(dqnCost)

    print(modelCosts, ddpgCosts, dqnCosts, remoteCosts, localCosts)

    # x = (1, 1.5, 2, 2.5, 3)
    # # plt.title('模型在不同服务器计算能力下的能量消耗')
    # plt.ylabel("Average Energy Cost", fontsize=15)
    # plt.xlabel("Server Computing Capacity", fontsize=15)
    # plt.grid(color='black',
    #          linestyle='--',
    #          linewidth=1,
    #          alpha=0.3)  # 打开网格线
    # plt.plot(x, modelCosts, color='cornflowerblue', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO')
    # plt.plot(x, localCosts, color='orange', marker='x', markersize=3, linewidth=1, label='Loacl processing only')
    # plt.plot(x, remoteCosts, color='green', marker='s', markersize=3, linewidth=1, label='Offloading only')
    # plt.plot(x, ddpgCosts, color='brown', marker='v', markersize=3, linewidth=1, label='DDPG')
    # plt.plot(x, dqnCosts, color='red', marker='^', markersize=3, linewidth=1, label='DQN')
    # plt.legend(loc='best', bbox_to_anchor=(1, 0.4), fontsize=12)
    # plt.rcParams['savefig.dpi'] = 360  # 图片像素
    # plt.rcParams['figure.dpi'] = 360  # 分辨率
    # plt.show()

def DrawTimeCost():
    modelCosts = []
    localCosts = []
    remoteCosts = []
    ddpgCosts = []
    dqnCosts = []
    for i in range(len(HYB_3_FILENAMES)):
        modelCost, localCost, remoteCost = loadCostData('./2-time-cost/' + HYB_3_FILENAMES[i])
        ddpgCost = loadOtherCost('./2-time-cost/' + DDPG_3_FILENAMES[i])
        dqnCost = loadOtherCost('./2-time-cost/' + DQN_3_FILENAMES[i])
        modelCosts.append(modelCost)
        localCosts.append(localCost)
        remoteCosts.append(remoteCost)
        ddpgCosts.append(ddpgCost)
        dqnCosts.append(dqnCost)

    print(modelCosts, ddpgCosts, dqnCosts, remoteCosts, localCosts)

    # x = (1, 1.5, 2, 2.5, 3)
    # # plt.title('模型在不同服务器计算能力下的时间消耗')
    # plt.ylabel("Average Time Cost", fontsize=15)
    # plt.xlabel("Server Computing Capacity", fontsize=15)
    # plt.grid(color='black',
    #          linestyle='--',
    #          linewidth=1,
    #          alpha=0.3)  # 打开网格线
    # plt.plot(x, modelCosts, color='cornflowerblue', marker='o', markersize=3, linewidth=1, label='Hybrid-PPO')
    # plt.plot(x, localCosts, color='orange', marker='x', markersize=3, linewidth=1, label='Loacl processing only')
    # plt.plot(x, remoteCosts, color='green', marker='s', markersize=3, linewidth=1, label='Offloading only')
    # plt.plot(x, ddpgCosts, color='brown', marker='v', markersize=3, linewidth=1, label='DDPG')
    # plt.plot(x, dqnCosts, color='red', marker='^', markersize=3, linewidth=1, label='DQN')
    # plt.legend(loc='best', fontsize=12)
    # plt.rcParams['savefig.dpi'] = 360  # 图片像素
    # plt.rcParams['figure.dpi'] = 360  # 分辨率
    # plt.show()

# DrawCost()
DrawEnergyCost()
# DrawTimeCost()
# DrawPowerCost()