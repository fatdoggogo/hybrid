import numpy as np
import matplotlib.pyplot as plt
import os
import json
currentPath = os.path.abspath(__file__)
parentDir = os.path.abspath(os.path.dirname(currentPath) + os.path.sep + ".")
configPath = parentDir + '/config.json'
# 读取json配置文件
with open(configPath,encoding='utf-8') as f:
    configStr = f.read()
    config = json.loads(configStr)
episodes = config['episodes']
timeSlots = config['time_slots']
algorithmConfig = config['algorithm_configs']
deviceConfigs = config['device_configs']
picConfigs = config['pic_configs']
deviceNumber = 0
for config in deviceConfigs:
    deviceNumber = deviceNumber + config['cnt']
def drawCost(algorithmDir,type):
    pic0Config = picConfigs[0]
    unit = pic0Config['unit']
    picSize = pic0Config['pic_size']
    imgType = pic0Config['type']
    if type == 'total':
        arr = np.loadtxt(algorithmDir+'/metrics/cost.csv',dtype=float,delimiter=',',unpack=False).T
        title = "Average weighted cost"
        saveFile = algorithmDir+'/images/costs.jpg'
    elif type =='time':
        arr = np.loadtxt(algorithmDir+'/metrics/time-cost.csv',dtype=float,delimiter=',',unpack=False).T
        title = "Average time cost"
        saveFile = algorithmDir+'/images/time-costs.jpg'
    elif type == 'energy':
        arr = np.loadtxt(algorithmDir+'/metrics/energy-cost.csv',dtype=float,delimiter=',',unpack=False).T
        title = "Average energy cost"
        saveFile = algorithmDir+'/images/energy-costs.jpg'
    Y1 = arr[0].reshape(int(episodes/unit),unit).mean(axis=1)/timeSlots
    Y2 = arr[1].reshape(int(episodes/unit),unit).mean(axis=1)/timeSlots
    Y3 = arr[2].reshape(int(episodes/unit),unit).mean(axis=1)/timeSlots
    plt.figure()
    plt.title(title)
    plt.xlabel("episodes")
    plt.ylabel("cost")
    plt.plot(Y1,label='part local part remote')
    plt.plot(Y2,label='total local')
    plt.plot(Y3,label='total remote')
    plt.legend(loc='best')
    plt.savefig(saveFile)
def drawAvgReward(algorithmDir):
    # reward pic2
    arr = np.loadtxt(algorithmDir+'/metrics/reward.csv',dtype=float,delimiter=',',unpack=False).T
    pic1Config = picConfigs[1]
    unit = pic1Config['unit']
    Y = arr.reshape(int(episodes/unit),unit).mean(axis=1)
    plt.figure()
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.plot(Y)
    plt.savefig(algorithmDir+'/images/reward.jpg')
def drawTotalReward(algorithmDir):
    # reward pic2
    arr = np.loadtxt(algorithmDir+'/metrics/reward.csv',dtype=float,delimiter=',',unpack=False).T
    pic1Config = picConfigs[1]
    unit = pic1Config['unit']
    Y = arr.reshape(int(episodes/unit),unit).mean(axis=1)
    Z = np.cumsum(Y)
    plt.figure()
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.plot(Z)
    plt.savefig(algorithmDir+'/images/total-rewards.jpg')
def algorithm_compare(envDir,type):
    if type != 'cost' and type != 'reward':
        print("type must be cost or reward")
        return
    compareList = []
    for algorithm in algorithmConfig:
        if algorithm['is_compare'] == 1:
            compareList.append(algorithm['name'])
    plt.figure()
    plt.xlabel("episodes")
    plt.ylabel(type)
    for algorithmName in compareList:
        try:
            algorithmDir = envDir +'/' + algorithmName
            pic2Config = picConfigs[2]
            unit = pic2Config['unit']
            arr = np.loadtxt(algorithmDir+'/metrics/'+type+'.csv',dtype=float,delimiter=',',unpack=False).T
            if type == 'cost':
                Y = arr[0].reshape(int(episodes/unit),unit).mean(axis=1)
            elif type == 'reward':
                Y = arr.reshape(int(episodes/unit),unit).mean(axis=1)
            plt.plot(Y,label=algorithmName)
        except BaseException:
            print("error you can ignore")
    plt.legend(loc='best')
    plt.savefig(envDir+'/'+type+'-compare.jpg')
def draw(envDir,algorithmDir):
    drawCost(algorithmDir,'total')
    drawCost(algorithmDir,'time')
    drawCost(algorithmDir,'energy')
    drawAvgReward(algorithmDir)
    drawTotalReward(algorithmDir)
    algorithm_compare(envDir,'cost')
    algorithm_compare(envDir,'reward')        
if __name__=='__main__':
    envDir = parentDir + '/result/env_1'
    algorithmDir = envDir + '/rl_dqn'
    draw(envDir,algorithmDir)
    



