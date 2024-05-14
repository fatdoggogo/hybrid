import pandas as pd
import matplotlib.pyplot as plt


# # Plotting the data
# if __name__ == '__main__':
#     df = pd.read_csv("../result/rl_capql/metrics/reward.csv", names=['reward'])
#     # df = pd.read_csv("../result/rl_hsac/metrics/reward.csv", names=['reward'])
#
#     # 创建一个'epi'列，它简单地代表了每个奖励值的索引（从1开始）
#     df['epi'] = range(1, len(df) + 1)
#
#     # 绘图
#     plt.figure(figsize=(15, 10))
#     plt.plot(df['epi'], df['reward'], marker='o', linestyle='-')
#     plt.title('Reward vs. Episode')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.grid(True)
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 加载第一个数据集
    df1 = pd.read_csv("../result/rl_capql/metrics/reward.csv", names=['reward'])
    df1['epi'] = range(1, len(df1) + 1)  # 创建索引列

    # 加载第二个数据集
    df2 = pd.read_csv("../result/rl_hsac/metrics/reward.csv", names=['reward'])
    df2['epi'] = range(1, len(df2) + 1)  # 创建索引列

    # 绘图设置
    plt.figure(figsize=(15, 10))

    # 绘制第一个数据集
    plt.plot(df1['epi'], df1['reward'], marker='o', linestyle='-', color='blue', label='RL_CAPQL')

    # 绘制第二个数据集
    plt.plot(df2['epi'], df2['reward'], marker='x', linestyle='-', color='red', label='RL_HSAC')

    # 添加图例、标题和轴标签
    plt.title('Reward Comparison Between Two Strategies')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.show()
