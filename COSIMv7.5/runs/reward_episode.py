import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('../result/rl_capql/metrics/reward.csv', header=None)  # 假设CSV文件没有列名

# 假设每行只有一个reward值，我们将其作为一个列的数据读取
rewards = df.iloc[:, 0]  # 选择第一列的所有行数据

# 创建episode索引
episodes = list(range(len(rewards)))
if __name__ == '__main__':
    # 绘制图形
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, marker='o', linestyle='-', color='b')
    plt.title('Episode vs. Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.grid(True)
    plt.show()
