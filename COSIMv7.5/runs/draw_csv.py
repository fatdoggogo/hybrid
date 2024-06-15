import pandas as pd
import matplotlib.pyplot as plt

def aggregate_rewards(df, interval):
    """
    每隔interval个episode取一个平均值
    """
    df['group'] = (df.index // interval) + 1
    aggregated_df = df.groupby('group').mean().reset_index(drop=True)
    aggregated_df['epi'] = aggregated_df.index * interval + 1
    return aggregated_df

if __name__ == '__main__':
    # 加载第一个数据集
    df1 = pd.read_csv("../result/rl_capql/metrics/reward.csv", names=['reward'])
    df1['epi'] = range(1, len(df1) + 1)  # 创建索引列

    # 加载第二个数据集
    df2 = pd.read_csv("../result/rl_hsac/metrics/reward.csv", names=['reward'])
    df2['epi'] = range(1, len(df2) + 1)  # 创建索引列

    # 加载第三个数据集
    df3 = pd.read_csv("../result/rl_modqn/metrics/reward.csv", names=['reward'])
    df3['epi'] = range(1, len(df3) + 1)  # 创建索引列

    # 加载第三个数据集
    df4 = pd.read_csv("../result/rl_pdmorl/metrics/reward.csv", names=['reward'])
    df4['epi'] = range(1, len(df4) + 1)  # 创建索引列

    # 每隔5个episode取一个平均值
    df1_aggregated = aggregate_rewards(df1, 5)
    df2_aggregated = aggregate_rewards(df2, 5)
    df3_aggregated = aggregate_rewards(df3, 5)
    df4_aggregated = aggregate_rewards(df4, 5)

    # 绘图设置
    plt.figure(figsize=(15, 10))

    # 绘制第一个数据集
    plt.plot(df1_aggregated['epi'], df1_aggregated['reward'], marker='o', linestyle='-', color='blue', label='RL_CAPQL')

    # 绘制第二个数据集
    plt.plot(df2_aggregated['epi'], df2_aggregated['reward'], marker='x', linestyle='-', color='red', label='RL_HSAC')

    # 绘制第三个数据集
    plt.plot(df3_aggregated['epi'], df3_aggregated['reward'], marker='^', linestyle='-', color='green', label='RL_MODQN')

    # 绘制第四个数据集
    plt.plot(df4_aggregated['epi'], df4_aggregated['reward'], marker='^', linestyle='-', color='purple', label='RL_PDMORL')

    # 添加图例、标题和轴标签
    plt.title('Reward Comparison Between Three Strategies (Averaged every 5 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.show()
