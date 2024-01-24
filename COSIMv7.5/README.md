
### 系统假设
* In this MEC system, each device n ∈ N = {1, . . . , N},
with wireless charging, has a computation-intensive task to be
processed timely during each time slot t
*  the data of computation task are fine-grained and
can be partitioned into subsets of any size
* C t n = βBa, where the parameter Ba is the averaged data size of a typical computation task and β is a parameter that obeys a Gaussian distribution. 
* the size of results obtained from the server
is small , the time and energy consumption of feedback transmission is negligible in this work.
* To model the server’s dynamics, we assume its available computation capacity varies randomly between different time slots but remains unchanged in each time slot.
*  In this paper, we focus on the energy consumption of
local computation and transmission and ignore the others
for simplicity. 
* For simplicity, the normalized
wireless channel gain is assumed to take values in [5, 14] uniformly
### 如何使用

1. 按需要修改配置文件(COSIM/config.json)
2. 直接运行 env.py

```python
env = Env()
env.run()
```

### 配置文件说明

```json
{
    // 需要运行的时间片
    "time_slots":100,
    // time 权重系数
    "energy_weight":1,
    //energy 权重系数
    "time_weight":3,
    //设备配置数组，不同类型的设备配置可能不同，可根据需要增加新类型的设备
    "deviceConfigs":[
        {
            // A类型设备优先级，不同类型的设备优先级不同
            "priority":3,
            // A类型设备中任务处理的最大延时，单位ms,超过此值任务未完成则标记失败
            "max_process_delay":10,
            // A类型设备每个time slot最大可以处理的数据量，单位bit
            "max_task_load_per_time_slot" :1500,
            // A类型设备上传数据时 每个time slot消耗的能量,单位mj
            "energy_consume_per_time_slot":0.1,
            // A类型设备的最大cpu频率,单位GHz
            "max_cpu_frequency":3,
            // A类型设备任务队列最大容量,单位bit
            "max_load_capacity":6000,
            // A类型设备上传数据时的传输功率，单位W(瓦)
            "transmission_power":2,
            // A类型设备在无线网中的信道增益
            "channel_gain":5,   
            // A类型设备的数量
            "cnt":1
        },
        { 
            /*
            	B类型的设备,key与上述A类型设备一致，value可根据需要修改
            	需要增加新的设备类型，只需要将上述配置copy一份，修改配置值即可
            */
        }
    ],
    "serverConfigs":[
        // MEC server配置数组，配置方式与device一样，只是参数更少
        {
            // 设备向此类Server上传数据时，此设备可分配的最大带宽，单位MHZ(兆赫兹)
            "bandwidth":2,
            // 此类Server的最大CPU频率，单位GHZ
            "max_cpu_frequency":16,
            // 此类server的数量
            "cnt" :3
        }
    ],
    "constants":{
        // 设备每个 time slot产生的任务大小服从高斯分布，均值为avg_load_per_time_slot(单位bit)
        // 标准差为 task_load_gaussian_scale
    	"avg_load_per_time_slot":1000,
        "task_load_gaussian_scale":100,
    	// 设备平均每处理1 bit数据需要的CPU周期数
    	"avg_cpu_cycle_per_bit":737.5,
    	// 高斯白噪声，单位W(瓦)
    	"gaussian_channel_noise":1,
    	// 有效电容系数
    	"effective_capacitance_coefficient":1e-27	
    },
    "pic_configs":[
        // 绘图相关配置
        {
            "id":1,
            "comment":"环境cost曲线图",
            // 每unit个time slot为一个点
            // 当unit = 1,即每个time slot 为一个点
            // 当unit = 10 即取10个数据的算术平均数作为一个点
            // 当执行的时间片较大，产生的数据较大，一个时间片一个点会导致画出的图过于密集，可通过调剂此参数解决
            "unit":1
        },
        {
            "id":2,
            "comment":"环境reward曲线图",
            "unit":1
        },
        {
            "id":3,
            "comment":"设备部分卸载vs全部本地处理vs全部卸载所花时间曲线图，每个设备一张",
            "unit":1
        },
        {
            "id":4,
            "comment":"设备部分卸载vs全部本地处理vs全部卸载所耗能量曲线图，每个设备一张",
            "unit":1
        },
        {
            "id":5,
            "comment":"不同设备处理时间对比曲线图",
            "pic_size":[8,4],
            "unit":1
        },
        {
            "id":6,
            "comment":"不同设备处理耗能对比曲线图",
            "pic_size":[8,4],
            "unit":1
        },
        {
            "id":7,
            "comment":"不同设备的任务失败数散点图(每unit个time slot内失败任务数的总和)",
            "unit":10
        }
    ]
}
```

### 指标相关

为了方便作图等分析，环境运行过程中相关指标以csv文件的形式保存到/COSIM/metrics，每一个time slot的数据作为一行

reward_and_cost.vsv 为环境的reward和cost数据 数据格式为(时间片数，2),第一列为reward,第二列为cost

device[%d].csv则为device[%d] 运行过程中的数据

| 第0列            | 第1列            | 第2列              | 第3列              | 第4列                | 第5列                | 第6列                | 第7列                |
| ---------------- | ---------------- | ------------------ | ------------------ | -------------------- | -------------------- | -------------------- | -------------------- |
| 设备ID           | 设备优先级       | 设备CPU频率        | 设备任务队列总负载 | 设备当前处理的负载   | 设备处理时延         | 设备处理能耗         | 设备本次处理结果     |
| 第8列            | 第9列            | 第10列             | 第11列             | 第12列               | 第13列               | 第14列               | 第15列               |
| 设备本地处理时延 | 设备本地处理能耗 | server远程处理时延 | server远程处理能耗 | 假设全部本地处理时延 | 假设全部本地处理能耗 | 假设全部远程处理时延 | 假设全部远程处理能耗 |

注意每次运行都会覆盖上次运行的数据，如需要保存，请手动保存

### 日志相关

运行过程中日志打印较多，如果time slot较大，日志在控制台只能看到一部分，为了方便排查问题，将运行日志输出到./COSIM/logs/目录之下

没有数字后缀的runs.log是最新的，带数字后缀的是最老的，数字越大，表示时间越久，最多保留10个log文件，每个最多10M

### 绘图相关
* 绘图相关代码文件analysis.py
* 绘制的图片保存到./COSIM/images目录下，每次运行都会进行覆盖，注意提前手动保存

### 调参经验
先在随机算法下调参，调到cost和reward都有上有下，然后验证优化算法是否可以稳定使得cost趋势向下，reward趋势向上
0.先确定device数量和server的数量,数量发生变化，卸载时分配到的带宽和CPU就会发生变化，往往需要相应的调整其他的参数
1.尽量保证完全本地处理的时间和能耗 > 卸载到server的延时和能耗
1) 本地处理时间 < 卸载到server的处理时间
    可以调小本地CPU使得本地处理时间增大，也可以通过提高带宽或者serverCPU使得server处理时间减小
2.rewards只增不减少(只有奖励没有惩罚)
1) 通过调小设备的超时容忍度 使得失败任务数增加
2）直接加大失败惩罚
