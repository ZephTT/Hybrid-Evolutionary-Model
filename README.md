# 混合进化模型 (Hybrid Evolutionary Model)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个模拟种群在动态环境和技术突破驱动下适应性演化的 Python 模型。适合用于教学演示、进化生物学研究、以及技术演化理论的计算机实验

**基础模型为本人制作，后辅以AI完善优化**

## 项目背景

生物种群的演化不仅受自然选择、突变和遗传漂变的影响，还受到环境变迁的强烈塑造。当环境发生剧烈变化（如气候剧变、新资源出现）时，种群需要快速适应新的生态位，否则可能面临灭绝。

本模型将这一过程抽象为 **“特征值（trait）”** 的演化，环境用一个**高斯生态位**（最优值和宽度）描述。特别地，模型引入了 **“技术突破”** 机制——当种群适应度改善停滞时，环境最优值会发生跃迁，模拟了人类社会中“技术范式转移”或自然界中“关键创新”对种群演化的推动作用。

通过调节突变率、选择压力、重组强度和环境噪声，你可以探索不同条件下种群的命运：是走向多样化、发生适应性辐射，还是因无法跟上变化而局部灭绝。

---

## 核心功能

- **连续特征值演化**：每个个体用一个实数表示其特征，如体型、代谢率或技术能力。
- **高斯适应度景观**：适应度由个体特征与环境最优值的距离决定，并添加乘性环境噪声和密度制约（种群越接近承载力，适应度越低）。
- **选择与繁殖**：适应度高的个体繁殖概率更高；后代通过**突变**（多尺度步长）和**重组**（父母特征平均加噪声）产生变异。
- **技术突破机制**：
  - **自适应模式**：当连续世代平均适应度改善率低于阈值时，自动触发突破（环境最优值发生随机偏移）。
  - **固定模式**：在预设世代发生突破（可用于对比实验）。
- **丰富的种群动态记录**：包括平均特征值、多样性（不同特征值的种类数）、种群大小、平均适应度、灭绝事件（某个特征值在种群中消失）。
- **可视化面板**：一次性展示所有关键指标，便于快速分析。
- **参数实验框架**：批量运行不同参数组合，自动生成对比柱状图。

---
## 版本说名

分支 `main` 中
Hybrid-Evolutionary-Model.py为完整版
而分支`dev-buggy-version`中
Hybrid-Evolutionary-Model_deving.py为未完成版
新增了 `run_interactive_simulation() ` 交互式动画（未完成）
在`visualize()`中加入了适应度函数
把 `calculate_evolutionary_metrics()` 的指标整合到新版的信息面板中。
技术时代标签：在 `__init__` 里加一个 `era_names` 参数，让用户自定义。

## 安装与依赖

### 环境要求
- Python 3.8 及以上
- NumPy
- SciPy
- Matplotlib

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/ZephTT/hybrid-evolutionary-model.git
cd hybrid-evolutionary-model

# 安装依赖（推荐使用虚拟环境）
pip install numpy scipy matplotlib
```
## 快速开始
**运行默认模拟**
```bash
python hybrid_evolution.py
脚本将运行 150 代，并弹出一个包含所有图表的面板。控制台会实时输出每 25 代的状态摘要。
```
自定义参数
在 `if __name__ == "__main__"`: 中修改 `HybridEvolutionaryModel` 的参数，例如：

```python
model = HybridEvolutionaryModel(
    initial_population=800,
    initial_trait_mean=1.2,
    initial_trait_std=0.2,
    generations=150,
    mutation_rate=0.03,
    recombination_rate=0.05,
    recombination_noise=0.05,
    carrying_capacity=2000,
    selection_pressure=2.0,
    env_noise=0.03,
    tech_mode='adaptive',          # 或 'fixed'
    adaptive_threshold=0.008,
    min_breakthrough_gap=15,
    seed=42                         # 固定种子保证可重复性
)
history = model.simulate()
model.visualize()
```
运行参数实验
脚本会询问是否运行一组预设实验（高/低突变率、高/低选择压力、小承载力、无重组）。输入 `y` 即可执行，并显示对比结果。

你也可以自定义实验列表：

```python
experiments = [
    {'name': '高突变率', 'mutation_rate': 0.08},
    {'name': '强选择', 'selection_pressure': 4.0},
    # ...
]
model.run_parameter_experiment(experiments)
```
## 主要参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `initial_population` | int | 800 | 初始个体数 |
| `initial_trait_mean` | float | 1.0 | 初始特征均值 |
| `initial_trait_std` | float | 0.2 | 初始特征标准差 |
| `generations` | int | 150 | 总模拟代数 |
| `mutation_rate` | float | 0.03 | 每个个体发生突变的概率 |
| `mutation_step_options` | List[float] | [0.1,0.3,0.5,1.0] | 突变步长可选值 |
| `mutation_step_weights` | List[float] | [0.5,0.3,0.15,0.05] | 对应步长的权重 |
| `recombination_rate` | float | 0.05 | 配对个体重组的概率 |
| `recombination_noise` | float | 0.05 | 重组后子代偏离父母均值的噪声标准差 |
| `carrying_capacity` | int | 2000 | 环境承载力（种群上限） |
| `selection_pressure` | float | 2.0 | 适应度指数放大的系数（越高选择越强） |
| `env_noise` | float | 0.03 | 环境适应度的乘性噪声幅度 |
| `tech_mode` | str | 'adaptive' | 'adaptive'（自适应）或 'fixed'（固定） |
| `fixed_breakthroughs` | List[int] | [] | 固定突破世代列表（仅 `fixed` 模式有效） |
| `adaptive_threshold` | float | 0.008 | 自适应突破阈值（改善率低于此值触发） |
| `min_breakthrough_gap` | int | 15 | 自适应模式下两次突破的最短间隔 |
| `seed` | int | None | 随机种子（设为整数可重现结果） |

## 输出与图表解读
运行 `visualize()` 后，你会看到一个包含 8 个子图的综合面板：

| 子图位置 | 内容 | 解读 |
|---------|------|------|
| 左上 (0, :2) | **平均特征值演化 vs 环境最优值** | 蓝色曲线为种群平均值，红色虚线为当前环境最优值。紫色竖线表示技术突破时刻，可观察种群是否跟随环境变化。 |
| 中上 (0, 2) | **种群动态**（双轴） | 绿色为多样性（特征值种类数），品红色为种群大小。可观察多样性变化与种群规模的关系。 |
| 右上 (0, 3) | **平均适应度** | 适应度越高表示种群越适应环境。突破后适应度通常会先下降再恢复。 |
| 中左 (1, :2) | **最终特征值分布**（核密度图） | 显示模拟结束时的特征值分布形态，可看出是否出现多峰（分化）或单峰。 |
| 中中 (1, 2) | **特征值变化速率** | 反映演化速度，突破处通常有剧烈变化。 |
| 中右 (1, 3) | **灭绝事件** | 散点表示在某代消失的特征值，气泡大小表示该特征值在消失前占种群的比例。大量灭绝可能表明环境剧变。 |
| 下左 (2, :3) | **技术时代变迁** | 横轴为世代，纵轴为时代编号，上升台阶即表示突破发生。 |
| 下右 (2, 3) | **模拟信息摘要** | 列出配置和关键结果数据，方便截图记录。 |

## 核心机制详解（简化版）
**适应度计算**：
个体特征值 `x` 的适应度正比于 `exp(- (x - opt)^2 / (2*width^2))` ，然后乘以环境噪声（0.7~1.3 的随机因子）和密度依赖因子（种群越满，因子越低）。最后用`selection_pressure`指数放大，以强化选择效果。

**突变**：
每个个体以 `mutation_rate` 概率发生突变。突变步长从 `mutation_step_options` 中按 `mutation_step_weights` 权重随机选择，并乘以一个 0.8~1.2 的抖动因子。突变方向有 70% 概率为正（增加特征值），30% 为负。

**重组**：
繁殖时，从父代中随机配对，每对以 `recombination_rate` 概率进行重组——两个子代特征值都取父母均值再加上一个均值为 0、标准差为 `recombination_noise` 的正态噪声。未重组的则直接继承父母特征。

**技术突破**：
自适应模式下，每代检查平均适应度相对上一代的改善率。若改善率低于 `adaptive_threshold` 且距上次突破已超过 `min_breakthrough_gap`，则触发突破。新最优值 = 当前种群平均特征值 + 随机偏移（正态分布均值 0.8，标准差 0.3），新的生态位宽度为原来的 0.8 倍（环境变得更加“狭窄”，要求更精确的适应）。

## 应用场景
**教学演示**：直观展示自然选择、遗传漂变、生态位构建等概念。
**假设检验**：探讨突变率、选择强度、环境变化频率如何影响种群的适应能力和多样性。
**技术演化研究**：模拟“技术范式转移”对群体文化演化的影响。
**参数敏感性分析**：利用实验框架快速评估不同参数的重要性。
## 贡献
欢迎提交 Issue 或 Pull Request！如果你添加了新功能或改进了代码，请确保：
代码符合 PEP 8 风格。
添加必要的注释。
更新 README（如适用）。
## 许可证
本项目采用 MIT 许可证。这意味着你可以自由地使用、复制、修改、合并、出版、分发、再许可及销售本软件的副本，只需保留原版权声明即可。更多细节请参见 LICENSE 文件。

## 联系
如有问题或建议，请通过 GitHub Issues 联系，或发送邮件至 kevin_dai2011@outlook.com
# ENGLISH VERSION
# Hybrid Evolutionary Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python model that simulates the adaptive evolution of a population driven by dynamic environments and technological breakthroughs. Suitable for educational demonstrations, evolutionary biology research, and computational experiments on theories of technological evolution.

**The base model was created by the author, and subsequently refined and optimized with AI assistance.**

## Project Background

The evolution of biological populations is shaped not only by natural selection, mutation, and genetic drift but also strongly by environmental changes. When the environment undergoes drastic shifts (e.g., climate upheaval, emergence of new resources), populations must rapidly adapt to new niches or face extinction.

This model abstracts this process into the evolution of a **“trait”** , with the environment described by a **Gaussian niche** (optimum value and width). Notably, the model introduces a **“technological breakthrough”** mechanism—when improvement in population fitness stagnates, the environmental optimum jumps, simulating the impact of “technological paradigm shifts” in human societies or “key innovations” in nature on population evolution.

By tuning parameters such as mutation rate, selection pressure, recombination intensity, and environmental noise, you can explore the fate of populations under different conditions: diversification, adaptive radiation, or local extinction due to inability to keep pace with change.

---

## Core Features

- **Continuous trait evolution**: Each individual is represented by a real number denoting its trait, e.g., body size, metabolic rate, or technological capability.
- **Gaussian fitness landscape**: Fitness is determined by the distance between an individual's trait and the environmental optimum, multiplied by environmental noise and a density-dependent factor (the closer the population is to the carrying capacity, the lower the fitness).
- **Selection and reproduction**: Individuals with higher fitness have a higher probability of reproducing; offspring generate variation through **mutation** (multi-scale step sizes) and **recombination** (mid-parent mean plus noise).
- **Technological breakthrough mechanism**:
  - **Adaptive mode**: Automatically triggers a breakthrough (random shift in the environmental optimum) when the improvement rate of the population's mean fitness over consecutive generations falls below a threshold.
  - **Fixed mode**: Breakthroughs occur at predefined generations (useful for controlled experiments).
- **Rich population dynamics recording**: Includes mean trait, diversity (number of distinct trait values), population size, average fitness, and extinction events (when a particular trait value disappears from the population).
- **Visualization panel**: Displays all key metrics at once for rapid analysis.
- **Parameter experiment framework**: Batch-runs different parameter combinations and automatically generates comparative bar charts.

---

## Version Notes

In the `main` branch,  
`Hybrid-Evolutionary-Model.py` is the full version.  
In the `dev-buggy-version` branch,  
`Hybrid-Evolutionary-Model_deving.py` is an unfinished version that adds:
- An interactive animation function `run_interactive_simulation()` (unfinished)
- Fitness function display within `visualize()`
- Integration of metrics from `calculate_evolutionary_metrics()` into the new information panel.
- Technological era labels: an `era_names` parameter in `__init__` for user customization.

## Installation and Dependencies

### Requirements
- Python 3.8 or higher
- NumPy
- SciPy
- Matplotlib

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/ZephTT/hybrid-evolutionary-model.git
cd hybrid-evolutionary-model

# Install dependencies (use of a virtual environment is recommended)
pip install numpy scipy matplotlib
```

Quick Start

Run the default simulation

```bash
python hybrid_evolution.py
The script will run for 150 generations and display a panel with all charts. The console will print a status summary every 25 generations.
```

Customizing Parameters
Modify the parameters of HybridEvolutionaryModel inside if __name__ == "__main__":, for example:

```python
model = HybridEvolutionaryModel(
    initial_population=800,
    initial_trait_mean=1.2,
    initial_trait_std=0.2,
    generations=150,
    mutation_rate=0.03,
    recombination_rate=0.05,
    recombination_noise=0.05,
    carrying_capacity=2000,
    selection_pressure=2.0,
    env_noise=0.03,
    tech_mode='adaptive',          # or 'fixed'
    adaptive_threshold=0.008,
    min_breakthrough_gap=15,
    seed=42                         # fix seed for reproducibility
)
history = model.simulate()
model.visualize()
```

Running a Parameter Experiment
The script will ask whether to run a set of preset experiments (high/low mutation rate, high/low selection pressure, small carrying capacity, no recombination). Enter y to execute and display the comparison results.

You can also customize the experiment list:

```python
experiments = [
    {'name': 'High Mutation Rate', 'mutation_rate': 0.08},
    {'name': 'Strong Selection', 'selection_pressure': 4.0},
    # ...
]
model.run_parameter_experiment(experiments)
```

Main Parameter Descriptions

Parameter Type Default Description
initial_population int 800 Initial number of individuals
initial_trait_mean float 1.0 Initial mean trait value
initial_trait_std float 0.2 Initial trait standard deviation
generations int 150 Total number of generations to simulate
mutation_rate float 0.03 Probability of mutation per individual
mutation_step_options List[float] [0.1,0.3,0.5,1.0] Available mutation step sizes
mutation_step_weights List[float] [0.5,0.3,0.15,0.05] Weights corresponding to each step size
recombination_rate float 0.05 Probability of recombination for a mated pair
recombination_noise float 0.05 Std dev of normal noise added to offspring's deviation from the parental mean
carrying_capacity int 2000 Environmental carrying capacity (population ceiling)
selection_pressure float 2.0 Exponential amplification factor for fitness (higher = stronger selection)
env_noise float 0.03 Magnitude of multiplicative environmental noise on fitness
tech_mode str 'adaptive' 'adaptive' or 'fixed'
fixed_breakthroughs List[int] [] List of generations for fixed breakthroughs (only used in fixed mode)
adaptive_threshold float 0.008 Adaptive breakthrough threshold (triggers when improvement rate falls below this)
min_breakthrough_gap int 15 Minimum gap between breakthroughs in adaptive mode
seed int None Random seed (set an integer for reproducibility)

Output and Chart Interpretation

After running visualize(), you will see a comprehensive panel with 8 subplots:

Subplot Position Content Interpretation
Top-left (0, :2) Mean trait evolution vs. environmental optimum Blue curve = population mean, red dashed line = current environmental optimum. Purple vertical lines indicate technological breakthroughs; observe whether the population tracks environmental changes.
Top-middle (0, 2) Population dynamics (dual y-axes) Green = diversity (number of distinct trait values), magenta = population size. Observe the relationship between diversity changes and population size.
Top-right (0, 3) Average fitness Higher fitness means better adaptation to the environment. After a breakthrough, fitness usually drops first and then recovers.
Middle-left (1, :2) Final trait distribution (kernel density plot) Shows the trait distribution at the end of the simulation, revealing whether it is multimodal (differentiated) or unimodal.
Middle-middle (1, 2) Trait change rate Reflects the speed of evolution; sharp changes often occur at breakthroughs.
Middle-right (1, 3) Extinction events Scatter plot showing trait values that went extinct in a given generation; bubble size indicates the proportion of that trait in the population before extinction. Many extinctions may indicate a drastic environmental shift.
Bottom-left (2, :3) Technological era transitions X-axis = generation, Y-axis = era number; upward steps indicate breakthroughs.
Bottom-right (2, 3) Simulation information summary Lists the configuration and key outcome statistics for easy screenshot recording.

Core Mechanism Details (Simplified)

Fitness Calculation:
The fitness of an individual with trait x is proportional to exp(- (x - opt)^2 / (2*width^2)) , then multiplied by environmental noise (a random factor between 0.7 and 1.3) and a density-dependent factor (which decreases as the population fills up). Finally, it is amplified exponentially by selection_pressure to strengthen selection effects.

Mutation:
Each individual mutates with probability mutation_rate. The mutation step size is randomly chosen from mutation_step_options according to mutation_step_weights, and multiplied by a jitter factor (0.8–1.2). The direction of mutation is positive (increasing trait value) 70% of the time and negative 30% of the time.

Recombination:
During reproduction, individuals in the parent pool are randomly paired. Each pair undergoes recombination with probability recombination_rate—both offspring take the parental mean plus normally distributed noise with mean 0 and standard deviation recombination_noise. Non-recombined offspring directly inherit the parental traits.

Technological Breakthrough:
In adaptive mode, every generation checks the improvement rate of average fitness relative to the previous generation. If the improvement rate drops below adaptive_threshold and the time since the last breakthrough exceeds min_breakthrough_gap, a breakthrough is triggered. The new optimum = current population mean trait + random offset (normally distributed with mean 0.8, std 0.3), and the new niche width becomes 0.8 times the previous width (the environment becomes "narrower", demanding more precise adaptation).

Application Scenarios

Educational demonstration: Intuitively showcase concepts such as natural selection, genetic drift, and niche construction.
Hypothesis testing: Explore how mutation rate, selection strength, and frequency of environmental change affect a population's adaptability and diversity.
Technological evolution research: Simulate the impact of "technological paradigm shifts" on cultural evolution in groups.
Parameter sensitivity analysis: Use the experiment framework to quickly assess the importance of different parameters.

Contributing

Issues and Pull Requests are welcome! If you add new features or improve the code, please ensure:
The code follows PEP 8 style.
Add necessary comments.
Update the README (if applicable).

License

This project is licensed under the MIT License. This means you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that you retain the original copyright notice. See the LICENSE file for more details.

Contact

For questions or suggestions, please reach out via GitHub Issues or email kevin_dai2011@outlook.com

```