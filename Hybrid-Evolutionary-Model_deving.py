import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # 优先使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
from scipy.stats import gaussian_kde
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

class HybridEvolutionaryModel:

    def __init__(self,
                 initial_population: int = 800,
                 initial_trait_mean: float = 1.0,
                 initial_trait_std: float = 0.2,
                 generations: int = 150,
                 mutation_rate: float = 0.03,
                 mutation_step_options: List[float] = None,
                 mutation_step_weights: List[float] = None,
                 recombination_rate: float = 0.05,
                 recombination_noise: float = 0.05,  # 新增：重组噪声标准差
                 carrying_capacity: int = 2000,
                 selection_pressure: float = 2.0,
                 env_noise: float = 0.03,
                 tech_mode: str = 'adaptive',
                 fixed_breakthroughs: Optional[List[int]] = None,
                 adaptive_threshold: float = 0.008,
                 min_breakthrough_gap: int = 15,
                 seed: Optional[int] = None):

        self.rng = np.random.default_rng(seed)

        self.generations = generations
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate
        self.recombination_noise = recombination_noise
        self.carrying_capacity = carrying_capacity
        self.selection_pressure = selection_pressure
        self.env_noise = env_noise

        self.mutation_step_options = mutation_step_options or [0.1, 0.3, 0.5, 1.0]
        self.mutation_step_weights = mutation_step_weights or [0.5, 0.3, 0.15, 0.05]

        self.tech_mode = tech_mode
        if tech_mode == 'fixed':
            self.fixed_breakthroughs = fixed_breakthroughs if fixed_breakthroughs else []
            self.adaptive_threshold = None
            self.min_breakthrough_gap = 0
        else:
            self.fixed_breakthroughs = []
            self.adaptive_threshold = adaptive_threshold
            self.min_breakthrough_gap = min_breakthrough_gap

        self.era_optima = [1.5]
        self.era_widths = [0.8]
        self.breakthrough_generations = []

        self.population = self.rng.normal(initial_trait_mean, initial_trait_std, initial_population)
        self.population = np.clip(self.population, 0.1, None)

        self.history = {
            'avg_trait': [],
            'diversity': [],
            'population_size': [],
            'fitness_avg': [],
            'extinction_events': [],
            'era_optima_record': [],
            'trait_distribution': [],
        }

        print("=" * 70)
        print("进化模型")
        print(f"模式: {'自适应技术突破' if tech_mode == 'adaptive' else '固定世代突破'}")
        if tech_mode == 'adaptive':
            print(f"  适应度改善率阈值: {adaptive_threshold}, 最小间隔: {min_breakthrough_gap}")
        else:
            print(f"  固定突破世代: {fixed_breakthroughs}")
        print(f"选择压力系数: {selection_pressure}")
        print(f"突变步长分布: {list(zip(self.mutation_step_options, self.mutation_step_weights))}")
        print(f"环境承载力: {carrying_capacity}")
        print("=" * 70)

    def get_current_era_params(self, generation: int, current_era: int) -> Tuple[float, float]:
        return self.era_optima[current_era], self.era_widths[current_era]

    def compute_fitness(self, generation: int, current_era: int) -> np.ndarray:
        opt, width = self.get_current_era_params(generation, current_era)
        traits = self.population

        # 高斯适应度（基础）
        fitness = np.exp(-((traits - opt) ** 2) / (2 * width ** 2))

        # 环境噪声（乘性）
        if self.env_noise > 0:
            noise = 1 + self.rng.normal(0, self.env_noise, len(traits))
            fitness *= np.clip(noise, 0.7, 1.3)

        # 密度依赖（在指数放大之前乘以密度因子，使影响更线性）
        density = len(traits) / self.carrying_capacity
        density_factor = np.clip(1 - density * 0.7, 0.3, 1.0)
        fitness *= density_factor

        # 选择压力（指数放大）
        fitness = np.exp(self.selection_pressure * fitness)
        return np.clip(fitness, 1e-10, None)

    def mutate_vectorized(self, population: np.ndarray) -> np.ndarray:
        mutation_mask = self.rng.random(len(population)) < self.mutation_rate
        n_mut = np.sum(mutation_mask)

        if n_mut > 0:
            directions = self.rng.choice([-1, 1], size=n_mut, p=[0.3, 0.7])
            steps = self.rng.choice(
                self.mutation_step_options,
                size=n_mut,
                p=self.mutation_step_weights
            )
            jitter = 0.8 + 0.4 * self.rng.random(n_mut)
            mutated = population[mutation_mask] + directions * steps * jitter
            population[mutation_mask] = np.clip(mutated, 0.1, None)

        return population

    def recombine_vectorized(self, parents: np.ndarray) -> np.ndarray:
        n = len(parents)
        if n < 2 or self.recombination_rate == 0:
            return parents

        shuffled_idx = self.rng.permutation(n)
        n_even = n - (n % 2)
        pairs = parents[shuffled_idx[:n_even]].reshape(-1, 2)

        rec_mask = self.rng.random(len(pairs)) < self.recombination_rate

        p1, p2 = pairs[:, 0], pairs[:, 1]
        mean_val = (p1 + p2) / 2
        noise1 = self.rng.normal(0, self.recombination_noise, len(pairs))
        noise2 = self.rng.normal(0, self.recombination_noise, len(pairs))

        offspring = np.empty_like(pairs)
        offspring[rec_mask, 0] = mean_val[rec_mask] + noise1[rec_mask]
        offspring[rec_mask, 1] = mean_val[rec_mask] + noise2[rec_mask]
        offspring[~rec_mask, 0] = p1[~rec_mask]
        offspring[~rec_mask, 1] = p2[~rec_mask]

        result = offspring.ravel()
        if n % 2 == 1:
            result = np.append(result, parents[shuffled_idx[-1]])

        return np.clip(result, 0.1, None)

    def reproduce(self, fitness: np.ndarray) -> np.ndarray:
        prob = fitness / np.sum(fitness)
        parent_indices = self.rng.choice(len(self.population), size=len(self.population), p=prob)
        parents = self.population[parent_indices]
        offspring = self.recombine_vectorized(parents)
        offspring = self.mutate_vectorized(offspring)
        if len(offspring) > self.carrying_capacity:
            offspring = self.rng.choice(offspring, self.carrying_capacity, replace=False)
        return offspring

    def detect_extinction(self, current_gen: int, prev_population: np.ndarray, threshold: float = 0.05):
        if prev_population is None or len(prev_population) == 0:
            return
        prev_round = np.round(prev_population, 1)
        curr_round = np.round(self.population, 1)
        prev_unique = set(prev_round)
        curr_unique = set(curr_round)
        extinct_vals = prev_unique - curr_unique
        for val in extinct_vals:
            prev_count = np.sum(prev_round == val)
            prev_ratio = prev_count / len(prev_population)
            if prev_ratio > threshold:
                self.history['extinction_events'].append({
                    'generation': current_gen,
                    'trait_value': val,
                    'previous_proportion': prev_ratio
                })

    def check_adaptive_breakthrough(self, generation: int, current_avg_fitness: float,
                                    previous_avg_fitness: float, current_era: int) -> bool:
        if generation < 10:
            return False

        if self.breakthrough_generations:
            last_break = self.breakthrough_generations[-1]
            if generation - last_break < self.min_breakthrough_gap:
                return False

        # 防止除以零
        if previous_avg_fitness <= 1e-12:
            return False
        improvement = (current_avg_fitness - previous_avg_fitness) / previous_avg_fitness

        if improvement < self.adaptive_threshold:
            # 突破方向改为随机偏移（可配置，此处使用正态分布）
            shift = self.rng.normal(0.8, 0.3)  # 均值0.8，标准差0.3，允许双向变化
            new_opt = np.mean(self.population) + shift
            new_width = self.era_widths[current_era] * 0.8
            self.era_optima.append(new_opt)
            self.era_widths.append(new_width)
            self.breakthrough_generations.append(generation)
            print(f"*** 第{generation}代: 自适应技术突破! 新最优值 = {new_opt:.2f}, 宽度 = {new_width:.3f} ***")
            return True
        return False

    def simulate(self) -> Dict:
        print("开始模拟...")
        prev_population = None
        prev_avg_fitness = 0.0
        current_era = 0

        for gen in range(self.generations):
            while (current_era < len(self.breakthrough_generations) and
                   gen >= self.breakthrough_generations[current_era]):
                current_era += 1

            fitness = self.compute_fitness(gen, current_era)
            avg_fitness = np.mean(fitness)

            self.history['avg_trait'].append(np.mean(self.population))
            self.history['diversity'].append(len(np.unique(np.round(self.population, 1))))
            self.history['population_size'].append(len(self.population))
            self.history['fitness_avg'].append(avg_fitness)
            self.history['era_optima_record'].append(self.era_optima[current_era])
            self.history['trait_distribution'].append(self.population.copy())

            if gen > 0 and gen % 5 == 0 and prev_population is not None:
                self.detect_extinction(gen, prev_population)

            prev_population = self.population
            self.population = self.reproduce(fitness)

            if self.tech_mode == 'adaptive':
                if self.check_adaptive_breakthrough(gen, avg_fitness, prev_avg_fitness, current_era):
                    current_era += 1
            else:
                if gen in self.fixed_breakthroughs:
                    print(f"第{gen}代: 固定技术突破!")
                    self.breakthrough_generations.append(gen)
                    shift = self.rng.normal(0.8, 0.3)
                    new_opt = np.mean(self.population) + shift
                    new_width = self.era_widths[current_era] * 0.8
                    self.era_optima.append(new_opt)
                    self.era_widths.append(new_width)
                    current_era += 1

            prev_avg_fitness = avg_fitness

            if gen % 25 == 0 or gen == self.generations - 1:
                print(f"世代 {gen:3d}: 平均特征值={self.history['avg_trait'][-1]:.2f}, "
                      f"多样性={self.history['diversity'][-1]}, 种群={len(self.population)}")

        print("\n模拟完成")
        return self.history

    def calculate_evolutionary_metrics(self) -> Dict:
        """计算进化指标（从旧版移植）"""
        if not self.history['avg_trait']:
            return {}

        metrics = {}

        # 进化速率
        trait_changes = np.diff(self.history['avg_trait'])
        metrics['avg_evolution_rate'] = float(np.mean(np.abs(trait_changes)))
        metrics['max_evolution_rate'] = float(np.max(np.abs(trait_changes)))

        # 检测间断平衡（快速进化期）
        if len(trait_changes) > 0:
            rapid_change_threshold = np.percentile(np.abs(trait_changes), 75)
            rapid_periods = np.where(np.abs(trait_changes) > rapid_change_threshold)[0]
            metrics['punctuated_periods'] = len(rapid_periods)
        else:
            metrics['punctuated_periods'] = 0

        # 收敛稳定性（最后10代的变化标准差）
        if len(self.history['avg_trait']) > 10:
            metrics['stability'] = float(np.std(self.history['avg_trait'][-10:]))
        else:
            metrics['stability'] = float(np.std(self.history['avg_trait'])) if self.history['avg_trait'] else 0.0

        # 多样性指标
        metrics['avg_diversity'] = float(np.mean(self.history['diversity']))
        metrics['diversity_change'] = int(self.history['diversity'][-1] - self.history['diversity'][0])

        # 适应度提升
        if self.history['fitness_avg']:
            metrics['fitness_gain'] = float(self.history['fitness_avg'][-1] - self.history['fitness_avg'][0])
        else:
            metrics['fitness_gain'] = 0.0

        return metrics

    def visualize(self):
        hist = self.history
        gens = range(len(hist['avg_trait']))

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(gens, hist['avg_trait'], 'b-', lw=2.5, label='种群平均特征值')
        ax1.plot(gens, hist['era_optima_record'], 'r--', lw=1.5, alpha=0.7, label='环境最优值')
        for bg in self.breakthrough_generations:
            ax1.axvline(bg, color='purple', linestyle=':', alpha=0.6, linewidth=1)
            ax1.text(bg, ax1.get_ylim()[1] * 0.9, f'突破', rotation=90, fontsize=8, color='purple')
        ax1.set_xlabel('世代')
        ax1.set_ylabel('特征值')
        ax1.set_title('平均特征值演化 vs 环境最优值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 2])
        ax2_twin = ax2.twinx()
        ax2.plot(gens, hist['diversity'], 'g-', lw=2, label='多样性')
        ax2_twin.plot(gens, hist['population_size'], 'm-', lw=2, label='种群大小')
        ax2.set_xlabel('世代')
        ax2.set_ylabel('多样性 (特征值种类)', color='g')
        ax2_twin.set_ylabel('种群大小', color='m')
        ax2.set_title('种群动态')
        ax2.grid(True, alpha=0.3)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax3 = fig.add_subplot(gs[0, 3])
        ax3.plot(gens, hist['fitness_avg'], 'c-', lw=2)
        for bg in self.breakthrough_generations:
            ax3.axvline(bg, color='purple', linestyle=':', alpha=0.5)
        ax3.set_xlabel('世代')
        ax3.set_ylabel('平均适应度')
        ax3.set_title('平均适应度')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, :2])
        if len(self.population) > 0:
            kde = gaussian_kde(self.population)
            x_vals = np.linspace(max(0.1, np.min(self.population) - 0.5),
                                 np.max(self.population) + 0.5, 200)
            y_vals = kde(x_vals)
            ax4.plot(x_vals, y_vals, 'b-', lw=2)
            ax4.fill_between(x_vals, y_vals, alpha=0.3)
            ax4.set_xlabel('特征值')
            ax4.set_ylabel('概率密度')
            ax4.set_title('最终种群特征值分布')
            ax4.grid(True, alpha=0.3)
        ax5 = fig.add_subplot(gs[1, 2])
        rates = np.abs(np.diff(hist['avg_trait']))
        ax5.plot(gens[1:], rates, 'orange', lw=1.5)
        for bg in self.breakthrough_generations:
            if bg > 0:
                ax5.axvline(bg, color='purple', linestyle=':', alpha=0.5)
        ax5.set_xlabel('世代')
        ax5.set_ylabel('进化速率 (Δ特征值/代)')
        ax5.set_title('特征值变化速率')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 3])
        if hist['extinction_events']:
            ext_gens = [e['generation'] for e in hist['extinction_events']]
            ext_vals = [e['trait_value'] for e in hist['extinction_events']]
            ext_sizes = [e['previous_proportion'] * 300 for e in hist['extinction_events']]
            ax6.scatter(ext_gens, ext_vals, s=ext_sizes, c='red', alpha=0.6, edgecolors='black')
            ax6.set_xlabel('灭绝世代')
            ax6.set_ylabel('灭绝的特征值')
            ax6.set_title(f'灭绝事件 (总数={len(hist["extinction_events"])})')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, '无灭绝事件', transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('灭绝事件')

        ax7 = fig.add_subplot(gs[2, :3])
        era_indices = []
        cur_era = 0
        next_break_idx = 0
        for g in range(self.generations):
            if next_break_idx < len(self.breakthrough_generations) and g >= self.breakthrough_generations[
                next_break_idx]:
                cur_era += 1
                next_break_idx += 1
            era_indices.append(cur_era)
        ax7.plot(gens, era_indices, 's', markersize=2, color='steelblue', alpha=0.5)
        ax7.set_xlabel('世代')
        ax7.set_ylabel('技术时代编号')
        ax7.set_title('技术时代变迁')
        ax7.set_yticks(range(len(self.era_optima)))
        ax7.grid(True, axis='y', alpha=0.3)

        ax8 = fig.add_subplot(gs[2, 3])
        ax8.axis('off')
        final_avg = hist['avg_trait'][-1]
        init_avg = hist['avg_trait'][0]
        total_break = len(self.breakthrough_generations)
        extinct_count = len(hist['extinction_events'])
        info_text = f"""
        模拟配置:
        - 世代: {self.generations}
        - 突变率: {self.mutation_rate}
        - 重组率: {self.recombination_rate}
        - 选择压力: {self.selection_pressure}
        - 承载力: {self.carrying_capacity}
        - 技术模式: {self.tech_mode}

        结果摘要:
        - 初始平均特征值: {init_avg:.2f}
        - 最终平均特征值: {final_avg:.2f}
        - 总变化: {final_avg - init_avg:.2f}
        - 最终多样性: {hist['diversity'][-1]}
        - 技术突破次数: {total_break}
        - 灭绝事件: {extinct_count}
        - 最终种群: {hist['population_size'][-1]}
        """
        ax8.text(0.05, 0.95, info_text, transform=ax8.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='sans-serif',
                 bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.8))
        plt.suptitle('进化模型模拟结果', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_enhanced(self):
        """增强版可视化：包含适应度函数对比 + 进化路径散点图"""
        if not self.history['avg_trait']:
            print("请先运行 simulate()！")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
        plt.rcParams['axes.unicode_minus'] = False

        # ---- 左图：各时代适应度函数对比 ----
        ax1 = axes[0]
        trait_range = np.linspace(0.5, 6, 200)
        # 收集所有突破世代 + 起始世代
        breakpoints = [0] + self.breakthrough_generations
        colors = ['#95A5A6', '#3498DB', '#2ECC71', '#E74C3C', '#F39C12', '#8E44AD', '#1ABC9C']

        for i, bp in enumerate(breakpoints[:len(self.era_optima)]):
            if bp < self.generations:
                opt = self.era_optima[i]
                width = self.era_widths[i]
                # 直接计算高斯适应度（不加噪声）
                fitness_vals = np.exp(-((trait_range - opt) ** 2) / (2 * width ** 2))
                label = f"时代{i}: 突破于{bp}代" if i > 0 else f"时代0: 初始"
                ax1.plot(trait_range, fitness_vals, label=label, color=colors[i % len(colors)],
                         linewidth=2.5, alpha=0.8)

        ax1.set_xlabel('特征值', fontsize=11)
        ax1.set_ylabel('适应度', fontsize=11)
        ax1.set_title('各技术时代的适应度函数对比', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ---- 右图：进化路径散点图 ----
        ax2 = axes[1]
        sample_step = max(1, self.generations // 30)
        for gen in range(0, len(self.history['trait_distribution']), sample_step):
            distribution = self.history['trait_distribution'][gen]
            if len(distribution) > 0:
                unique_vals, counts = np.unique(np.round(distribution, 1), return_counts=True)
                for val, cnt in zip(unique_vals, counts):
                    alpha = min(0.7, cnt / len(distribution) * 5)
                    ax2.scatter(gen, val, s=cnt / 10 + 5, alpha=alpha,
                                color='#3498DB', edgecolors='none')

        for bg in self.breakthrough_generations:
            ax2.axvline(x=bg, color='#E74C3C', linestyle='--', alpha=0.7, linewidth=1.5)

        ax2.set_xlabel('世代', fontsize=11)
        ax2.set_ylabel('特征值', fontsize=11)
        ax2.set_title('进化路径与多样性（点大小表示频率）', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.2)

        plt.suptitle('增强版可视化：适应度景观与进化路径', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # 可选：顺便把指标打出来
        metrics = self.calculate_evolutionary_metrics()
        if metrics:
            print("\n📊 进化指标摘要:")
            print(f"  • 平均进化速率: {metrics['avg_evolution_rate']:.4f}/代")
            print(f"  • 最大进化速率: {metrics['max_evolution_rate']:.4f}/代")
            print(f"  • 快速进化期: {metrics['punctuated_periods']} 次")
            print(f"  • 多样性变化: {metrics['diversity_change']:+d}")
            print(f"  • 适应度提升: {metrics['fitness_gain']:.3f}")

    def run_interactive_simulation(self):
        """交互式动画模拟（从旧版移植！）"""
        print("\n准备交互式模拟...")

        # 复制当前种群，避免污染原数据
        self.population = self.rng.normal(
            np.mean(self.population) if len(self.population) > 0 else 1.0,
            0.2,
            len(self.population)
        )
        self.population = np.clip(self.population, 0.1, None)

        # 重置历史记录
        self.history = {k: [] for k in self.history.keys()}
        self.breakthrough_generations = []
        self.era_optima = [1.5]
        self.era_widths = [0.8]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
        plt.rcParams['axes.unicode_minus'] = False

        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        # 初始化各子图
        scat = ax1.scatter([], [], alpha=0.6, s=15, c=[], cmap='viridis')
        ax1.set_xlim(0, self.generations)
        ax1.set_ylim(0.5, 5.5)
        ax1.set_xlabel('世代')
        ax1.set_ylabel('特征值')
        ax1.set_title('种群特征值分布')
        ax1.grid(True, alpha=0.2)

        line_avg, = ax2.plot([], [], 'b-', linewidth=2, label='平均特征值')
        ax2.set_xlim(0, self.generations)
        ax2.set_ylim(0.5, 5.5)
        ax2.set_xlabel('世代')
        ax2.set_ylabel('特征值')
        ax2.set_title('平均特征值进化')
        ax2.grid(True, alpha=0.2)
        ax2.legend()

        line_div, = ax3.plot([], [], 'g-', linewidth=2, label='多样性')
        ax3.set_xlim(0, self.generations)
        ax3.set_ylim(0, 20)
        ax3.set_xlabel('世代')
        ax3.set_ylabel('不同特征值数量')
        ax3.set_title('种群多样性')
        ax3.grid(True, alpha=0.2)
        ax3.legend()

        line_pop, = ax4.plot([], [], 'r-', linewidth=2, label='种群大小')
        ax4.set_xlim(0, self.generations)
        ax4.set_ylim(0, self.carrying_capacity * 1.1)
        ax4.set_xlabel('世代')
        ax4.set_ylabel('个体数量')
        ax4.set_title('种群规模')
        ax4.grid(True, alpha=0.2)
        ax4.legend()

        line_fit, = ax5.plot([], [], 'm-', linewidth=2, label='适应度函数')
        ax5.set_xlim(0.5, 5.5)
        ax5.set_ylim(0, 1.2)
        ax5.set_xlabel('特征值')
        ax5.set_ylabel('适应度')
        ax5.set_title('当前环境适应度')
        ax5.grid(True, alpha=0.2)
        ax5.legend()

        info_text = ax6.text(0.05, 0.95, '', transform=ax6.transAxes,
                             fontsize=10, verticalalignment='top', fontfamily='sans-serif')
        ax6.axis('off')
        ax6.set_title('实时信息')

        plt.tight_layout()

        # 存储历史
        historical_populations = []
        current_era = 0
        prev_avg_fitness = 0.0
        prev_population = None

        def update(frame):
            nonlocal current_era, prev_avg_fitness, prev_population

            if frame >= self.generations:
                return scat, line_avg, line_div, line_pop, line_fit, info_text

            # 计算适应度
            fitness = self.compute_fitness(frame, current_era)
            avg_fitness = np.mean(fitness)

            # 记录历史
            self.history['avg_trait'].append(np.mean(self.population))
            self.history['diversity'].append(len(np.unique(np.round(self.population, 1))))
            self.history['population_size'].append(len(self.population))
            self.history['fitness_avg'].append(avg_fitness)
            self.history['era_optima_record'].append(self.era_optima[current_era])
            historical_populations.append(self.population.copy())

            # 检查自适应突破
            if self.tech_mode == 'adaptive':
                if self.check_adaptive_breakthrough(frame, avg_fitness, prev_avg_fitness, current_era):
                    current_era += 1
            else:
                if frame in self.fixed_breakthroughs:
                    print(f"第{frame}代: 固定技术突破!")
                    self.breakthrough_generations.append(frame)
                    shift = self.rng.normal(0.8, 0.3)
                    new_opt = np.mean(self.population) + shift
                    new_width = self.era_widths[current_era] * 0.8
                    self.era_optima.append(new_opt)
                    self.era_widths.append(new_width)
                    current_era += 1

            prev_avg_fitness = avg_fitness
            prev_population = self.population
            self.population = self.reproduce(fitness)

            # ---- 更新图形 ----
            display_range = 30
            start_gen = max(0, frame - display_range)

            # 收集最近 display_range 代的数据
            scatter_gens, scatter_traits, scatter_fitness = [], [], []
            for i in range(start_gen, frame + 1):
                if i < len(historical_populations):
                    pop = historical_populations[i]
                    scatter_gens.extend([i] * len(pop))
                    scatter_traits.extend(pop)
                    # 用当前时代的 opt/width 计算适应度（用于颜色映射）
                    fit_vals = np.exp(-((np.array(pop) - self.era_optima[current_era]) ** 2) /
                                      (2 * self.era_widths[current_era] ** 2))
                    scatter_fitness.extend(fit_vals)

            # 清空 ax1 并重新绘制
            ax1.clear()
            ax1.set_xlim(0, self.generations)
            ax1.set_ylim(0.5, 5.5)
            ax1.set_xlabel('世代')
            ax1.set_ylabel('特征值')
            ax1.set_title('种群特征值分布')
            ax1.grid(True, alpha=0.2)

            # 绘制所有历史突破线（持久化）
            for bg in self.breakthrough_generations:
                ax1.axvline(x=bg, color='red', linestyle='--', alpha=0.5)

            # 绘制散点图
            if scatter_traits:
                scat_new = ax1.scatter(scatter_gens, scatter_traits, c=scatter_fitness,
                                       cmap='viridis', alpha=0.6, s=15, edgecolors='none',
                                       vmin=0, vmax=1)
            else:
                scat_new = ax1.scatter([], [], c=[], cmap='viridis', alpha=0.6, s=15, edgecolors='none')

            # 更新其他线图
            avg_history = [np.mean(historical_populations[i]) for i in range(frame + 1)]
            div_history = [len(np.unique(np.round(historical_populations[i], 1))) for i in range(frame + 1)]
            pop_history = [len(historical_populations[i]) for i in range(frame + 1)]

            line_avg.set_data(range(frame + 1), avg_history)
            line_div.set_data(range(frame + 1), div_history)
            line_pop.set_data(range(frame + 1), pop_history)

            # 当前适应度函数曲线
            trait_range = np.linspace(0.5, 6, 100)
            opt = self.era_optima[current_era]
            width = self.era_widths[current_era]
            fit_curve = np.exp(-((trait_range - opt) ** 2) / (2 * width ** 2))
            line_fit.set_data(trait_range, fit_curve)

            era_name = f"时代{current_era}"
            info = f"""
            当前世代: {frame}/{self.generations}
            当前时代: {era_name}

            种群统计:
            • 种群大小: {len(self.population)}
            • 平均特征值: {np.mean(self.population):.2f}
            • 特征值范围: [{np.min(self.population):.2f}, {np.max(self.population):.2f}]
            • 多样性: {len(np.unique(np.round(self.population, 1)))}
            • 平均适应度: {avg_fitness:.3f}

            技术突破: {self.breakthrough_generations}
            """
            info_text.set_text(info)

            # 额外：如果当前帧是突破，也在其他子图上标注（可选）
            if frame in self.breakthrough_generations:
                for ax in [ax2, ax3, ax4]:
                    ax.axvline(x=frame, color='red', linestyle='--', alpha=0.5)

            return scat_new, line_avg, line_div, line_pop, line_fit, info_text
        anim = FuncAnimation(fig, update, frames=self.generations,
                             interval=150, blit=False, repeat=False)
        plt.suptitle('交互式进化模拟（动态展示）', fontsize=14, fontweight='bold')
        plt.show()
        return anim

    def run_parameter_experiment(self, param_variations: List[Dict]) -> List[Dict]:
        """参数实验框架（修复了 'name' 参数传递错误）"""
        print("\n" + "=" * 70)
        print("开始参数实验")
        print("=" * 70)
        results = []

        base_params = {
            'initial_population': 800,
            'initial_trait_mean': 1.0,
            'initial_trait_std': 0.2,
            'generations': self.generations,
            'mutation_rate': self.mutation_rate,
            'recombination_rate': self.recombination_rate,
            'recombination_noise': self.recombination_noise,  # 新增
            'carrying_capacity': self.carrying_capacity,
            'selection_pressure': self.selection_pressure,
            'env_noise': self.env_noise,
            'tech_mode': self.tech_mode,
            'adaptive_threshold': self.adaptive_threshold,
            'min_breakthrough_gap': self.min_breakthrough_gap,
            'fixed_breakthroughs': self.fixed_breakthroughs,
        }

        for i, params in enumerate(param_variations):
            # 提取实验名称，并从参数字典中移除（不传给模型）
            exp_name = params.get('name', f'实验{i + 1}')
            print(f"\n实验 {i + 1}/{len(param_variations)}: {exp_name}")
            print("-" * 50)

            # 只保留模型构造函数接受的参数（过滤掉 'name' 及其他非预期键）
            pure_params = {k: v for k, v in params.items() if k != 'name'}
            model_params = base_params.copy()
            model_params.update(pure_params)
            model_params['seed'] = 42 + i  # 为每个实验分配不同种子

            exp_model = HybridEvolutionaryModel(**model_params)
            hist = exp_model.simulate()

            result = {
                'name': exp_name,
                'final_avg_trait': hist['avg_trait'][-1],
                'total_change': hist['avg_trait'][-1] - hist['avg_trait'][0],
                'breakthroughs': len(exp_model.breakthrough_generations),
                'extinctions': len(hist['extinction_events']),
                'final_diversity': hist['diversity'][-1],
                'final_population': hist['population_size'][-1],
                'avg_fitness': np.mean(hist['fitness_avg'])
            }
            results.append(result)
            print(f"  结果: 最终特征值={result['final_avg_trait']:.2f}, "
                  f"突破次数={result['breakthroughs']}, 灭绝={result['extinctions']}")

        self._plot_experiment_results(results)
        return results

    def _plot_experiment_results(self, results: List[Dict]):
        names = [r['name'] for r in results]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        metrics = [
            ('final_avg_trait', '最终平均特征值', 'steelblue'),
            ('total_change', '特征值总变化', 'coral'),
            ('breakthroughs', '技术突破次数', 'green'),
            ('extinctions', '灭绝事件次数', 'red'),
            ('final_diversity', '最终多样性', 'purple'),
            ('avg_fitness', '平均适应度', 'orange')
        ]

        for ax, (key, title, color) in zip(axes.flat, metrics):
            values = [r[key] for r in results]
            ax.bar(names, values, color=color)
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=45)

        plt.suptitle('参数实验对比', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# ========================
# 使用示例
if __name__ == "__main__":
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
        tech_mode='adaptive',
        adaptive_threshold=0.008,
        min_breakthrough_gap=15,
        seed=42
    )

    history = model.simulate()
    model.visualize()
    model.visualize_enhanced()
#    model.run_interactive_simulation()   还有bug

    run_exp = input("\n是否运行参数实验？(y/n): ")
    if run_exp.lower() == 'y':
        experiments = [
            {'name': '高突变率', 'mutation_rate': 0.08},
            {'name': '低突变率', 'mutation_rate': 0.005},
            {'name': '高选择压力', 'selection_pressure': 3.5},
            {'name': '低选择压力', 'selection_pressure': 1.0},
            {'name': '小承载力', 'carrying_capacity': 800},
            {'name': '无重组', 'recombination_rate': 0.0}
        ]
        model.run_parameter_experiment(experiments)
