"""
EML Experiments - EML实验管理模块

提供EML研究的实验管理功能，包括实验配置、运行、结果记录、
对比分析和超参数搜索。

核心功能:
- 实验配置管理：定义和验证实验参数
- 实验运行引擎：自动化执行实验流程
- 结果记录与存储：结构化保存实验结果
- 对比分析：多实验结果对比与可视化数据生成
- 超参数搜索：网格搜索和随机搜索

⚠️ 研究用途警告: 本模块为实验性实现
"""

import os
import json
import time
import copy
import random
import math
import statistics
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime


# ============================================================
# 实验状态枚举
# ============================================================

class ExperimentStatus(Enum):
    """实验状态"""
    PENDING = auto()       # 待运行
    RUNNING = auto()       # 运行中
    COMPLETED = auto()     # 已完成
    FAILED = auto()        # 失败
    SKIPPED = auto()       # 跳过


# ============================================================
# 实验配置
# ============================================================

@dataclass
class ExperimentConfig:
    """
    单个实验的配置

    定义一个EML实验的所有参数，包括数据配置、模型配置和训练配置。
    """
    # 实验基本信息
    name: str = ""                           # 实验名称
    description: str = ""                    # 实验描述
    tags: List[str] = field(default_factory=list)  # 实验标签

    # 数据配置
    data_size: int = 100                     # 数据集大小
    data_range: Tuple[float, float] = (0.1, 10.0)  # 数据范围
    noise_level: float = 0.0                 # 噪声水平
    target_function: str = "exp"             # 目标函数类型

    # EML配置
    safety_level: str = "MODERATE"           # 安全级别
    min_y: float = 1e-10                     # y最小值
    max_y: float = 1e10                      # y最大值
    use_stable: bool = True                  # 使用稳定版本

    # 遗传编程配置
    gp_population_size: int = 100            # GP种群大小
    gp_max_generations: int = 200            # GP最大代数
    gp_max_tree_depth: int = 6               # GP最大树深度
    gp_crossover_rate: float = 0.8           # GP交叉率
    gp_mutation_rate: float = 0.15           # GP变异率

    # 评估配置
    metrics: List[str] = field(default_factory=lambda: ["mse", "mae", "max_error"])
    cv_folds: int = 1                        # 交叉验证折数

    # 随机种子
    seed: int = 42

    def validate(self) -> Tuple[bool, str]:
        """
        验证配置参数的有效性

        Returns:
            (是否有效, 错误信息)
        """
        if not self.name:
            return False, "实验名称不能为空"

        if self.data_size <= 0:
            return False, f"数据大小必须为正数，当前: {self.data_size}"

        if self.data_range[0] >= self.data_range[1]:
            return False, f"数据范围无效: {self.data_range}"

        if self.noise_level < 0:
            return False, f"噪声水平不能为负数: {self.noise_level}"

        if self.gp_population_size <= 0:
            return False, f"种群大小必须为正数: {self.gp_population_size}"

        if self.gp_max_generations <= 0:
            return False, f"最大代数必须为正数: {self.gp_max_generations}"

        if not (0 <= self.gp_crossover_rate <= 1):
            return False, f"交叉率必须在[0,1]范围内: {self.gp_crossover_rate}"

        if not (0 <= self.gp_mutation_rate <= 1):
            return False, f"变异率必须在[0,1]范围内: {self.gp_mutation_rate}"

        if self.cv_folds < 1:
            return False, f"交叉验证折数必须>=1: {self.cv_folds}"

        return True, ""

    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        d = asdict(self)
        # 处理tuple类型的序列化
        d['data_range'] = list(self.data_range)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        """从字典创建配置"""
        d = copy.deepcopy(d)
        if 'data_range' in d and isinstance(d['data_range'], list):
            d['data_range'] = tuple(d['data_range'])
        return cls(**d)


# ============================================================
# 实验结果
# ============================================================

@dataclass
class ExperimentResult:
    """
    单个实验的结果

    记录实验运行的完整结果，包括指标、模型信息和元数据。
    """
    # 实验标识
    experiment_id: str = ""                  # 实验唯一ID
    experiment_name: str = ""                # 实验名称

    # 运行状态
    status: str = "PENDING"                  # 运行状态
    start_time: str = ""                     # 开始时间
    end_time: str = ""                       # 结束时间
    elapsed_seconds: float = 0.0             # 耗时（秒）

    # 评估指标
    metrics: Dict[str, float] = field(default_factory=dict)  # 指标名称 -> 值

    # 模型结果
    best_expression: str = ""                # 最优表达式
    best_fitness: float = float('inf')       # 最优适应度
    generations_run: int = 0                 # 实际运行代数

    # 额外信息
    config: Dict[str, Any] = field(default_factory=dict)     # 使用的配置
    error_message: str = ""                  # 错误信息（失败时）
    notes: str = ""                          # 备注

    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentResult':
        """从字典创建结果"""
        return cls(**d)


# ============================================================
# 超参数搜索空间
# ============================================================

@dataclass
class SearchSpace:
    """
    超参数搜索空间定义

    定义每个超参数的搜索范围和类型。
    """
    # 参数名 -> (类型, 取值列表或范围)
    # 类型: 'choice' 表示离散选择, 'range' 表示连续范围
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_choice(self, name: str, values: List[Any]):
        """
        添加离散选择参数

        Args:
            name: 参数名
            values: 可选值列表
        """
        self.parameters[name] = {
            'type': 'choice',
            'values': values,
        }

    def add_range(self, name: str, low: float, high: float,
                  log_scale: bool = False):
        """
        添加连续范围参数

        Args:
            name: 参数名
            low: 最小值
            high: 最大值
            log_scale: 是否使用对数尺度
        """
        self.parameters[name] = {
            'type': 'range',
            'low': low,
            'high': high,
            'log_scale': log_scale,
        }

    def sample_random(self) -> Dict[str, Any]:
        """
        从搜索空间中随机采样一组参数

        Returns:
            参数字典
        """
        result = {}
        for name, param_def in self.parameters.items():
            if param_def['type'] == 'choice':
                result[name] = random.choice(param_def['values'])
            elif param_def['type'] == 'range':
                low = param_def['low']
                high = param_def['high']
                if param_def.get('log_scale', False):
                    # 对数尺度采样
                    log_low = math.log(low)
                    log_high = math.log(high)
                    value = math.exp(random.uniform(log_low, log_high))
                else:
                    value = random.uniform(low, high)
                result[name] = value
        return result

    def generate_grid(self, num_samples_per_param: int = 3) -> List[Dict[str, Any]]:
        """
        生成网格搜索参数组合

        Args:
            num_samples_per_param: 每个参数的采样数量

        Returns:
            参数组合列表
        """
        import itertools

        param_values = {}
        for name, param_def in self.parameters.items():
            if param_def['type'] == 'choice':
                param_values[name] = param_def['values']
            elif param_def['type'] == 'range':
                low = param_def['low']
                high = param_def['high']
                if param_def.get('log_scale', False):
                    log_low = math.log(low)
                    log_high = math.log(high)
                    values = [math.exp(v) for v in
                              [log_low + i * (log_high - log_low) / (num_samples_per_param - 1)
                               for i in range(num_samples_per_param)]]
                else:
                    values = [low + i * (high - low) / (num_samples_per_param - 1)
                              for i in range(num_samples_per_param)]
                param_values[name] = values

        # 生成所有组合
        keys = list(param_values.keys())
        value_lists = [param_values[k] for k in keys]
        combinations = []

        for combo in itertools.product(*value_lists):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)

        return combinations


# ============================================================
# 对比分析结果
# ============================================================

@dataclass
class ComparisonResult:
    """对比分析结果"""
    experiment_names: List[str] = field(default_factory=list)
    metric_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ranking: List[str] = field(default_factory=list)  # 按主要指标排序
    best_experiment: str = ""
    summary: str = ""


# ============================================================
# EML实验管理器主类
# ============================================================

class EMLExperiments:
    """
    EML实验管理器

    统一管理EML研究的实验生命周期，包括:
    - 创建和配置实验
    - 运行实验（支持批量运行）
    - 记录和存储结果
    - 对比分析多个实验
    - 超参数搜索（网格搜索、随机搜索）

    使用示例:
        >>> mgr = EMLExperiments()
        >>> # 创建实验配置
        >>> config = ExperimentConfig(name="exp_baseline", target_function="exp")
        >>> # 运行实验
        >>> result = mgr.run_experiment(config)
        >>> # 查看结果
        >>> print(result.metrics)
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化实验管理器

        Args:
            output_dir: 结果输出目录，为None时使用默认目录
        """
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "experiment_results"
        )

        # 实验注册表: experiment_id -> ExperimentResult
        self._results: Dict[str, ExperimentResult] = {}

        # 实验配置注册表: experiment_id -> ExperimentConfig
        self._configs: Dict[str, ExperimentConfig] = {}

        # 实验计数器（用于生成唯一ID）
        self._experiment_counter = 0

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 实验配置
    # ----------------------------------------------------------

    def create_config(self, name: str, **kwargs) -> ExperimentConfig:
        """
        创建实验配置

        快捷方法，用于创建带有指定参数的实验配置。

        Args:
            name: 实验名称
            **kwargs: 配置参数，覆盖默认值

        Returns:
            验证后的实验配置

        Raises:
            ValueError: 配置验证失败时
        """
        config = ExperimentConfig(name=name, **kwargs)
        valid, msg = config.validate()
        if not valid:
            raise ValueError(f"实验配置验证失败: {msg}")
        return config

    def create_config_batch(self, base_name: str,
                            param_variations: List[Dict[str, Any]]) -> List[ExperimentConfig]:
        """
        批量创建实验配置

        基于基础配置，通过参数变体生成多个配置。

        Args:
            base_name: 基础名称
            param_variations: 参数变体列表，每个变体是一个参数字典

        Returns:
            实验配置列表

        示例:
            >>> configs = mgr.create_config_batch("test", [
            ...     {"gp_population_size": 50},
            ...     {"gp_population_size": 100},
            ...     {"gp_population_size": 200},
            ... ])
        """
        configs = []
        for i, variation in enumerate(param_variations):
            name = f"{base_name}_variant_{i}"
            config = self.create_config(name=name, **variation)
            configs.append(config)
        return configs

    # ----------------------------------------------------------
    # 目标函数生成
    # ----------------------------------------------------------

    def _generate_target_data(self, config: ExperimentConfig) -> Tuple[List[float], List[float]]:
        """
        根据配置生成目标函数的训练数据

        Args:
            config: 实验配置

        Returns:
            (train_x, train_y) 训练数据
        """
        random.seed(config.seed)

        # 生成输入数据
        low, high = config.data_range
        train_x = [random.uniform(low, high) for _ in range(config.data_size)]
        train_x.sort()

        # 根据目标函数类型生成输出
        target_fn = self._get_target_function(config.target_function)
        train_y = [target_fn(x) for x in train_x]

        # 添加噪声
        if config.noise_level > 0:
            train_y = [y + random.gauss(0, config.noise_level) for y in train_y]

        return train_x, train_y

    def _get_target_function(self, func_type: str) -> Callable[[float], float]:
        """
        获取目标函数

        Args:
            func_type: 函数类型标识符

        Returns:
            目标函数
        """
        functions = {
            'exp': lambda x: math.exp(min(x, 700)),          # 指数函数
            'log': lambda x: math.log(max(x, 1e-10)),        # 对数函数
            'linear': lambda x: 2.0 * x + 1.0,               # 线性函数
            'quadratic': lambda x: x ** 2 + x + 1.0,         # 二次函数
            'sin': lambda x: math.sin(x),                     # 正弦函数
            'eml_basic': lambda x: math.exp(x) - math.log(max(x, 1e-10)),  # 基础EML
            'sqrt': lambda x: math.sqrt(max(x, 0)),           # 平方根
            'reciprocal': lambda x: 1.0 / max(x, 1e-10),     # 倒数
        }

        if func_type not in functions:
            raise ValueError(f"未知的目标函数类型: {func_type}，"
                             f"可选: {list(functions.keys())}")

        return functions[func_type]

    # ----------------------------------------------------------
    # 实验运行
    # ----------------------------------------------------------

    def run_experiment(self, config: ExperimentConfig,
                       verbose: bool = False) -> ExperimentResult:
        """
        运行单个实验

        完整的实验流程:
        1. 验证配置
        2. 生成训练数据
        3. 运行EML遗传编程
        4. 评估结果
        5. 记录结果

        Args:
            config: 实验配置
            verbose: 是否输出详细信息

        Returns:
            实验结果
        """
        # 生成唯一ID
        self._experiment_counter += 1
        exp_id = f"{config.name}_{self._experiment_counter:04d}"

        # 创建结果对象
        result = ExperimentResult(
            experiment_id=exp_id,
            experiment_name=config.name,
            status="RUNNING",
            start_time=datetime.now().isoformat(),
            config=config.to_dict(),
        )

        if verbose:
            print(f"[实验] 开始运行: {config.name} (ID: {exp_id})")

        try:
            # 验证配置
            valid, msg = config.validate()
            if not valid:
                raise ValueError(f"配置验证失败: {msg}")

            # 生成训练数据
            train_x, train_y = self._generate_target_data(config)

            if verbose:
                print(f"[实验] 数据生成完成: {len(train_x)} 个样本")

            # 运行EML遗传编程
            from .genetic import EMLGeneticProgramming, GPConfig

            gp_config = GPConfig(
                population_size=config.gp_population_size,
                max_generations=config.gp_max_generations,
                max_tree_depth=config.gp_max_tree_depth,
                crossover_rate=config.gp_crossover_rate,
                mutation_rate=config.gp_mutation_rate,
                seed=config.seed,
            )

            gp = EMLGeneticProgramming(gp_config)
            gp_result = gp.evolve(train_x, train_y, verbose=verbose)

            # 计算评估指标
            metrics = self._compute_metrics(
                gp_result['best_individual'], train_x, train_y
            )

            # 填充结果
            result.status = "COMPLETED"
            result.metrics = metrics
            result.best_expression = gp_result['best_expression']
            result.best_fitness = gp_result['best_fitness']
            result.generations_run = gp_result['generations']

            if verbose:
                print(f"[实验] 完成: {config.name}")
                print(f"  最优表达式: {result.best_expression}")
                print(f"  适应度: {result.best_fitness:.6f}")
                for metric_name, metric_val in metrics.items():
                    print(f"  {metric_name}: {metric_val:.6f}")

        except Exception as e:
            result.status = "FAILED"
            result.error_message = str(e)
            if verbose:
                print(f"[实验] 失败: {config.name} - {e}")

        # 记录结束时间
        result.end_time = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(result.start_time)
        end_dt = datetime.fromisoformat(result.end_time)
        result.elapsed_seconds = (end_dt - start_dt).total_seconds()

        # 保存结果
        self._results[exp_id] = result
        self._configs[exp_id] = config

        return result

    def run_batch(self, configs: List[ExperimentConfig],
                  verbose: bool = False) -> List[ExperimentResult]:
        """
        批量运行实验

        按顺序运行多个实验，适用于对比实验。

        Args:
            configs: 实验配置列表
            verbose: 是否输出详细信息

        Returns:
            实验结果列表
        """
        results = []
        total = len(configs)

        for i, config in enumerate(configs):
            if verbose:
                print(f"\n{'=' * 50}")
                print(f"[批量] 实验 {i + 1}/{total}: {config.name}")
                print(f"{'=' * 50}")

            result = self.run_experiment(config, verbose=verbose)
            results.append(result)

        if verbose:
            print(f"\n[批量] 全部 {total} 个实验完成")

        return results

    # ----------------------------------------------------------
    # 指标计算
    # ----------------------------------------------------------

    def _compute_metrics(self, individual, train_x: List[float],
                         train_y: List[float]) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            individual: 表达式树个体
            train_x: 输入数据
            train_y: 目标数据

        Returns:
            指标字典
        """
        predictions = []
        for x in train_x:
            try:
                pred = individual.evaluate(x)
                if math.isfinite(pred):
                    predictions.append(pred)
                else:
                    predictions.append(float('nan'))
            except Exception:
                predictions.append(float('nan'))

        metrics = {}
        valid_pairs = [(p, t) for p, t in zip(predictions, train_y)
                       if math.isfinite(p)]

        if valid_pairs:
            preds = [p for p, _ in valid_pairs]
            targets = [t for _, t in valid_pairs]

            # 均方误差 (MSE)
            mse = statistics.mean((p - t) ** 2 for p, t in valid_pairs)
            metrics['mse'] = mse

            # 均方根误差 (RMSE)
            metrics['rmse'] = math.sqrt(mse)

            # 平均绝对误差 (MAE)
            metrics['mae'] = statistics.mean(abs(p - t) for p, t in valid_pairs)

            # 最大误差
            metrics['max_error'] = max(abs(p - t) for p, t in valid_pairs)

            # R平方决定系数
            mean_target = statistics.mean(targets)
            ss_tot = sum((t - mean_target) ** 2 for t in targets)
            ss_res = sum((p - t) ** 2 for p, t in valid_pairs)
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics['r_squared'] = r_squared

            # 有效预测比例
            metrics['valid_ratio'] = len(valid_pairs) / len(train_x)
        else:
            metrics['mse'] = float('inf')
            metrics['rmse'] = float('inf')
            metrics['mae'] = float('inf')
            metrics['max_error'] = float('inf')
            metrics['r_squared'] = float('-inf')
            metrics['valid_ratio'] = 0.0

        return metrics

    # ----------------------------------------------------------
    # 结果记录与存储
    # ----------------------------------------------------------

    def save_results(self, filepath: Optional[str] = None) -> str:
        """
        保存所有实验结果到JSON文件

        Args:
            filepath: 保存路径，为None时使用默认路径

        Returns:
            实际保存的文件路径
        """
        if filepath is None:
            filepath = os.path.join(
                self.output_dir,
                f"eml_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        # 序列化结果
        data = {
            'saved_at': datetime.now().isoformat(),
            'total_experiments': len(self._results),
            'experiments': [],
        }

        for exp_id, result in self._results.items():
            data['experiments'].append(result.to_dict())

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def load_results(self, filepath: str) -> int:
        """
        从JSON文件加载实验结果

        Args:
            filepath: 文件路径

        Returns:
            加载的实验数量
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for exp_data in data.get('experiments', []):
            result = ExperimentResult.from_dict(exp_data)
            self._results[result.experiment_id] = result

            # 尝试恢复配置
            if result.config:
                try:
                    config = ExperimentConfig.from_dict(result.config)
                    self._configs[result.experiment_id] = config
                except Exception:
                    pass

            count += 1
            # 更新计数器
            num = int(result.experiment_id.split('_')[-1])
            if num > self._experiment_counter:
                self._experiment_counter = num

        return count

    def get_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """
        获取指定实验的结果

        Args:
            experiment_id: 实验ID

        Returns:
            实验结果，不存在时返回None
        """
        return self._results.get(experiment_id)

    def get_all_results(self) -> List[ExperimentResult]:
        """
        获取所有实验结果

        Returns:
            实验结果列表
        """
        return list(self._results.values())

    def get_successful_results(self) -> List[ExperimentResult]:
        """
        获取所有成功完成的实验结果

        Returns:
            成功的实验结果列表
        """
        return [r for r in self._results.values() if r.status == "COMPLETED"]

    # ----------------------------------------------------------
    # 对比分析
    # ----------------------------------------------------------

    def compare_experiments(self, experiment_ids: Optional[List[str]] = None,
                            primary_metric: str = 'mse') -> Dict[str, Any]:
        """
        对比多个实验的结果

        生成详细的对比分析报告，包括各指标的排名和统计分析。

        Args:
            experiment_ids: 要对比的实验ID列表，为None时对比所有成功实验
            primary_metric: 主要排序指标

        Returns:
            对比分析结果字典
        """
        # 确定要对比的实验
        if experiment_ids is None:
            results = self.get_successful_results()
        else:
            results = [self._results[eid] for eid in experiment_ids
                       if eid in self._results and self._results[eid].status == "COMPLETED"]

        if not results:
            return {
                'error': '没有可对比的实验结果',
                'experiments': [],
                'comparison': {},
            }

        # 收集所有指标名称
        all_metrics = set()
        for r in results:
            all_metrics.update(r.metrics.keys())
        all_metrics = sorted(all_metrics)

        # 构建对比表
        comparison = {}
        for metric in all_metrics:
            values = {}
            for r in results:
                if metric in r.metrics:
                    values[r.experiment_name] = r.metrics[metric]

            comparison[metric] = values

        # 按主要指标排序
        # 对于MSE/MAE等越小越好的指标，升序排列
        if primary_metric in comparison:
            sorted_experiments = sorted(
                comparison[primary_metric].items(),
                key=lambda x: x[1]
            )
            ranking = [name for name, _ in sorted_experiments]
            best_name = ranking[0]
        else:
            ranking = [r.experiment_name for r in results]
            best_name = ranking[0] if ranking else ""

        # 生成摘要
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("EML 实验对比分析报告")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        summary_lines.append(f"对比实验数量: {len(results)}")
        summary_lines.append(f"主要排序指标: {primary_metric}")
        summary_lines.append(f"最优实验: {best_name}")
        summary_lines.append("")

        # 指标对比表
        header = f"  {'实验名称':<30s}"
        for metric in all_metrics:
            header += f" | {metric:>12s}"
        summary_lines.append(header)
        summary_lines.append("  " + "-" * (30 + len(all_metrics) * 16))

        for r in results:
            row = f"  {r.experiment_name:<30s}"
            for metric in all_metrics:
                val = r.metrics.get(metric, float('nan'))
                if math.isfinite(val):
                    row += f" | {val:>12.6f}"
                else:
                    row += f" | {'N/A':>12s}"
            summary_lines.append(row)

        summary_lines.append("")

        # 排名
        summary_lines.append("排名 (按主要指标升序):")
        for i, name in enumerate(ranking):
            marker = " <-- 最优" if i == 0 else ""
            val = comparison.get(primary_metric, {}).get(name, float('nan'))
            if math.isfinite(val):
                summary_lines.append(f"  {i + 1}. {name}: {val:.6f}{marker}")
            else:
                summary_lines.append(f"  {i + 1}. {name}: N/A{marker}")

        summary_lines.append("")
        summary_lines.append("=" * 60)

        return {
            'experiments': [r.experiment_name for r in results],
            'metrics': all_metrics,
            'comparison': comparison,
            'ranking': ranking,
            'best_experiment': best_name,
            'primary_metric': primary_metric,
            'summary': "\n".join(summary_lines),
        }

    # ----------------------------------------------------------
    # 超参数搜索
    # ----------------------------------------------------------

    def grid_search(self, base_config: ExperimentConfig,
                    search_space: SearchSpace,
                    verbose: bool = False) -> Dict[str, Any]:
        """
        网格搜索

        在给定的搜索空间内，穷举所有参数组合进行实验。

        Args:
            base_config: 基础配置
            search_space: 搜索空间定义
            verbose: 是否输出详细信息

        Returns:
            搜索结果字典，包含:
            - all_results: 所有实验结果
            - best_params: 最优参数
            - best_result: 最优实验结果
            - summary: 搜索摘要
        """
        # 生成参数组合
        combinations = search_space.generate_grid(num_samples_per_param=3)

        if verbose:
            print(f"[网格搜索] 参数组合数量: {len(combinations)}")

        # 运行所有组合
        all_results = []
        for i, params in enumerate(combinations):
            # 创建配置副本并更新参数
            config = copy.deepcopy(base_config)
            config.name = f"{base_config.name}_grid_{i}"

            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            if verbose:
                print(f"\n[网格搜索] 组合 {i + 1}/{len(combinations)}: {params}")

            result = self.run_experiment(config, verbose=False)
            result.notes = f"参数: {params}"
            all_results.append((params, result))

        # 找到最优结果
        successful = [(p, r) for p, r in all_results if r.status == "COMPLETED"]

        if successful:
            best_params, best_result = min(
                successful,
                key=lambda x: x[1].metrics.get('mse', float('inf'))
            )
        else:
            best_params = {}
            best_result = None

        # 生成摘要
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("EML 网格搜索结果")
        summary_lines.append("=" * 60)
        summary_lines.append(f"总组合数: {len(combinations)}")
        summary_lines.append(f"成功实验数: {len(successful)}")
        summary_lines.append("")

        if best_result:
            summary_lines.append(f"最优参数: {best_params}")
            summary_lines.append(f"最优MSE: {best_result.metrics.get('mse', 'N/A')}")
            summary_lines.append(f"最优表达式: {best_result.best_expression}")
        else:
            summary_lines.append("没有成功的实验")

        summary_lines.append("=" * 60)

        return {
            'search_type': 'grid',
            'total_combinations': len(combinations),
            'all_results': [(p, r.to_dict()) for p, r in all_results],
            'best_params': best_params,
            'best_result': best_result.to_dict() if best_result else None,
            'summary': "\n".join(summary_lines),
        }

    def random_search(self, base_config: ExperimentConfig,
                      search_space: SearchSpace,
                      n_trials: int = 20,
                      verbose: bool = False) -> Dict[str, Any]:
        """
        随机搜索

        从搜索空间中随机采样指定数量的参数组合进行实验。

        Args:
            base_config: 基础配置
            search_space: 搜索空间定义
            n_trials: 随机试验次数
            verbose: 是否输出详细信息

        Returns:
            搜索结果字典
        """
        if verbose:
            print(f"[随机搜索] 试验次数: {n_trials}")

        all_results = []
        for i in range(n_trials):
            # 随机采样参数
            params = search_space.sample_random()

            # 创建配置
            config = copy.deepcopy(base_config)
            config.name = f"{base_config.name}_random_{i}"

            for key, value in params.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            if verbose:
                print(f"\n[随机搜索] 试验 {i + 1}/{n_trials}: {params}")

            result = self.run_experiment(config, verbose=False)
            result.notes = f"参数: {params}"
            all_results.append((params, result))

        # 找到最优结果
        successful = [(p, r) for p, r in all_results if r.status == "COMPLETED"]

        if successful:
            best_params, best_result = min(
                successful,
                key=lambda x: x[1].metrics.get('mse', float('inf'))
            )
        else:
            best_params = {}
            best_result = None

        # 生成摘要
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("EML 随机搜索结果")
        summary_lines.append("=" * 60)
        summary_lines.append(f"试验次数: {n_trials}")
        summary_lines.append(f"成功实验数: {len(successful)}")
        summary_lines.append("")

        if best_result:
            summary_lines.append(f"最优参数: {best_params}")
            summary_lines.append(f"最优MSE: {best_result.metrics.get('mse', 'N/A')}")
            summary_lines.append(f"最优表达式: {best_result.best_expression}")
        else:
            summary_lines.append("没有成功的实验")

        summary_lines.append("=" * 60)

        return {
            'search_type': 'random',
            'n_trials': n_trials,
            'all_results': [(p, r.to_dict()) for p, r in all_results],
            'best_params': best_params,
            'best_result': best_result.to_dict() if best_result else None,
            'summary': "\n".join(summary_lines),
        }

    # ----------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------

    def clear_results(self):
        """清除所有实验结果"""
        self._results.clear()
        self._configs.clear()
        self._experiment_counter = 0

    def get_experiment_count(self) -> int:
        """获取已运行的实验总数"""
        return len(self._results)

    def get_summary(self) -> str:
        """
        获取实验管理器摘要

        Returns:
            格式化的摘要字符串
        """
        total = len(self._results)
        completed = sum(1 for r in self._results.values() if r.status == "COMPLETED")
        failed = sum(1 for r in self._results.values() if r.status == "FAILED")

        lines = []
        lines.append("=" * 60)
        lines.append("EML 实验管理器摘要")
        lines.append("=" * 60)
        lines.append(f"  总实验数: {total}")
        lines.append(f"  已完成: {completed}")
        lines.append(f"  失败: {failed}")
        lines.append(f"  输出目录: {self.output_dir}")
        lines.append("")

        if self._results:
            lines.append("  实验列表:")
            for exp_id, result in self._results.items():
                status_icon = {
                    "COMPLETED": "[OK]",
                    "FAILED": "[FAIL]",
                    "RUNNING": "[RUN]",
                    "PENDING": "[WAIT]",
                }.get(result.status, "[??]")
                lines.append(
                    f"    {status_icon} {exp_id}: "
                    f"MSE={result.metrics.get('mse', 'N/A')}"
                )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================
# 模块自测
# ============================================================

if __name__ == "__main__":
    print("EML Experiments - 模块自测")
    print("=" * 60)

    # 创建实验管理器
    mgr = EMLExperiments()

    # 创建实验配置
    config = mgr.create_config(
        name="test_exp",
        target_function="exp",
        data_size=30,
        gp_population_size=30,
        gp_max_generations=20,
        gp_max_tree_depth=3,
    )

    # 运行实验
    result = mgr.run_experiment(config, verbose=True)

    # 输出摘要
    print("\n" + mgr.get_summary())
