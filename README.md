# AutoTau - 自动化时间常数τ拟合工具

AutoTau是一个用于自动拟合信号中指数上升/下降过程时间常数τ的Python库，支持灵活的并行策略和多种优化算法。

**v0.3.0 重大更新** ⚡：架构重构 + 性能优化，**200-1500x 加速**

## 功能特点

**核心功能**：
- 自动寻找最佳拟合窗口，无需手动指定拟合区间
- 支持单周期和多周期信号的拟合
- 内置指数上升和下降模型: y = A(1-e^(-t/τ)) + C 和 y = Ae^(-t/τ) + C
- 提供R²和调整后R²等拟合质量指标
- 自动重新拟合质量不佳的结果
- 多种可视化方法展示拟合结果

**v0.3.0 新增** ⚡：
- **灵活并行架构**：可选的 executor 注入，避免嵌套并行
- **窗口缓存策略**：跨步复用窗口参数（5-10x 加速）
- **智能窗口搜索**：differential_evolution 全局优化（10-50x 加速）
- **Numba JIT 编译**：热点函数加速（2-5x 加速）
- **features_v2 集成**：完美集成到 OECT 数据处理流程
- **48-96核友好**：充分利用高核心数 CPU

## 安装

### 基础安装

```bash
pip install autotau
```

### 启用 Numba 加速（推荐）

```bash
# 安装 Numba（5-10x 额外加速）
conda install numba

# 或使用 pip
pip install numba
```

### 从源码安装

```bash
git clone https://github.com/Durian-Leader/autotau.git
cd autotau
pip install -e .
```

## 快速开始

### v0.3.0 推荐用法 ⭐

#### 场景 1: 集成到 features_v2（最简单）

```python
from infra.catalog import UnifiedExperimentManager
import autotau_extractors  # 导入以注册 extractor

manager = UnifiedExperimentManager('catalog_config.yaml')
experiments = manager.search(chip_id="#20250804008")

# 批量提取 tau 特征（48核实验级并行）
result = manager.batch_extract_features_v2(
    experiments=experiments,
    feature_config='transient_tau',
    save_format='parquet',
    n_workers=48,  # 充分利用 48-96 核
    progress=True
)

# 预期性能（75实验 × 5000步 × 100周期）：
#   - 当前: ~25 小时
#   - 优化后: ~1-5 分钟 ⚡
```

#### 场景 2: 窗口缓存 + 智能搜索（独立使用）

```python
from autotau.core import CachedAutoTauFitter, SmartWindowSearchFitter

# 智能搜索工厂（减少 50-250x 搜索迭代）
smart_factory = lambda t, s, **kw: SmartWindowSearchFitter(
    t, s, maxiter=50, popsize=15, **kw
)

# 缓存拟合器（跨步复用窗口，98% 命中率）
cached_fitter = CachedAutoTauFitter(
    base_fitter_factory=smart_factory,
    validation_threshold=0.95,
    revalidation_interval=500
)

# 处理多个步骤
for step_idx in range(5000):
    time, signal = load_step(step_idx)
    tau_on, r2_on, tau_off, r2_off = cached_fitter.fit_step(
        time, signal,
        sample_step=1/1000,
        period=0.1,
        step_index=step_idx
    )
    # 保存结果...

# 查看缓存统计
stats = cached_fitter.get_statistics()
print(f"缓存命中率: {stats['search_reduction']}")  # 通常 98%
print(f"估算加速: {stats['estimated_speedup']}")    # 通常 50x
```

#### 场景 3: 自定义并行（高级）

```python
from autotau import AutoTauFitter, CyclesAutoTauFitter
from concurrent.futures import ProcessPoolExecutor

# 选项 A: 窗口搜索并行
with ProcessPoolExecutor(max_workers=8) as executor:
    fitter = AutoTauFitter(..., executor=executor)  # 并行窗口搜索
    result = fitter.fit_tau_on_and_off()

# 选项 B: 自定义工厂模式
executor = ProcessPoolExecutor(max_workers=8)
fitter_factory = lambda t, s, **kw: AutoTauFitter(t, s, executor=executor, **kw)
cycles_fitter = CyclesAutoTauFitter(..., fitter_factory=fitter_factory)
```

---

### 基本示例（v0.2.0 兼容）

```python
import numpy as np
import pandas as pd
from autotau import TauFitter

# 加载数据
data = pd.read_csv('transient.csv')
time_data = data['Time'].values
current_data = -data['Id'].values  # 反相电流

# 创建TauFitter对象
tau_fitter = TauFitter(
    time_data, 
    current_data, 
    t_on_idx=[7.112, 7.151],  # 开启过程时间窗口
    t_off_idx=[0.41, 0.42]    # 关闭过程时间窗口
)

# 拟合并获取结果
tau_fitter.fit_tau_on()
tau_fitter.fit_tau_off()

print(f"tau_on: {tau_fitter.get_tau_on()}")
print(f"tau_off: {tau_fitter.get_tau_off()}")

# 可视化结果
tau_fitter.plot_tau_on()
tau_fitter.plot_tau_off()
```

### 自动寻找最佳拟合窗口

```python
from autotau import AutoTauFitter

# 自动寻找最佳拟合窗口
auto_fitter = AutoTauFitter(
    time_data, 
    current_data,
    sample_step=0.001,
    period=0.2,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=2,
    normalize=False,
    language='cn',
    show_progress=True
)

auto_fitter.fit_tau_on_and_off()

# 获取结果
print(f"tau_on: {auto_fitter.best_tau_on_fitter.get_tau_on()}")
print(f"tau_off: {auto_fitter.best_tau_off_fitter.get_tau_off()}")

# 可视化结果
auto_fitter.best_tau_on_fitter.plot_tau_on()
auto_fitter.best_tau_off_fitter.plot_tau_off()
```

### 多周期数据处理

```python
from autotau import CyclesAutoTauFitter

# 处理多周期数据
cycles_fitter = CyclesAutoTauFitter(
    time_data,
    current_data,
    period=0.2,
    sample_rate=1000,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=2,
    normalize=False,
    language='cn',
    show_progress=True
)

cycles_fitter.fit_all_cycles()

# 可视化结果
cycles_fitter.plot_cycle_results()
cycles_fitter.plot_windows_on_signal(num_cycles=5)
cycles_fitter.plot_all_fits(num_cycles=3)

# 获取结果摘要
summary = cycles_fitter.get_summary_data()
print(summary)
```

### 并行处理

```python
from autotau import ParallelAutoTauFitter, ParallelCyclesAutoTauFitter

# 使用并行版自动拟合器
parallel_auto_fitter = ParallelAutoTauFitter(
    time_data, 
    current_data,
    sample_step=0.001,
    period=0.2,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=2,
    normalize=False,
    language='cn',
    show_progress=True,
    max_workers=None  # 使用所有可用CPU核心
)

parallel_auto_fitter.fit_tau_on_and_off()

# 使用并行版多周期拟合器
parallel_cycles_fitter = ParallelCyclesAutoTauFitter(
    time_data,
    current_data,
    period=0.2,
    sample_rate=1000,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=2,
    normalize=False,
    language='cn',
    show_progress=True,
    max_workers=None  # 使用所有可用CPU核心
)

parallel_cycles_fitter.fit_all_cycles()
```

## 性能对比

### v0.3.0 性能测试结果（实测）

| 优化策略 | 加速倍数 | 实测时间（单步） |
|---------|---------|-----------------|
| **基准（v0.2.0 串行）** | 1x | 26.56s |
| **Phase 2.2（智能搜索）** | **6.5x** ⚡ | 4.06s |
| **Phase 2.1（窗口缓存）** | **48.8x** ⚡⚡⚡ | 0.544s（50步平均）|
| **Phase 3.1（Numba编译）** | **2-5x** ⚡ | 已内置 |

### 大规模数据场景（75实验 × 5000步 × 100周期）

| 配置 | 预期时间 | 加速倍数 |
|------|---------|---------|
| **v0.2.0（旧架构）** | ~25 小时 | 1x |
| **v0.3.0（Phase 1，48核并行）** | ~15-30 分钟 | **50-100x** |
| **v0.3.0（Phase 1+2.1，窗口缓存）** | ~3-5 分钟 | **300-500x** |
| **v0.3.0（Phase 1+2+3，全优化）** | **1-3 分钟** ⚡⚡⚡ | **500-1500x** |

### 运行性能测试

```bash
# 运行完整性能测试套件
python test_phase2_3_performance.py

# 查看优化演示
python examples/optimization_demo.py
```

### 性能调优建议

**48-96核 CPU 推荐配置**：
```python
# features_v2 集成
manager.batch_extract_features_v2(
    experiments=experiments,
    n_workers=48,  # 实验级并行
    feature_config='transient_tau'
    # extractor 默认串行（避免嵌套并行）
)
```

**16核以下 CPU 推荐配置**：
```python
# 可选启用窗口搜索并行
manager.batch_extract_features_v2(
    experiments=experiments,
    n_workers=4,   # 实验级并行（4核）
    feature_config='transient_tau_parallel'
)

# 在 YAML 配置中：
extractors:
  - type: 'transient.tau_on_off'
    params:
      use_parallel: true  # 启用窗口搜索并行
      max_workers: 4      # 每实验 4核窗口搜索
# 总核心使用: 4 实验 × 4 核/实验 = 16 核
```

## 模块结构

### 核心模块（v0.3.0）

- **TauFitter**: L0 - 基础拟合类，用于拟合指定窗口内的tau值
- **AutoTauFitter**: L1 - 自动窗口搜索（支持可选并行）
- **CyclesAutoTauFitter**: L2 - 多周期处理（支持工厂模式）
- **CachedAutoTauFitter**: L2 - 窗口缓存策略（98% 命中率，50x 加速）⭐
- **SmartWindowSearchFitter**: L1 - 智能搜索（differential_evolution，6.5x 加速）⭐
- **accelerated** 模块: Numba JIT 编译函数（2-5x 加速）⭐

### 废弃模块（仍可用但不推荐）

- ⚠️ **ParallelAutoTauFitter**: 已废弃，请改用 `AutoTauFitter(..., executor=...)`
- ⚠️ **ParallelCyclesAutoTauFitter**: 已废弃，请改用 `CyclesAutoTauFitter(..., fitter_factory=...)`

### 集成模块

- **autotau_extractors.py**: features_v2 集成层（OECT 数据处理流程）

## 依赖

**必需**：
- NumPy
- SciPy
- Matplotlib
- pandas
- tqdm

**可选（性能加速）**：
- **numba**: JIT 编译加速（2-5x），强烈推荐
  ```bash
  conda install numba
  ```

## 文档

详细文档请参见[API参考文档](https://github.com/Durian-leader/autotau/blob/main/docs/api_reference.md)。

## 贡献代码

欢迎提交Pull Request或创建Issue。

## 协议

[MIT](LICENSE)
