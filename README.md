# AutoTau - 自动化时间常数τ拟合工具

AutoTau是一个用于自动拟合信号中指数上升/下降过程时间常数τ的Python库，支持并行处理以加速计算。

## 功能特点

- 自动寻找最佳拟合窗口，无需手动指定拟合区间
- 支持单周期和多周期信号的拟合
- 内置指数上升和下降模型: y = A(1-e^(-t/τ)) + C 和 y = Ae^(-t/τ) + C
- 提供R²和调整后R²等拟合质量指标
- 自动重新拟合质量不佳的结果
- 多种可视化方法展示拟合结果
- 支持并行处理，充分利用多核CPU提升性能

## 安装

```bash
pip install -e .
```

## 快速开始

### 基本示例

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

可以使用examples.py中的compare_performance()函数来比较串行和并行处理的性能差异:

```python
from autotau.examples import compare_performance

compare_performance()
```

在多核CPU上，并行处理通常可以获得2-8倍的性能提升，具体取决于CPU核心数和任务特性。

## 模块结构

- **TauFitter**: 基础拟合类，用于拟合指定窗口内的tau值
- **AutoTauFitter**: 自动寻找最佳拟合窗口的拟合器
- **CyclesAutoTauFitter**: 处理多周期数据的拟合器
- **ParallelAutoTauFitter**: AutoTauFitter的并行版本
- **ParallelCyclesAutoTauFitter**: CyclesAutoTauFitter的并行版本

## 依赖

- NumPy
- SciPy
- Matplotlib
- pandas
- tqdm

## 协议

MIT
