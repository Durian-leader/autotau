# AutoTau API 参考文档

## 核心类

### TauFitter

基础拟合类，用于拟合指定窗口内的tau值。

```python
from autotau import TauFitter

fitter = TauFitter(time_data, signal_data, t_on_idx=None, t_off_idx=None)
```

**参数**

- `time_data` (numpy.ndarray): 时间数据
- `signal_data` (numpy.ndarray): 信号数据
- `t_on_idx` (list, optional): 开启过程的时间窗口 [start, end]
- `t_off_idx` (list, optional): 关闭过程的时间窗口 [start, end]

**方法**

- `set_t_on_idx(t_on_idx)`: 设置开启过程拟合窗口
- `set_t_off_idx(t_off_idx)`: 设置关闭过程拟合窗口
- `fit_tau_on()`: 拟合开启过程时间常数
- `fit_tau_off()`: 拟合关闭过程时间常数
- `get_tau_on()`: 获取开启过程时间常数
- `get_tau_off()`: 获取关闭过程时间常数
- `get_r_squared_on()`: 获取开启过程R²值
- `get_r_squared_off()`: 获取关闭过程R²值
- `get_adj_r_squared_on()`: 获取开启过程调整后R²值
- `get_adj_r_squared_off()`: 获取关闭过程调整后R²值
- `plot_tau_on()`: 可视化开启过程拟合结果
- `plot_tau_off()`: 可视化关闭过程拟合结果

### AutoTauFitter

自动寻找最佳拟合窗口的拟合器。

```python
from autotau import AutoTauFitter

auto_fitter = AutoTauFitter(
    time_data, 
    signal_data,
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
```

**参数**

- `time_data` (numpy.ndarray): 时间数据
- `signal_data` (numpy.ndarray): 信号数据
- `sample_step` (float): 采样步长
- `period` (float): 信号周期
- `window_scalar_min` (float): 最小窗口比例
- `window_scalar_max` (float): 最大窗口比例
- `window_points_step` (int): 窗口点数步长
- `window_start_idx_step` (int): 窗口起始索引步长
- `normalize` (bool): 是否归一化信号
- `language` (str): 语言设置，'cn'或'en'
- `show_progress` (bool): 是否显示进度条

**方法**

- `fit_tau_on()`: 自动拟合开启过程时间常数
- `fit_tau_off()`: 自动拟合关闭过程时间常数
- `fit_tau_on_and_off()`: 自动拟合开启和关闭过程时间常数

### CyclesAutoTauFitter

处理多周期数据的拟合器。

```python
from autotau import CyclesAutoTauFitter

cycles_fitter = CyclesAutoTauFitter(
    time_data,
    signal_data,
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
```

**方法**

- `fit_all_cycles()`: 拟合所有周期
- `plot_cycle_results()`: 可视化所有周期的拟合结果
- `plot_windows_on_signal(num_cycles=5)`: 在原始信号上可视化拟合窗口
- `plot_all_fits(num_cycles=3)`: 可视化所有拟合
- `get_summary_data()`: 获取结果摘要

### ParallelAutoTauFitter

AutoTauFitter的并行版本。

```python
from autotau import ParallelAutoTauFitter

parallel_auto_fitter = ParallelAutoTauFitter(
    time_data, 
    signal_data,
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
```

### ParallelCyclesAutoTauFitter

CyclesAutoTauFitter的并行版本。

```python
from autotau import ParallelCyclesAutoTauFitter

parallel_cycles_fitter = ParallelCyclesAutoTauFitter(
    time_data,
    signal_data,
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
``` 