# AutoTau

[English](README.md) | 中文

AutoTau 是一个用于拟合指数上升/衰减过程中时间常数 (τ) 的 Python 工具，专为瞬态和周期信号分析设计。

**模型：**
- 开启（上升）：`y = A * (1 - exp(-t/τ)) + C`
- 关闭（衰减）：`y = A * exp(-t/τ) + C`

## 功能特性

- 单窗口 τ 拟合，提供 R² 和调整 R² 指标
- 自动滑动窗口搜索最优拟合区域
- 多周期批量拟合（串行和并行）
- 解耦工作流：窗口搜索与拟合分离
- PyQt5 图形界面，支持交互式分析
- 双语支持（中文/英文）

## 安装

```bash
pip install autotau
```

安装 GUI 支持：

```bash
pip install autotau[gui]
```

本地开发安装：

```bash
pip install -e .
# 或带 GUI 支持
pip install -e ".[gui]"
```

**Python 版本：** >=3.6

## 快速入门

### 1) 单窗口拟合 (TauFitter)

```python
import numpy as np
from autotau import TauFitter

time = np.linspace(0, 0.4, 400)
signal = np.where(
    time <= 0.2,
    1 - np.exp(-time / 0.05),
    np.exp(-(time - 0.2) / 0.05),
)

fitter = TauFitter(
    time,
    signal,
    t_on_idx=[0.0, 0.2],
    t_off_idx=[0.2, 0.4],
    normalize=False,
    language="cn",
)

fitter.fit_tau_on(interp=True, points_after_interp=200)
fitter.fit_tau_off(interp=True, points_after_interp=200)

print("tau_on:", fitter.get_tau_on())
print("tau_off:", fitter.get_tau_off())
print("R2_on:", fitter.tau_on_r_squared, "R2_adj_on:", fitter.tau_on_r_squared_adj)
```

### 2) 自动窗口搜索 (AutoTauFitter)

```python
from autotau import AutoTauFitter

auto_fitter = AutoTauFitter(
    time,
    signal,
    sample_step=time[1] - time[0],
    period=0.4,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=1,
    normalize=False,
    language="cn",
    show_progress=True,
)

auto_fitter.fit_tau_on_and_off()

print("最佳 tau_on:", auto_fitter.best_tau_on_fitter.get_tau_on())
print("最佳 tau_off:", auto_fitter.best_tau_off_fitter.get_tau_off())
print("开启窗口:", auto_fitter.best_tau_on_window_start_time, auto_fitter.best_tau_on_window_end_time)
```

### 3) 多周期自动拟合 (CyclesAutoTauFitter)

```python
from autotau import CyclesAutoTauFitter

period = 0.4
sample_rate = 1000

cycles_fitter = CyclesAutoTauFitter(
    time,
    signal,
    period=period,
    sample_rate=sample_rate,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=1,
    normalize=False,
    language="cn",
)

results = cycles_fitter.fit_all_cycles(r_squared_threshold=0.95)
summary = cycles_fitter.get_summary_data()
print(summary.head())
```

### 4) 解耦工作流 (WindowFinder + ParallelCyclesTauFitter)

为了最大灵活性，将窗口搜索与拟合分离：

```python
from autotau import WindowFinder, ParallelCyclesTauFitter

# 步骤 1：使用前 2 个周期搜索最佳窗口
two_period_mask = time <= time[0] + 2 * period
time_subset = time[two_period_mask]
signal_subset = signal[two_period_mask]

finder = WindowFinder(
    time_subset,
    signal_subset,
    sample_step=1 / sample_rate,
    period=period,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    show_progress=True,
    max_workers=4,
)
windows = finder.find_best_windows()

print("开启窗口偏移:", windows['on']['offset'])
print("开启窗口大小:", windows['on']['size'])
print("开启窗口 R²:", windows['on']['r_squared'])

# 步骤 2：将窗口应用到所有周期（并行）
fitter = ParallelCyclesTauFitter(
    time,
    signal,
    period=period,
    sample_rate=sample_rate,
    window_on_offset=windows["on"]["offset"],
    window_on_size=windows["on"]["size"],
    window_off_offset=windows["off"]["offset"],
    window_off_size=windows["off"]["size"],
    show_progress=True,
    max_workers=4,
)

df = fitter.fit_all_cycles()  # 默认返回 DataFrame
summary = fitter.get_summary_data()
fitter.plot_cycle_results()
```

## API 概览

| 类名 | 用途 | 使用场景 |
|------|------|----------|
| `TauFitter` | 单窗口 τ 拟合 | 已知确切拟合窗口 |
| `AutoTauFitter` | 自动窗口搜索 + 拟合 | 单周期，自动查找最佳窗口 |
| `WindowFinder` | 仅窗口搜索（不拟合） | 在多个数据集间复用窗口 |
| `CyclesTauFitter` | 多周期，手动窗口（串行） | 少量周期 (<10)，已知窗口 |
| `ParallelCyclesTauFitter` | 多周期，手动窗口（并行） | 大量周期 (>=10)，已知窗口 |
| `CyclesAutoTauFitter` | 多周期，自动窗口 | 大量周期，未知窗口 |
| `ParallelAutoTauFitter` | *（已弃用）* | 使用 `AutoTauFitter(..., executor=...)` |
| `ParallelCyclesAutoTauFitter` | *（已弃用）* | 使用 `CyclesAutoTauFitter(..., fitter_factory=...)` |

## 并行处理

### AutoTauFitter 使用 Executor

```python
from concurrent.futures import ProcessPoolExecutor
from autotau import AutoTauFitter

with ProcessPoolExecutor(max_workers=8) as executor:
    fitter = AutoTauFitter(time, signal, sample_step=..., period=..., executor=executor)
    fitter.fit_tau_on_and_off()
```

### CyclesAutoTauFitter 使用自定义工厂

```python
from autotau import CyclesAutoTauFitter

def custom_fitter_factory(time, signal, **kwargs):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        return AutoTauFitter(time, signal, executor=executor, **kwargs)

cycles_fitter = CyclesAutoTauFitter(
    time, signal, period=0.4, sample_rate=1000,
    fitter_factory=custom_fitter_factory
)
```

### 性能建议

| 场景 | 推荐类 |
|------|--------|
| 周期数 < 10 | `CyclesTauFitter`（避免多进程开销） |
| 周期数 >= 10 | `ParallelCyclesTauFitter`（显著加速） |
| 窗口未知 | `CyclesAutoTauFitter` 或 `WindowFinder` + `ParallelCyclesTauFitter` |

**注意：** 在 Windows 系统上，请将并行代码包装在 `if __name__ == "__main__":` 块中。

## 高级工作流

### 探索不同窗口参数

先找到最优窗口，然后尝试不同变体：

```python
from autotau import WindowFinder, ParallelCyclesTauFitter

# 查找初始最佳窗口
finder = WindowFinder(time_subset, signal_subset, sample_step=0.001, period=period)
windows = finder.find_best_windows()

# 使用原始窗口拟合
fitter1 = ParallelCyclesTauFitter(
    time, signal, period, sample_rate,
    window_on_offset=windows['on']['offset'],
    window_on_size=windows['on']['size'],
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size']
)
results1 = fitter1.fit_all_cycles()
summary1 = fitter1.get_summary_data()

# 尝试调整后的参数
fitter2 = ParallelCyclesTauFitter(
    time, signal, period, sample_rate,
    window_on_offset=windows['on']['offset'] + 0.1,  # 偏移 +0.1s
    window_on_size=windows['on']['size'] * 1.1,      # 扩大 10%
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size']
)
results2 = fitter2.fit_all_cycles()
summary2 = fitter2.get_summary_data()

# 比较结果
print("原始平均 R²:", summary1['r_squared_on'].mean())
print("调整后平均 R²:", summary2['r_squared_on'].mean())
```

## GUI 使用

v0.6.0 包含一个 PyQt5 图形界面，采用解耦架构（避免多进程/matplotlib 冲突）。

### 启动

```bash
# 模块方式启动
python -m autotau.gui

# 命令行方式（安装后）
autotau-gui
```

### 工作流程

1. 从 Excel 复制两列数据（时间 + 信号）
2. 点击"解析剪贴板"
3. 设置周期和采样率
4. 选择模式（自动/手动）
5. 点击"拟合数据"
6. 查看结果，导出到 CSV/Excel

### GUI 功能

- 自动模式：自动窗口搜索
- 手动模式：指定窗口参数
- 可视化：原始数据、τ 演化、R² 质量
- 导出：CSV/Excel 输出

## API 参数

### WindowFinder

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `time` | 时间数组 | 必需 |
| `signal` | 信号数组 | 必需 |
| `sample_step` | 采样步长（秒） | 必需 |
| `period` | 信号周期（秒） | 必需 |
| `window_scalar_min` | 最小窗口大小比例 | 1/5 |
| `window_scalar_max` | 最大窗口大小比例 | 1/3 |
| `show_progress` | 显示进度条 | False |
| `max_workers` | 最大工作进程数 | CPU 核心数 |

**方法：**
- `find_best_windows()` - 返回包含 'on' 和 'off' 窗口参数的字典
- `get_window_params()` - 获取窗口参数

### CyclesTauFitter / ParallelCyclesTauFitter

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `time` | 时间数组 | 必需 |
| `signal` | 信号数组 | 必需 |
| `period` | 信号周期（秒） | 必需 |
| `sample_rate` | 采样率（Hz） | 必需 |
| `window_on_offset` | 开启窗口偏移（秒） | 必需 |
| `window_on_size` | 开启窗口大小（秒） | 必需 |
| `window_off_offset` | 关闭窗口偏移（秒） | 必需 |
| `window_off_size` | 关闭窗口大小（秒） | 必需 |
| `normalize` | 归一化信号 | False |
| `language` | 'cn' 或 'en' | 'en' |
| `show_progress` | 显示进度（仅并行版） | False |
| `max_workers` | 工作进程数（仅并行版） | CPU 核心数 |

**方法：**
- `fit_all_cycles()` - 返回 DataFrame（默认）或记录列表
- `get_summary_data()` - 获取汇总统计数据
- `plot_cycle_results()` - 绘制 τ 演化图
- `plot_r_squared_values()` - 绘制 R² 质量图
- `plot_windows_on_signal()` - 在信号上可视化窗口
- `plot_all_fits()` - 绘制所有拟合曲线

## 注意事项

1. **WindowFinder** 应使用数据的前 1-2 个周期进行窗口搜索
2. 少量周期时，并行处理可能无法提升性能
3. 窗口偏移相对于每个周期的起始时间
4. 确保窗口参数不超过周期边界
5. `ParallelCyclesTauFitter` 使用分块并行策略（每个进程处理一组周期）以提高效率

## 示例与数据

- 示例脚本：`examples/basic_examples.py`
- 示例数据：`autotau/transient.csv`

## 更新日志

### v0.6.0

- **GUI 完全解耦**：GUI 现在使用独立的拟合模块，不再依赖核心实现
- **修复 GUI 卡顿**：使用正确的 QThread 信号机制进行后台拟合
- 新增 `autotau.gui.fitters` 模块：`ExponentialFitter`、`WindowSearcher`、`CyclesFitter`、`AutoCyclesFitter`
- 避免了多进程和 matplotlib 与 GUI 的冲突

### v0.5.0

- 添加 PyQt5 图形界面
- 支持剪贴板粘贴、自动/手动拟合、可视化、导出

### v0.4.5

- 优化分块并行策略
- 修复窗口偏移计算

## 许可证

MIT 许可证，详见 `LICENSE`。
