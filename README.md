# AutoTau

AutoTau 是一个用于拟合指数上升/下降过程时间常数 τ 的 Python 工具，面向瞬态或周期信号的 τ 提取与分析。

模型形式：
- 上升（turn-on）：`y = A * (1 - exp(-t/τ)) + C`
- 下降（turn-off）：`y = A * exp(-t/τ) + C`

## 主要功能

- 指定窗口的 τ 拟合（`TauFitter`），输出 τ、R²、调整后 R²，并支持绘图。
- 滑动窗口自动搜索最佳拟合区间（`AutoTauFitter`），可选并行执行。
- 多周期自动拟合（`CyclesAutoTauFitter`），自动从前两个周期推断窗口并应用到全部周期。
- 窗口搜索与拟合解耦：`WindowFinder` 只找窗口；`CyclesTauFitter`/`ParallelCyclesTauFitter` 使用手动窗口拟合。
- 多进程并行周期拟合（`ParallelCyclesTauFitter`），默认返回 `pandas.DataFrame`。
- **PyQt5 图形界面**（v0.5.0 新增）：支持从 Excel 粘贴数据、自动/手动拟合、可视化结果、导出 CSV/Excel。

## 安装

```bash
pip install autotau
```

安装 GUI 版本（包含 PyQt5）：

```bash
pip install autotau[gui]
```

本地开发安装：

```bash
pip install -e .
# 或带 GUI
pip install -e ".[gui]"
```

Python 版本要求：`>=3.6`。

## 快速开始

### 1) 单窗口拟合（TauFitter）

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
    language="en",
)

fitter.fit_tau_on(interp=True, points_after_interp=200)
fitter.fit_tau_off(interp=True, points_after_interp=200)

print("tau_on:", fitter.get_tau_on())
print("tau_off:", fitter.get_tau_off())
print("R2_on:", fitter.tau_on_r_squared, "R2_adj_on:", fitter.tau_on_r_squared_adj)
```

### 2) 自动窗口搜索（AutoTauFitter）

假设已有 `time`、`signal`（同采样、按时间升序），且包含一个完整周期。

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
    language="en",
    show_progress=True,
)

auto_fitter.fit_tau_on_and_off()

print("best tau_on:", auto_fitter.best_tau_on_fitter.get_tau_on())
print("best tau_off:", auto_fitter.best_tau_off_fitter.get_tau_off())
print("on window:", auto_fitter.best_tau_on_window_start_time, auto_fitter.best_tau_on_window_end_time)
```

### 3) 多周期自动拟合（CyclesAutoTauFitter）

假设 `time`、`signal` 为多周期数据，`period` 为单周期长度。

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
    language="en",
)

results = cycles_fitter.fit_all_cycles(r_squared_threshold=0.95)
summary = cycles_fitter.get_summary_data()
print(summary.head())
```

### 4) 手动窗口 + 并行多周期拟合

先在前两个周期搜索窗口，再并行拟合全部周期。

```python
from autotau import WindowFinder, ParallelCyclesTauFitter

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
```

## 并行使用建议

- `AutoTauFitter` 可通过注入 `executor` 并行化窗口搜索：
  ```python
  from concurrent.futures import ProcessPoolExecutor
  from autotau import AutoTauFitter

  with ProcessPoolExecutor(max_workers=8) as executor:
      fitter = AutoTauFitter(time, signal, sample_step=..., period=..., executor=executor)
      fitter.fit_tau_on_and_off()
  ```
- `CyclesAutoTauFitter` 可通过 `fitter_factory` 传入自定义 `AutoTauFitter`（例如带并行的版本）。
- `ParallelCyclesTauFitter` 已内置多进程“分块并行”，适合周期数较多的场景。
- Windows 下使用多进程时，请将并行入口放在 `if __name__ == "__main__":` 中。

## API 概览

- `TauFitter`：指定窗口拟合 τ，输出拟合参数与 R²。
- `AutoTauFitter`：滑动窗口搜索最佳拟合区间（可选并行）。
- `CyclesAutoTauFitter`：多周期自动拟合，支持低 R² 周期的再拟合。
- `WindowFinder`：只搜索窗口，不做拟合。
- `CyclesTauFitter`：使用手动窗口拟合多个周期（串行）。
- `ParallelCyclesTauFitter`：使用手动窗口拟合多个周期（并行，默认返回 DataFrame）。
- `ParallelAutoTauFitter` / `ParallelCyclesAutoTauFitter`：保留兼容但已废弃，建议改用 `AutoTauFitter(..., executor=...)` 与 `CyclesAutoTauFitter(..., fitter_factory=...)`。

## 图形界面（GUI）

v0.5.0 新增 PyQt5 图形界面，支持：

- 从 Excel 复制粘贴两列数据（时间、信号）
- 自动模式：自动搜索最佳拟合窗口
- 手动模式：手动指定窗口参数
- 可视化：原始数据、τ 演化图、R² 质量图
- 导出结果到 CSV/Excel

### 启动 GUI

```bash
# 方式 1：模块启动
python -m autotau.gui

# 方式 2：命令行（安装后）
autotau-gui
```

### 使用流程

1. 从 Excel 复制两列数据（时间 + 信号）
2. 点击"Parse Clipboard"解析数据
3. 设置周期（Period）和采样率（Sample Rate）
4. 选择拟合模式（Auto/Manual）
5. 点击"Fit Data"开始拟合
6. 查看结果图表，导出 CSV/Excel

## 示例与数据

- 示例脚本：`examples/basic_examples.py`
- 示例数据：`autotau/transient.csv`

## 更新日志

### v0.5.0

- 新增 PyQt5 图形界面（GUI）
- 支持从 Excel 粘贴数据、自动/手动拟合、可视化、导出

### v0.4.5

- 优化分块并行策略
- 修复窗口偏移量计算问题

## 许可证

MIT License，见 `LICENSE`。
