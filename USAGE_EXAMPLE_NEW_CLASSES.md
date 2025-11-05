# 新类使用指南 / New Classes Usage Guide

本文档介绍 autotau 包中新增的三个类的使用方法，它们可以让你更灵活地控制窗口搜索和拟合过程。

This document introduces the usage of three new classes in the autotau package, which provide more flexible control over window search and fitting processes.

---

## 新增的类 / New Classes

### 1. WindowFinder - 窗口查找器
**用途 / Purpose**: 只搜索最佳窗口，不进行拟合
**Use Case**: Find best windows without performing fitting

### 2. CyclesTauFitter - 手动窗口周期拟合器（非并行）
**用途 / Purpose**: 使用手动指定的窗口参数对多个周期进行拟合（顺序处理）
**Use Case**: Fit multiple cycles with manually specified window parameters (sequential processing)

### 3. ParallelCyclesTauFitter - 手动窗口周期拟合器（并行）
**用途 / Purpose**: 使用手动指定的窗口参数对多个周期进行拟合（并行处理）
**Use Case**: Fit multiple cycles with manually specified window parameters (parallel processing)

---

## 使用场景 / Use Cases

### 场景 1: 分步骤工作流（解耦窗口搜索和拟合）
### Scenario 1: Step-by-step workflow (Decouple window search and fitting)

这是最灵活的工作流程，允许你先找到最佳窗口，然后用不同的参数重复拟合。

This is the most flexible workflow, allowing you to find the best windows first, then repeatedly fit with different parameters.

```python
import numpy as np
from autotau import WindowFinder, ParallelCyclesTauFitter

# 1. 准备数据 / Prepare data
time = np.array([...])  # 你的时间数据 / your time data
signal = np.array([...])  # 你的信号数据 / your signal data
period = 10.0  # 周期 / period in seconds
sample_rate = 100  # 采样率 / sampling rate in Hz

# 2. 第一步：搜索最佳窗口（使用前两个周期的数据）
#    Step 1: Find best windows (using first two cycles)
two_period_mask = (time <= time[0] + 2 * period)
time_subset = time[two_period_mask]
signal_subset = signal[two_period_mask]

finder = WindowFinder(
    time_subset,
    signal_subset,
    sample_step=1/sample_rate,
    period=period,
    window_scalar_min=1/5,
    window_scalar_max=1/3,
    show_progress=True,
    max_workers=4  # 使用 4 个进程并行搜索
)

# 搜索最佳窗口 / Search for best windows
windows = finder.find_best_windows()

# 查看找到的窗口参数 / View found window parameters
print("On window offset:", windows['on']['offset'])
print("On window size:", windows['on']['size'])
print("On window R²:", windows['on']['r_squared'])
print("Off window offset:", windows['off']['offset'])
print("Off window size:", windows['off']['size'])
print("Off window R²:", windows['off']['r_squared'])

# 3. 第二步：使用找到的窗口参数拟合所有周期
#    Step 2: Fit all cycles using found window parameters
fitter = ParallelCyclesTauFitter(
    time,
    signal,
    period=period,
    sample_rate=sample_rate,
    window_on_offset=windows['on']['offset'],
    window_on_size=windows['on']['size'],
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size'],
    show_progress=True,
    max_workers=4,
    language='en'  # or 'cn' for Chinese
)

# 拟合所有周期 / Fit all cycles
results = fitter.fit_all_cycles()

# 4. 查看和分析结果 / View and analyze results
summary = fitter.get_summary_data()
print(summary)

# 绘制结果 / Plot results
fitter.plot_cycle_results(dual_y_axis=True)
fitter.plot_r_squared_values(threshold=0.95)
fitter.plot_windows_on_signal(num_cycles=5)
```

---

### 场景 2: 已知窗口参数，直接拟合
### Scenario 2: Known window parameters, direct fitting

如果你已经知道最佳的窗口参数（例如从之前的实验中获得），可以直接使用 CyclesTauFitter 或 ParallelCyclesTauFitter。

If you already know the optimal window parameters (e.g., from previous experiments), you can directly use CyclesTauFitter or ParallelCyclesTauFitter.

```python
from autotau import ParallelCyclesTauFitter

# 已知的窗口参数 / Known window parameters
window_on_offset = 1.5  # seconds
window_on_size = 2.0    # seconds
window_off_offset = 6.0 # seconds
window_off_size = 2.5   # seconds

# 直接创建拟合器并拟合 / Create fitter and fit directly
fitter = ParallelCyclesTauFitter(
    time,
    signal,
    period=10.0,
    sample_rate=100,
    window_on_offset=window_on_offset,
    window_on_size=window_on_size,
    window_off_offset=window_off_offset,
    window_off_size=window_off_size,
    show_progress=True,
    max_workers=4
)

results = fitter.fit_all_cycles()
fitter.plot_cycle_results()
```

---

### 场景 3: 探索不同的窗口参数
### Scenario 3: Explore different window parameters

你可以使用 WindowFinder 找到最佳窗口，然后尝试微调窗口参数，重新拟合并比较结果。

You can use WindowFinder to find the best windows, then try fine-tuning the window parameters, refit and compare results.

```python
from autotau import WindowFinder, ParallelCyclesTauFitter

# 1. 找到初始最佳窗口 / Find initial best windows
finder = WindowFinder(...)
windows = finder.find_best_windows()

# 2. 使用原始窗口拟合 / Fit with original windows
fitter1 = ParallelCyclesTauFitter(
    time, signal, period, sample_rate,
    window_on_offset=windows['on']['offset'],
    window_on_size=windows['on']['size'],
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size']
)
results1 = fitter1.fit_all_cycles()
summary1 = fitter1.get_summary_data()

# 3. 微调窗口参数后重新拟合 / Refit with adjusted parameters
fitter2 = ParallelCyclesTauFitter(
    time, signal, period, sample_rate,
    window_on_offset=windows['on']['offset'] + 0.1,  # 调整 +0.1s
    window_on_size=windows['on']['size'] * 1.1,      # 增大 10%
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size']
)
results2 = fitter2.fit_all_cycles()
summary2 = fitter2.get_summary_data()

# 4. 比较结果 / Compare results
print("Original average R²:", summary1['r_squared_on'].mean())
print("Adjusted average R²:", summary2['r_squared_on'].mean())
```

---

## 性能对比 / Performance Comparison

### CyclesTauFitter vs ParallelCyclesTauFitter

- **CyclesTauFitter**: 顺序处理，适合周期数较少的情况
  Sequential processing, suitable for fewer cycles

- **ParallelCyclesTauFitter**: 并行处理，适合周期数较多的情况
  Parallel processing, suitable for many cycles

**建议 / Recommendation**:
- 周期数 < 10: 使用 CyclesTauFitter (避免多进程开销)
  Cycles < 10: Use CyclesTauFitter (avoid multiprocessing overhead)

- 周期数 >= 10: 使用 ParallelCyclesTauFitter (显著加速)
  Cycles >= 10: Use ParallelCyclesTauFitter (significant speedup)

---

## 完整工作流程示例 / Complete Workflow Example

```python
import numpy as np
from autotau import WindowFinder, ParallelCyclesTauFitter

# === 数据准备 / Data Preparation ===
time = np.linspace(0, 100, 10000)  # 100秒，10000个点
signal = np.sin(2 * np.pi * time / 10) + np.random.normal(0, 0.1, len(time))
period = 10.0
sample_rate = 100

# === 步骤 1: 搜索窗口 / Step 1: Search Windows ===
print("Step 1: Searching for best windows...")
two_period_mask = (time <= time[0] + 2 * period)
finder = WindowFinder(
    time[two_period_mask],
    signal[two_period_mask],
    sample_step=1/sample_rate,
    period=period,
    show_progress=True,
    max_workers=4
)
windows = finder.find_best_windows()

# === 步骤 2: 拟合所有周期 / Step 2: Fit All Cycles ===
print("\nStep 2: Fitting all cycles...")
fitter = ParallelCyclesTauFitter(
    time,
    signal,
    period=period,
    sample_rate=sample_rate,
    window_on_offset=windows['on']['offset'],
    window_on_size=windows['on']['size'],
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size'],
    show_progress=True,
    max_workers=4
)
results = fitter.fit_all_cycles()

# === 步骤 3: 分析结果 / Step 3: Analyze Results ===
print("\nStep 3: Analyzing results...")
summary = fitter.get_summary_data()
print(summary.describe())

# === 步骤 4: 可视化 / Step 4: Visualization ===
print("\nStep 4: Generating plots...")
fitter.plot_cycle_results()
fitter.plot_r_squared_values(threshold=0.95)
fitter.plot_all_fits(start_cycle=0, num_cycles=3)

print("\nAnalysis complete!")
```

---

## API 参考 / API Reference

### WindowFinder

**初始化参数 / Initialization Parameters**:
- `time`: 时间数组 / Time array
- `signal`: 信号数组 / Signal array
- `sample_step`: 采样步长（秒）/ Sampling step (seconds)
- `period`: 信号周期（秒）/ Signal period (seconds)
- `window_scalar_min`: 最小窗口大小比例 / Minimum window size ratio (default: 1/5)
- `window_scalar_max`: 最大窗口大小比例 / Maximum window size ratio (default: 1/3)
- `show_progress`: 是否显示进度 / Show progress (default: False)
- `max_workers`: 最大工作进程数 / Max worker processes (default: CPU count)

**方法 / Methods**:
- `find_best_windows()`: 搜索最佳窗口 / Search for best windows
- `get_window_params()`: 获取窗口参数 / Get window parameters

---

### CyclesTauFitter / ParallelCyclesTauFitter

**初始化参数 / Initialization Parameters**:
- `time`: 时间数组 / Time array
- `signal`: 信号数组 / Signal array
- `period`: 信号周期（秒）/ Signal period (seconds)
- `sample_rate`: 采样率（Hz）/ Sampling rate (Hz)
- `window_on_offset`: 开启窗口偏移量（秒）/ On window offset (seconds)
- `window_on_size`: 开启窗口大小（秒）/ On window size (seconds)
- `window_off_offset`: 关闭窗口偏移量（秒）/ Off window offset (seconds)
- `window_off_size`: 关闭窗口大小（秒）/ Off window size (seconds)
- `normalize`: 是否归一化 / Normalize signal (default: False)
- `language`: 语言 'cn' 或 'en' / Language 'cn' or 'en' (default: 'en')
- `show_progress`: 是否显示进度 / Show progress (default: False, ParallelCyclesTauFitter only)
- `max_workers`: 最大工作进程数 / Max worker processes (ParallelCyclesTauFitter only)

**方法 / Methods**:
- `fit_all_cycles()`: 拟合所有周期 / Fit all cycles
- `get_summary_data()`: 获取摘要数据 / Get summary data
- `plot_cycle_results()`: 绘制周期结果 / Plot cycle results
- `plot_r_squared_values()`: 绘制R²值 / Plot R² values
- `plot_windows_on_signal()`: 在信号上绘制窗口 / Plot windows on signal
- `plot_all_fits()`: 绘制所有拟合结果 / Plot all fits

---

## 优势 / Advantages

1. **解耦设计 / Decoupled Design**: 窗口搜索和拟合分离，更灵活
   Window search and fitting are separated for more flexibility

2. **性能优化 / Performance Optimization**: 并行处理大幅提升速度
   Parallel processing significantly improves speed

3. **可重复使用 / Reusable**: 找到窗口后可以用不同参数重复拟合
   Once windows are found, you can refit with different parameters

4. **易于调试 / Easy Debugging**: 分步骤执行，便于检查中间结果
   Step-by-step execution makes it easy to inspect intermediate results

---

## 注意事项 / Notes

1. `WindowFinder` 建议使用前1-2个周期的数据来搜索窗口
   Use first 1-2 cycles of data with `WindowFinder` to search for windows

2. 并行处理在周期数较少时可能不会带来性能提升
   Parallel processing may not improve performance with few cycles

3. 窗口偏移量是相对于每个周期起始时间的
   Window offset is relative to the start time of each cycle

4. 确保窗口参数不会超出周期范围
   Ensure window parameters do not exceed cycle boundaries

5. ParallelCyclesTauFitter 采用分块并行（每个进程处理一组周期），避免“每周期一个任务”的高开销；通常只需设置 `max_workers` 即可获得良好性能，无需额外并行参数。
   ParallelCyclesTauFitter uses chunked parallelism (each process handles a group of cycles), avoiding the high overhead of one-task-per-cycle; in most cases setting `max_workers` is sufficient, no extra tuning needed.
