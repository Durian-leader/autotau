# AutoTau API 参考文档

AutoTau 接收 `time` 与 `signal` 数组作为输入。常见数据格式为 CSV，列名为 `Time` 与 `Id`，通常使用 `signal = -Id` 作为拟合信号。

所有拟合函数默认假设：
- `time` 单调递增
- `time` 与 `signal` 长度一致
- `t_on_idx` / `t_off_idx` 为 `[start_time, end_time]`（单位与 `time` 一致）

---

## TauFitter

指定窗口内的指数上升/下降拟合。

```python
from autotau import TauFitter

fitter = TauFitter(
    time,
    signal,
    t_on_idx=[0.2, 0.21],
    t_off_idx=[0.24, 0.25],
    normalize=False,
    language="en",
)
```

**常用方法**
- `fit_tau_on(interp=True, points_after_interp=100)`
- `fit_tau_off(interp=True, points_after_interp=100)`
- `get_tau_on()`
- `get_tau_off()`
- `plot_tau_on()`
- `plot_tau_off()`
- `set_language("cn" | "en")`

**常用属性**
- `tau_on_popt`, `tau_off_popt`
- `tau_on_pcov`, `tau_off_pcov`
- `tau_on_r_squared`, `tau_off_r_squared`
- `tau_on_r_squared_adj`, `tau_off_r_squared_adj`

---

## AutoTauFitter

滑动窗口搜索最佳拟合区间（可选并行执行器）。

```python
from autotau import AutoTauFitter

auto_fitter = AutoTauFitter(
    time,
    signal,
    sample_step=0.001,
    period=0.2,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    window_points_step=10,
    window_start_idx_step=1,
    normalize=False,
    language="en",
    show_progress=True,
    executor=None,
)
```

**常用方法**
- `fit_tau_on_and_off(interp=True, points_after_interp=100)`
  - 返回 `(tau_on_popt, tau_on_r_squared_adj, tau_off_popt, tau_off_r_squared_adj)`

**常用属性**
- `best_tau_on_fitter`, `best_tau_off_fitter`
- `best_tau_on_window_start_time`, `best_tau_on_window_end_time`, `best_tau_on_window_size`
- `best_tau_off_window_start_time`, `best_tau_off_window_end_time`, `best_tau_off_window_size`

---

## WindowFinder

只搜索最佳窗口，不返回拟合器。

```python
from autotau import WindowFinder

finder = WindowFinder(
    time_subset,
    signal_subset,
    sample_step=0.001,
    period=0.2,
    show_progress=True,
)
windows = finder.find_best_windows()
```

**返回结构**
```text
{
  "on": {
    "offset": ...,
    "size": ...,
    "start_time": ...,
    "end_time": ...,
    "r_squared": ...,
    "r_squared_adj": ...
  },
  "off": { ... }
}
```

**常用方法**
- `find_best_windows(interp=True, points_after_interp=100)`
- `get_window_params()`

---

## CyclesAutoTauFitter

多周期自动拟合：先用前两个周期确定窗口，再应用到所有周期。支持低 R² 周期自动再拟合。

```python
from autotau import CyclesAutoTauFitter

cycles_fitter = CyclesAutoTauFitter(
    time,
    signal,
    period=0.2,
    sample_rate=1000,
    window_scalar_min=0.2,
    window_scalar_max=1/3,
    normalize=False,
    language="en",
)
```

**常用方法**
- `find_best_windows(interp=True, points_after_interp=100)`
- `fit_all_cycles(interp=True, points_after_interp=100, r_squared_threshold=0.95)`
- `get_summary_data()`
- `get_refitted_cycles_info()`
- `plot_cycle_results()`
- `plot_all_fits()`
- `plot_r_squared_values()`
- `plot_windows_on_signal()`

**fitter_factory**
- 通过 `fitter_factory` 注入自定义 `AutoTauFitter`（可用于并行窗口搜索）。

---

## CyclesTauFitter

手动指定窗口，串行拟合多个周期。

```python
from autotau import CyclesTauFitter

fitter = CyclesTauFitter(
    time,
    signal,
    period=0.2,
    sample_rate=1000,
    window_on_offset=0.02,
    window_on_size=0.03,
    window_off_offset=0.12,
    window_off_size=0.03,
)
```

**常用方法**
- `fit_all_cycles(interp=True, points_after_interp=100)`
- `get_summary_data()`
- `plot_cycle_results()`
- `plot_all_fits()`
- `plot_r_squared_values()`
- `plot_windows_on_signal()`

---

## ParallelCyclesTauFitter

手动指定窗口，多进程并行拟合多个周期（分块并行）。

```python
from autotau import ParallelCyclesTauFitter

fitter = ParallelCyclesTauFitter(
    time,
    signal,
    period=0.2,
    sample_rate=1000,
    window_on_offset=0.02,
    window_on_size=0.03,
    window_off_offset=0.12,
    window_off_size=0.03,
    max_workers=4,
)

df = fitter.fit_all_cycles(return_format="dataframe")
```

**返回格式**
- `return_format="dataframe"`（默认）：`pandas.DataFrame`
- `return_format="records"`：字典列表（更接近旧版，但不包含每周期 `fitter` 对象）

---

## 已废弃（仍可用，但不推荐）

- `ParallelAutoTauFitter`：改用 `AutoTauFitter(..., executor=...)`
- `ParallelCyclesAutoTauFitter`：改用 `CyclesAutoTauFitter(..., fitter_factory=...)`
