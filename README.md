# AutoTau

[中文文档](README_CN.md) | English

AutoTau is a Python tool for fitting time constants (τ) in exponential rise/decay processes, designed for transient and periodic signal analysis.

**Models:**
- Turn-on (rise): `y = A * (1 - exp(-t/τ)) + C`
- Turn-off (decay): `y = A * exp(-t/τ) + C`

## Features

- Single-window τ fitting with R² and adjusted R² metrics
- Automatic sliding window search for optimal fit regions
- Multi-cycle batch fitting (serial and parallel)
- Decoupled workflow: separate window search from fitting
- PyQt5 GUI for interactive analysis
- Bilingual support (Chinese/English)

## Installation

```bash
pip install autotau
```

With GUI support:

```bash
pip install autotau[gui]
```

Local development:

```bash
pip install -e .
# or with GUI
pip install -e ".[gui]"
```

**Python version:** >=3.6

## Quick Start

### 1) Single Window Fitting (TauFitter)

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

### 2) Auto Window Search (AutoTauFitter)

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

### 3) Multi-Cycle Auto Fitting (CyclesAutoTauFitter)

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

### 4) Decoupled Workflow (WindowFinder + ParallelCyclesTauFitter)

For maximum flexibility, separate window search from fitting:

```python
from autotau import WindowFinder, ParallelCyclesTauFitter

# Step 1: Find best windows using first 2 cycles
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

print("On window offset:", windows['on']['offset'])
print("On window size:", windows['on']['size'])
print("On window R²:", windows['on']['r_squared'])

# Step 2: Apply windows to all cycles (parallel)
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

df = fitter.fit_all_cycles()  # Returns DataFrame by default
summary = fitter.get_summary_data()
fitter.plot_cycle_results()
```

## API Overview

| Class | Purpose | Use When |
|-------|---------|----------|
| `TauFitter` | Single window τ fitting | You know the exact fit window |
| `AutoTauFitter` | Auto window search + fitting | Single cycle, find best window automatically |
| `WindowFinder` | Window search only (no fitting) | Reuse windows across multiple datasets |
| `CyclesTauFitter` | Multi-cycle, manual windows (serial) | Few cycles (<10), known windows |
| `ParallelCyclesTauFitter` | Multi-cycle, manual windows (parallel) | Many cycles (>=10), known windows |
| `CyclesAutoTauFitter` | Multi-cycle, auto windows | Many cycles, unknown windows |
| `ParallelAutoTauFitter` | *(Deprecated)* | Use `AutoTauFitter(..., executor=...)` |
| `ParallelCyclesAutoTauFitter` | *(Deprecated)* | Use `CyclesAutoTauFitter(..., fitter_factory=...)` |

## Parallel Processing

### AutoTauFitter with Executor

```python
from concurrent.futures import ProcessPoolExecutor
from autotau import AutoTauFitter

with ProcessPoolExecutor(max_workers=8) as executor:
    fitter = AutoTauFitter(time, signal, sample_step=..., period=..., executor=executor)
    fitter.fit_tau_on_and_off()
```

### CyclesAutoTauFitter with Custom Factory

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

### Performance Recommendations

| Scenario | Recommended Class |
|----------|-------------------|
| Cycles < 10 | `CyclesTauFitter` (avoid multiprocessing overhead) |
| Cycles >= 10 | `ParallelCyclesTauFitter` (significant speedup) |
| Windows unknown | `CyclesAutoTauFitter` or `WindowFinder` + `ParallelCyclesTauFitter` |

**Note:** On Windows, wrap parallel code in `if __name__ == "__main__":` block.

## Advanced Workflows

### Exploring Different Window Parameters

Find optimal windows, then experiment with variations:

```python
from autotau import WindowFinder, ParallelCyclesTauFitter

# Find initial best windows
finder = WindowFinder(time_subset, signal_subset, sample_step=0.001, period=period)
windows = finder.find_best_windows()

# Fit with original windows
fitter1 = ParallelCyclesTauFitter(
    time, signal, period, sample_rate,
    window_on_offset=windows['on']['offset'],
    window_on_size=windows['on']['size'],
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size']
)
results1 = fitter1.fit_all_cycles()
summary1 = fitter1.get_summary_data()

# Experiment with adjusted parameters
fitter2 = ParallelCyclesTauFitter(
    time, signal, period, sample_rate,
    window_on_offset=windows['on']['offset'] + 0.1,  # Shift +0.1s
    window_on_size=windows['on']['size'] * 1.1,      # Enlarge 10%
    window_off_offset=windows['off']['offset'],
    window_off_size=windows['off']['size']
)
results2 = fitter2.fit_all_cycles()
summary2 = fitter2.get_summary_data()

# Compare
print("Original average R²:", summary1['r_squared_on'].mean())
print("Adjusted average R²:", summary2['r_squared_on'].mean())
```

## GUI Usage

v0.6.0 includes a PyQt5 GUI with decoupled architecture (avoids multiprocessing/matplotlib conflicts).

### Launch

```bash
# Module launch
python -m autotau.gui

# Command line (after installation)
autotau-gui
```

### Workflow

1. Copy two-column data from Excel (Time + Signal)
2. Click "Parse Clipboard"
3. Set Period and Sample Rate
4. Select mode (Auto/Manual)
5. Click "Fit Data"
6. View results, export to CSV/Excel

### GUI Features

- Auto mode: Automatic window search
- Manual mode: Specify window parameters
- Visualization: Original data, τ evolution, R² quality
- Export: CSV/Excel output

## API Parameters

### WindowFinder

| Parameter | Description | Default |
|-----------|-------------|---------|
| `time` | Time array | Required |
| `signal` | Signal array | Required |
| `sample_step` | Sampling step (seconds) | Required |
| `period` | Signal period (seconds) | Required |
| `window_scalar_min` | Minimum window size ratio | 1/5 |
| `window_scalar_max` | Maximum window size ratio | 1/3 |
| `show_progress` | Show progress bar | False |
| `max_workers` | Max worker processes | CPU count |

**Methods:**
- `find_best_windows()` - Returns dict with 'on' and 'off' window parameters
- `get_window_params()` - Get window parameters

### CyclesTauFitter / ParallelCyclesTauFitter

| Parameter | Description | Default |
|-----------|-------------|---------|
| `time` | Time array | Required |
| `signal` | Signal array | Required |
| `period` | Signal period (seconds) | Required |
| `sample_rate` | Sampling rate (Hz) | Required |
| `window_on_offset` | On window offset (seconds) | Required |
| `window_on_size` | On window size (seconds) | Required |
| `window_off_offset` | Off window offset (seconds) | Required |
| `window_off_size` | Off window size (seconds) | Required |
| `normalize` | Normalize signal | False |
| `language` | 'cn' or 'en' | 'en' |
| `show_progress` | Show progress (Parallel only) | False |
| `max_workers` | Worker processes (Parallel only) | CPU count |

**Methods:**
- `fit_all_cycles()` - Returns DataFrame (default) or list of records
- `get_summary_data()` - Get summary statistics
- `plot_cycle_results()` - Plot τ evolution
- `plot_r_squared_values()` - Plot R² quality
- `plot_windows_on_signal()` - Visualize windows on signal
- `plot_all_fits()` - Plot all fitting curves

## Important Notes

1. **WindowFinder** should use first 1-2 cycles of data for window search
2. Parallel processing may not improve performance with few cycles
3. Window offset is relative to each cycle's start time
4. Ensure window parameters don't exceed cycle boundaries
5. `ParallelCyclesTauFitter` uses chunked parallelism (each process handles a group of cycles) for efficiency

## Examples & Data

- Example script: `examples/basic_examples.py`
- Sample data: `autotau/transient.csv`

## Changelog

### v0.6.0

- **GUI fully decoupled**: GUI now uses independent fitting module, no longer depends on core implementation
- **Fixed GUI freezing**: Uses correct QThread signal mechanism for background fitting
- New `autotau.gui.fitters` module: `ExponentialFitter`, `WindowSearcher`, `CyclesFitter`, `AutoCyclesFitter`
- Avoids multiprocessing and matplotlib conflicts with GUI

### v0.5.0

- Added PyQt5 GUI
- Support clipboard paste, auto/manual fitting, visualization, export

### v0.4.5

- Optimized chunked parallel strategy
- Fixed window offset calculation

## License

MIT License, see `LICENSE`.
