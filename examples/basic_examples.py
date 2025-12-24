import os
import sys
import time

import numpy as np
import pandas as pd

# 添加包根目录到系统路径，便于直接运行示例
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autotau import (
    TauFitter,
    AutoTauFitter,
    CyclesAutoTauFitter,
    WindowFinder,
    CyclesTauFitter,
    ParallelCyclesTauFitter,
)


def load_example_data(file_path="../autotau/transient.csv", time_range=None, max_cycles=None, period=None):
    """加载示例数据，默认格式为 Time/Id 列。"""
    data = pd.read_csv(file_path)

    if time_range is not None:
        start_t, end_t = time_range
        data = data[(data["Time"] >= start_t) & (data["Time"] <= end_t)]

    if max_cycles is not None:
        if period is None:
            raise ValueError("max_cycles requires period to be provided")
        start_time = data["Time"].iloc[0]
        end_time = start_time + period * max_cycles
        data = data[data["Time"] <= end_time]

    time_data = data["Time"].to_numpy()
    signal_data = (-data["Id"]).to_numpy()
    return time_data, signal_data


def infer_sample_step(time_data):
    """从时间序列推断采样步长。"""
    if len(time_data) < 2:
        raise ValueError("time_data must contain at least two points")
    return float(np.median(np.diff(time_data)))


def format_value(value, digits=6):
    """格式化数值输出，支持 None。"""
    if value is None:
        return "None"
    return f"{value:.{digits}f}"


def basic_tau_fitter_example():
    """TauFitter 基本用法示例。"""
    print("运行 TauFitter 示例...")

    time_data, signal_data = load_example_data(time_range=(0.2, 0.25))

    fitter = TauFitter(
        time_data,
        signal_data,
        t_on_idx=[0.2, 0.21],
        t_off_idx=[0.24, 0.25],
        normalize=False,
        language="en",
    )

    fitter.fit_tau_on(interp=True, points_after_interp=200)
    fitter.fit_tau_off(interp=True, points_after_interp=200)

    print(f"tau_on: {format_value(fitter.get_tau_on())} s")
    print(f"tau_off: {format_value(fitter.get_tau_off())} s")
    print(
        "R2_on:",
        format_value(fitter.tau_on_r_squared, digits=4),
        "R2_adj_on:",
        format_value(fitter.tau_on_r_squared_adj, digits=4),
    )
    print(
        "R2_off:",
        format_value(fitter.tau_off_r_squared, digits=4),
        "R2_adj_off:",
        format_value(fitter.tau_off_r_squared_adj, digits=4),
    )

    # 可视化
    fitter.plot_tau_on()
    fitter.plot_tau_off()


def auto_tau_fitter_example():
    """AutoTauFitter 自动窗口搜索示例。"""
    print("运行 AutoTauFitter 示例...")

    time_data, signal_data = load_example_data(time_range=(0.2, 0.4))
    sample_step = infer_sample_step(time_data)
    period = 0.2

    auto_fitter = AutoTauFitter(
        time_data,
        signal_data,
        sample_step=sample_step,
        period=period,
        window_scalar_min=0.2,
        window_scalar_max=1 / 3,
        window_points_step=10,
        window_start_idx_step=1,
        normalize=False,
        language="en",
        show_progress=True,
    )

    auto_fitter.fit_tau_on_and_off()

    if auto_fitter.best_tau_on_fitter is not None:
        print(f"best tau_on: {format_value(auto_fitter.best_tau_on_fitter.get_tau_on())} s")
        auto_fitter.best_tau_on_fitter.plot_tau_on()
    if auto_fitter.best_tau_off_fitter is not None:
        print(f"best tau_off: {format_value(auto_fitter.best_tau_off_fitter.get_tau_off())} s")
        auto_fitter.best_tau_off_fitter.plot_tau_off()
    print(
        "on window:",
        auto_fitter.best_tau_on_window_start_time,
        auto_fitter.best_tau_on_window_end_time,
    )


def cycles_auto_tau_fitter_example():
    """CyclesAutoTauFitter 多周期拟合示例。"""
    print("运行 CyclesAutoTauFitter 示例...")

    period = 0.2
    sample_rate = 1000
    time_data, signal_data = load_example_data(max_cycles=5, period=period)

    cycles_fitter = CyclesAutoTauFitter(
        time_data,
        signal_data,
        period=period,
        sample_rate=sample_rate,
        window_scalar_min=0.2,
        window_scalar_max=1 / 3,
        window_points_step=10,
        window_start_idx_step=1,
        normalize=False,
        language="en",
    )

    cycles_fitter.fit_all_cycles(r_squared_threshold=0.95)

    summary = cycles_fitter.get_summary_data()
    if summary is not None:
        print("结果摘要:")
        print(summary.head())

    cycles_fitter.plot_cycle_results()
    cycles_fitter.plot_windows_on_signal(num_cycles=3)
    cycles_fitter.plot_all_fits(num_cycles=2)


def parallel_example():
    """串行 vs 并行多周期拟合示例（手动窗口）。"""
    print("运行 并行处理对比示例...")

    period = 0.2
    sample_rate = 1000
    time_data, signal_data = load_example_data(max_cycles=6, period=period)

    # 先用前两个周期搜索窗口
    two_period_mask = time_data <= time_data[0] + 2 * period
    time_subset = time_data[two_period_mask]
    signal_subset = signal_data[two_period_mask]

    finder = WindowFinder(
        time_subset,
        signal_subset,
        sample_step=1 / sample_rate,
        period=period,
        window_scalar_min=0.2,
        window_scalar_max=1 / 3,
        window_points_step=10,
        window_start_idx_step=1,
        normalize=False,
        language="en",
        show_progress=True,
        max_workers=4,
    )
    windows = finder.find_best_windows()

    window_on_offset = windows["on"]["offset"] % period
    window_off_offset = windows["off"]["offset"] % period
    window_on_size = windows["on"]["size"]
    window_off_size = windows["off"]["size"]

    # 串行拟合
    serial_start = time.time()
    serial_fitter = CyclesTauFitter(
        time_data,
        signal_data,
        period=period,
        sample_rate=sample_rate,
        window_on_offset=window_on_offset,
        window_on_size=window_on_size,
        window_off_offset=window_off_offset,
        window_off_size=window_off_size,
        normalize=False,
        language="en",
    )
    serial_fitter.fit_all_cycles()
    serial_time = time.time() - serial_start

    # 并行拟合
    parallel_start = time.time()
    parallel_fitter = ParallelCyclesTauFitter(
        time_data,
        signal_data,
        period=period,
        sample_rate=sample_rate,
        window_on_offset=window_on_offset,
        window_on_size=window_on_size,
        window_off_offset=window_off_offset,
        window_off_size=window_off_size,
        normalize=False,
        language="en",
        show_progress=True,
        max_workers=4,
    )
    parallel_fitter.fit_all_cycles()
    parallel_time = time.time() - parallel_start

    print(f"串行处理时间: {serial_time:.2f} s")
    print(f"并行处理时间: {parallel_time:.2f} s")
    if parallel_time > 0:
        print(f"加速比: {serial_time / parallel_time:.2f}x")


def compare_performance():
    """性能比较入口。"""
    parallel_example()


if __name__ == "__main__":
    print("AutoTau 示例程序")
    print("=" * 50)

    basic_tau_fitter_example()
    auto_tau_fitter_example()
    cycles_auto_tau_fitter_example()
    compare_performance()
