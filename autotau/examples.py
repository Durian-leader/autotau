import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import sys

# 添加当前目录到系统路径，以便可以导入autotau模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入autotau模块
from autotau import (
    TauFitter, 
    AutoTauFitter, 
    CyclesAutoTauFitter,
    ParallelAutoTauFitter,
    ParallelCyclesAutoTauFitter
)



def normalize_signal(signal):
    """将信号归一化到0-1范围"""
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max - signal_min > 1e-10:
        return (signal - signal_min) / (signal_max - signal_min)
    else:
        return np.zeros_like(signal)

def load_example_data(file_path='transient.csv', time_range=None):
    """
    加载示例数据
    
    参数:
    -----
    file_path : str
        数据文件路径
    time_range : tuple, optional
        时间范围 (min, max)
        
    返回:
    -----
    tuple
        (time_data, current_data)
    """
    try:
        data = pd.read_csv(file_path)
        if time_range:
            data = data[(data['Time'] >= time_range[0]) & (data['Time'] <= time_range[1])]
        time_data = data['Time'].values
        current_data = -data['Id'].values  # 反相电流
        return time_data, current_data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        # 创建模拟数据
        print("创建模拟数据...")
        t = np.linspace(0, 10, 10000)
        period = 0.2
        signal = np.zeros_like(t)
        
        for i in range(len(t)):
            cycle = int(t[i] / period)
            phase = (t[i] % period) / period
            
            if phase < 0.5:  # 开启阶段
                signal[i] = 1 - np.exp(-(phase*5)**2)
            else:  # 关闭阶段
                signal[i] = np.exp(-((phase-0.5)*5)**2)
        
        return t, signal

def compare_performance():
    """
    比较串行和并行处理的性能
    """
    print("加载数据...")
    time_data, current_data = load_example_data()
    
    period = 0.2  # 假设周期为0.2秒
    sample_step = 0.001  # 采样步长
    
    # 只取前两个周期的数据用于AutoTauFitter
    two_cycles_end_time = period * 2
    two_cycles_mask = time_data <= two_cycles_end_time
    time_data_two_cycles = time_data[two_cycles_mask]
    current_data_two_cycles = current_data[two_cycles_mask]
    
    print("=== 性能对比测试 ===")
    
    # 测试AutoTauFitter - 只使用前两个周期
    print("\n串行版自动拟合器:")
    t_start = time.time()
        auto_fitter = AutoTauFitter(
            time_data_two_cycles,
            current_data_two_cycles,
            sample_step=sample_step,
            period=period,
            window_scalar_min=0.2,
            window_scalar_max=1/3,
            window_points_step=10,
            window_start_idx_step=2,
            normalize=False,
            language='en',
            show_progress=True
        )    auto_fitter.fit_tau_on_and_off(interp=True)
    t_end = time.time()
    serial_time = t_end - t_start
    print(f"串行自动拟合耗时: {serial_time:.2f} 秒")
    
    # 测试ParallelAutoTauFitter - 只使用前两个周期
    print("\n并行版自动拟合器:")
    t_start = time.time()
        parallel_auto_fitter = ParallelAutoTauFitter(
            time_data_two_cycles,
            current_data_two_cycles,
            sample_step=sample_step,
            period=period,
            window_scalar_min=0.2,
            window_scalar_max=1/3,
            window_points_step=10,
            window_start_idx_step=2,
            normalize=False,
            language='en',
            show_progress=True
        )    parallel_auto_fitter.fit_tau_on_and_off(interp=True)
    t_end = time.time()
    parallel_time = t_end - t_start
    print(f"并行自动拟合耗时: {parallel_time:.2f} 秒")
    
    speedup = serial_time / parallel_time if parallel_time > 0 else 0
    print(f"加速比: {speedup:.2f}x")
    
    # 检查结果是否一致
    print("\n检查结果是否一致:")
    tau_on_serial = auto_fitter.best_tau_on_fitter.get_tau_on()
    tau_off_serial = auto_fitter.best_tau_off_fitter.get_tau_off()
    
    tau_on_parallel = parallel_auto_fitter.best_tau_on_fitter.get_tau_on()
    tau_off_parallel = parallel_auto_fitter.best_tau_off_fitter.get_tau_off()
    
    print(f"串行 tau_on: {tau_on_serial:.6f}, tau_off: {tau_off_serial:.6f}")
    print(f"并行 tau_on: {tau_on_parallel:.6f}, tau_off: {tau_off_parallel:.6f}")
    
    # 测试多周期处理 - 使用全部数据
    print("\n\n=== 多周期处理性能对比 ===")
    
    print("\n串行版多周期自动拟合器:")
    t_start = time.time()
    cycles_fitter = CyclesAutoTauFitter(
        time_data,
        current_data,
        period=period,
        sample_rate=1/sample_step,
        window_scalar_min=0.2,
        window_scalar_max=1/3,
        window_points_step=10,
        window_start_idx_step=2,
        normalize=False,
        language='en',
        show_progress=True
    )
    cycles_fitter.fit_all_cycles(interp=True)
    t_end = time.time()
    serial_cycles_time = t_end - t_start
    print(f"串行多周期自动拟合耗时: {serial_cycles_time:.2f} 秒")
    
    print("\n并行版多周期自动拟合器:")
    t_start = time.time()
    parallel_cycles_fitter = ParallelCyclesAutoTauFitter(
        time_data,
        current_data,
        period=period,
        sample_rate=1/sample_step,
        window_scalar_min=0.2,
        window_scalar_max=1/3,
        window_points_step=10,
        window_start_idx_step=2,
        normalize=False,
        language='en',
        show_progress=True,
        max_workers=None  # 使用所有可用CPU核心
    )
    parallel_cycles_fitter.fit_all_cycles(interp=True)
    t_end = time.time()
    parallel_cycles_time = t_end - t_start
    print(f"并行多周期自动拟合耗时: {parallel_cycles_time:.2f} 秒")
    
    cycles_speedup = serial_cycles_time / parallel_cycles_time if parallel_cycles_time > 0 else 0
    print(f"多周期加速比: {cycles_speedup:.2f}x")
    
    # 比较结果
    print("\n比较多周期处理结果:")
    serial_summary = cycles_fitter.get_summary_data()
    parallel_summary = parallel_cycles_fitter.get_summary_data()
    
    if serial_summary is not None and parallel_summary is not None:
        print(f"串行处理周期数: {len(serial_summary)}")
        print(f"并行处理周期数: {len(parallel_summary)}")
        
        # 打印前3个周期的tau值比较
        for i in range(min(3, len(serial_summary), len(parallel_summary))):
            print(f"\n周期 {i+1}:")
            print(f"  串行: tau_on={serial_summary['tau_on'].iloc[i]:.6f}, tau_off={serial_summary['tau_off'].iloc[i]:.6f}")
            print(f"  并行: tau_on={parallel_summary['tau_on'].iloc[i]:.6f}, tau_off={parallel_summary['tau_off'].iloc[i]:.6f}")

def plot_parallel_results():
    """
    展示并行处理结果
    """
    print("加载数据...")
    time_data, current_data = load_example_data()
    
    period = 0.2  # 假设周期为0.2秒
    sample_rate = 1000  # 采样率
    
    print("使用并行多周期自动拟合器...")
    parallel_cycles_fitter = ParallelCyclesAutoTauFitter(
        time_data,
        current_data,
        period=period,
        sample_rate=sample_rate,
        window_scalar_min=0.2,
        window_scalar_max=1/3,
        window_points_step=10,
        window_start_idx_step=2,
        normalize=False,
        language='en',
        show_progress=True,
        max_workers=None  # 使用所有可用CPU核心
    )
    
    print("拟合所有周期...")
    parallel_cycles_fitter.fit_all_cycles(interp=True)
    
    print("绘制结果...")
    # 绘制tau值随周期的变化
    parallel_cycles_fitter.plot_cycle_results(figsize=(10, 6))
    
    # 绘制R²值随周期的变化
    parallel_cycles_fitter.plot_r_squared_values(figsize=(10, 6))
    
    # 显示带窗口的信号
    parallel_cycles_fitter.plot_windows_on_signal(num_cycles=5, figsize=(12, 6))
    
    # 绘制拟合结果
    parallel_cycles_fitter.plot_all_fits(num_cycles=3, figsize=(15, 10))

    # 打印摘要数据
    summary = parallel_cycles_fitter.get_summary_data()
    print("\n拟合结果摘要:")
    print(summary)

if __name__ == "__main__":
    print("AutoTau并行处理示例")
    print("====================")
    
    # 性能对比
    compare_performance()
    
    # 绘制并行处理结果
    plot_parallel_results()
