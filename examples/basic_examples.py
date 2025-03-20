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

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def normalize_signal(signal):
    """将信号归一化到0-1范围"""
    signal_min = np.min(signal)
    signal_max = np.max(signal)
    if signal_max - signal_min > 1e-10:
        return (signal - signal_min) / (signal_max - signal_min)
    else:
        return np.zeros_like(signal)

def load_example_data(file_path='../autotau/transient.csv', time_range=None):
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
        
        # 应用时间范围过滤（如果提供）
        if time_range is not None:
            min_time, max_time = time_range
            data = data[(data['Time'] >= min_time) & (data['Time'] <= max_time)]
            
        time_data = data['Time'].values
        current_data = -data['Id'].values  # 反相电流
        
        return time_data, current_data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

def basic_tau_fitter_example():
    """TauFitter基本使用示例"""
    print("运行TauFitter基本示例...")
    
    # 加载数据
    time_data, current_data = load_example_data(time_range=(0.2, 0.25))
    
    # 创建TauFitter对象
    tau_fitter = TauFitter(
        time_data, 
        current_data, 
        t_on_idx=[0.2, 0.21],  # 开启过程时间窗口
        t_off_idx=[0.24, 0.25]  # 关闭过程时间窗口
    )
    
    # 拟合并获取结果
    tau_fitter.fit_tau_on()
    tau_fitter.fit_tau_off()
    
    print(f"tau_on: {tau_fitter.get_tau_on():.6f} s")
    print(f"R²_on: {tau_fitter.get_r_squared_on():.4f}")
    
    print(f"tau_off: {tau_fitter.get_tau_off():.6f} s")
    print(f"R²_off: {tau_fitter.get_r_squared_off():.4f}")
    
    # 可视化结果
    tau_fitter.plot_tau_on()
    tau_fitter.plot_tau_off()
    
    return tau_fitter

def auto_tau_fitter_example():
    """AutoTauFitter示例"""
    print("运行AutoTauFitter示例...")
    
    # 加载数据
    time_data, current_data = load_example_data(time_range=(0.2, 0.4))
    
    # 自动寻找最佳拟合窗口
    auto_fitter = AutoTauFitter(
        time_data, 
        current_data,
        sample_step=0.001,
        period=0.2,
        window_scalar_min=0.2,
        window_scalar_max=1/3,
        window_points_step=5,
        window_start_idx_step=2,
        normalize=False,
        language='cn',
        show_progress=True
    )
    
    print("拟合开启和关闭过程...")
    auto_fitter.fit_tau_on_and_off()
    
    # 获取结果
    print(f"最佳tau_on: {auto_fitter.best_tau_on_fitter.get_tau_on():.6f} s")
    print(f"最佳R²_on: {auto_fitter.best_tau_on_fitter.get_r_squared_on():.4f}")
    
    print(f"最佳tau_off: {auto_fitter.best_tau_off_fitter.get_tau_off():.6f} s")
    print(f"最佳R²_off: {auto_fitter.best_tau_off_fitter.get_r_squared_off():.4f}")
    
    # 可视化结果
    auto_fitter.best_tau_on_fitter.plot_tau_on()
    auto_fitter.best_tau_off_fitter.plot_tau_off()
    
    return auto_fitter

def cycles_auto_tau_fitter_example():
    """CyclesAutoTauFitter示例"""
    print("运行CyclesAutoTauFitter示例...")
    
    # 加载数据
    time_data, current_data = load_example_data()
    
    # 处理多周期数据
    cycles_fitter = CyclesAutoTauFitter(
        time_data,
        current_data,
        period=0.2,
        sample_rate=1000,
        window_scalar_min=0.2,
        window_scalar_max=1/3,
        window_points_step=5,
        window_start_idx_step=2,
        normalize=False,
        language='cn',
        show_progress=True
    )
    
    print("拟合所有周期...")
    cycles_fitter.fit_all_cycles()
    
    # 可视化结果
    cycles_fitter.plot_cycle_results()
    cycles_fitter.plot_windows_on_signal(num_cycles=3)
    cycles_fitter.plot_all_fits(num_cycles=2)
    
    # 获取结果摘要
    summary = cycles_fitter.get_summary_data()
    print("\n结果摘要:")
    print(summary)
    
    return cycles_fitter

def parallel_example():
    """并行处理示例"""
    print("运行并行处理示例...")
    
    # 加载数据
    time_data, current_data = load_example_data()
    
    # 串行版多周期拟合器
    print("创建串行版多周期拟合器...")
    serial_start = time.time()
    
    serial_cycles_fitter = CyclesAutoTauFitter(
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
        max_cycles=5  # 仅处理前5个周期以加快示例运行速度
    )
    
    serial_cycles_fitter.fit_all_cycles()
    serial_end = time.time()
    serial_time = serial_end - serial_start
    
    # 并行版多周期拟合器
    print("\n创建并行版多周期拟合器...")
    parallel_start = time.time()
    
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
        max_cycles=5,  # 仅处理前5个周期以加快示例运行速度
        max_workers=None  # 使用所有可用CPU核心
    )
    
    parallel_cycles_fitter.fit_all_cycles()
    parallel_end = time.time()
    parallel_time = parallel_end - parallel_start
    
    # 比较性能
    print("\n性能比较:")
    print(f"串行处理时间: {serial_time:.2f} 秒")
    print(f"并行处理时间: {parallel_time:.2f} 秒")
    print(f"加速比: {serial_time/parallel_time:.2f}x")
    
    return serial_cycles_fitter, parallel_cycles_fitter

def compare_performance():
    """性能比较函数"""
    return parallel_example()

if __name__ == "__main__":
    print("AutoTau示例程序")
    print("=" * 50)
    
    # 运行基本示例
    basic_tau_fitter_example()
    
    # 运行自动拟合示例
    auto_tau_fitter_example()
    
    # 运行多周期示例
    cycles_auto_tau_fitter_example()
    
    # 运行并行处理性能比较
    compare_performance() 