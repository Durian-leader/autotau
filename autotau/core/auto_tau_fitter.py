import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .tau_fitter import TauFitter

class AutoTauFitter:
    """
    自动使用滑动窗口方法寻找最佳拟合区间来拟合tau值
    
    通过遍历不同窗口大小和位置，找到R²最高的拟合结果
    """
    
    def __init__(self, time, signal, sample_step, period, window_scalar_min=1/5, window_scalar_max=1/3, 
                 window_points_step=10, window_start_idx_step=1, normalize=False, language='cn', show_progress=False):
        """
        初始化AutoTauFitter
        
        参数:
        -----
        time : array-like
            时间数据
        signal : array-like
            信号数据
        sample_step : float
            采样步长(s)
        period : float
            信号周期(s)
        window_scalar_min : float, optional
            最小窗口大小相对于周期的比例
        window_scalar_max : float, optional
            最大窗口大小相对于周期的比例
        window_points_step : int, optional
            窗口点数步长，用于控制窗口大小搜索粒度
        window_start_idx_step : int, optional
            窗口起始位置步长，用于控制窗口位置搜索粒度
        normalize : bool, optional
            是否将信号归一化
        language : str, optional
            语言选择 ('cn'为中文, 'en'为英文)
        show_progress : bool, optional
            是否显示进度条
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.sample_step = sample_step
        self.period = period
        self.normalize = normalize
        self.language = language
        self.window_length_min = window_scalar_min * self.period
        self.window_length_max = window_scalar_max * self.period
        self.show_progress = show_progress

        self.window_points_step = window_points_step
        self.window_start_idx_step = window_start_idx_step

        # 最佳拟合结果和类
        self.best_tau_on_fitter = None
        self.best_tau_off_fitter = None

        # 最佳拟合窗口参数，开始时间，结束时间，窗口点数
        self.best_tau_on_window_start_time = None
        self.best_tau_off_window_start_time = None
        self.best_tau_on_window_end_time = None
        self.best_tau_off_window_end_time = None
        self.best_tau_on_window_size = None
        self.best_tau_off_window_size = None

    def fit_tau_on_and_off(self, interp=True, points_after_interp=100):
        """
        同时拟合开启(on)和关闭(off)过程的tau值
        
        使用滑动窗口法找到最佳拟合区间：
        1. 外层循环遍历窗口大小
        2. 内层循环遍历窗口起始位置
        3. 在每个窗口内，使用TauFitter拟合tau值
        4. 计算拟合质量(r_squared_adj)
        5. 保存拟合质量最高的窗口参数和结果
        
        参数:
            interp: 是否使用插值，默认为True
            points_after_interp: 插值后的点数，默认为100
            
        返回:
            tuple: (tau_on_popt, tau_on_r_squared_adj, tau_off_popt, tau_off_r_squared_adj)
        """
        # 初始化最佳拟合结果
        best_tau_on = {
            'r_squared_adj': 0,
            'window_size': 0,
            'window_start_time': 0,
            'window_end_time': 0,
            'popt': None,
            'fitter': None
        }
        
        best_tau_off = {
            'r_squared_adj': 0,
            'window_size': 0,
            'window_start_time': 0,
            'window_end_time': 0,
            'popt': None,
            'fitter': None
        }
        
        # 计算窗口大小的点数范围
        min_window_points = int(self.window_length_min / self.sample_step)
        max_window_points = int(self.window_length_max / self.sample_step)
        
        # 确保窗口大小至少有3个点
        min_window_points = max(3, min_window_points)
        
        # 外层循环：遍历窗口大小
        window_points_range = range(min_window_points, max_window_points + 1, self.window_points_step)
        if self.show_progress:
            window_points_range = tqdm(window_points_range, desc="窗口大小",leave=False)

        for window_points in window_points_range:
            # 内层循环：遍历窗口起始位置
            max_start_idx = len(self.time) - window_points
            start_idx_range = range(0, max_start_idx, self.window_start_idx_step)
            if self.show_progress:
                start_idx_range = tqdm(start_idx_range, desc="窗口起始位置", leave=False)
            
            for start_idx in start_idx_range:
                end_idx = start_idx + window_points
                
                # 提取当前窗口的时间和信号数据
                window_time = self.time[start_idx:end_idx]
                window_signal = self.signal[start_idx:end_idx]
                
                try:
                    # 尝试拟合on off过程
                    tau_fitter = TauFitter(
                        window_time,
                        window_signal,
                        t_on_idx=[window_time[0], window_time[-1]],
                        t_off_idx=[window_time[0], window_time[-1]],
                        language=self.language,
                        normalize=self.normalize
                    )
                    
                    tau_fitter.fit_tau_on(interp=interp, points_after_interp=points_after_interp)
                    tau_fitter.fit_tau_off(interp=interp, points_after_interp=points_after_interp)
                    r_squared_adj_on = tau_fitter.tau_on_r_squared_adj
                    r_squared_adj_off = tau_fitter.tau_off_r_squared_adj

                    # 更新最佳on拟合结果
                    if r_squared_adj_on > best_tau_on['r_squared_adj']:
                        best_tau_on['r_squared_adj'] = r_squared_adj_on
                        best_tau_on['window_size'] = window_points * self.sample_step
                        best_tau_on['window_start_time'] = window_time[0]
                        best_tau_on['window_end_time'] = window_time[-1]
                        best_tau_on['popt'] = tau_fitter.tau_on_popt
                        best_tau_on['fitter'] = tau_fitter
                    
                    # 更新最佳off拟合结果
                    if r_squared_adj_off > best_tau_off['r_squared_adj']:
                        best_tau_off['r_squared_adj'] = r_squared_adj_off
                        best_tau_off['window_size'] = window_points * self.sample_step
                        best_tau_off['window_start_time'] = window_time[0]
                        best_tau_off['window_end_time'] = window_time[-1]
                        best_tau_off['popt'] = tau_fitter.tau_off_popt
                        best_tau_off['fitter'] = tau_fitter
                            
                except Exception as e:
                    # 拟合失败，继续下一个窗口
                    continue
        
        # 保存最佳拟合结果到类属性
        self.best_tau_on_fitter = best_tau_on['fitter']
        self.best_tau_off_fitter = best_tau_off['fitter']
        
        # 保存窗口参数
        self.best_tau_on_window_start_time = best_tau_on['window_start_time']
        self.best_tau_on_window_end_time = best_tau_on['window_end_time']
        self.best_tau_on_window_size = best_tau_on['window_size']
        
        self.best_tau_off_window_start_time = best_tau_off['window_start_time']
        self.best_tau_off_window_end_time = best_tau_off['window_end_time']
        self.best_tau_off_window_size = best_tau_off['window_size']
        
        return (
            best_tau_on['popt'], 
            best_tau_on['r_squared_adj'], 
            best_tau_off['popt'], 
            best_tau_off['r_squared_adj']
        )
