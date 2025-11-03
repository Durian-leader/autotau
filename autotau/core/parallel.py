import numpy as np
import concurrent.futures
import multiprocessing
import pandas as pd
from tqdm import tqdm
from .tau_fitter import TauFitter
from .auto_tau_fitter import AutoTauFitter
import matplotlib.pyplot as plt

class ParallelAutoTauFitter:
    """
    AutoTauFitter的并行版本，使用多进程加速窗口搜索
    
    利用多核CPU并行处理不同窗口大小和位置的拟合任务，大幅提升滑动窗口搜索速度
    """
    
    def __init__(self, time, signal, sample_step, period, window_scalar_min=1/5, window_scalar_max=1/3, 
                 window_points_step=10, window_start_idx_step=1, normalize=False, language='en', 
                 show_progress=False, max_workers=None):
        """
        初始化并行版AutoTauFitter
        
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
        max_workers : int, optional
            最大工作进程数，默认为None，表示使用系统CPU核心数
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
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()

        self.window_points_step = window_points_step
        self.window_start_idx_step = window_start_idx_step

        # 最佳拟合结果
        self.best_tau_on_fitter = None
        self.best_tau_off_fitter = None

        # 最佳拟合窗口参数
        self.best_tau_on_window_start_time = None
        self.best_tau_off_window_start_time = None
        self.best_tau_on_window_end_time = None
        self.best_tau_off_window_end_time = None
        self.best_tau_on_window_size = None
        self.best_tau_off_window_size = None

    def _process_window(self, window_params, interp=True, points_after_interp=100):
        """
        处理单个窗口的拟合任务，被并行调用
        
        参数:
        -----
        window_params : tuple
            窗口参数 (window_points, start_idx)
        interp : bool, optional
            是否使用插值
        points_after_interp : int, optional
            插值后的点数
            
        返回:
        -----
        dict
            拟合结果字典，包含on和off的拟合结果
        """
        window_points, start_idx = window_params
        end_idx = start_idx + window_points
        
        # 提取当前窗口的时间和信号数据
        window_time = self.time[start_idx:end_idx]
        window_signal = self.signal[start_idx:end_idx]
        
        try:
            # 尝试拟合on和off过程
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
            
            # 收集结果
            result = {
                'on': {
                    'r_squared_adj': tau_fitter.tau_on_r_squared_adj if tau_fitter.tau_on_r_squared_adj is not None else 0,
                    'window_size': window_points * self.sample_step,
                    'window_start_time': window_time[0],
                    'window_end_time': window_time[-1],
                    'popt': tau_fitter.tau_on_popt,
                    'fitter': tau_fitter
                },
                'off': {
                    'r_squared_adj': tau_fitter.tau_off_r_squared_adj if tau_fitter.tau_off_r_squared_adj is not None else 0,
                    'window_size': window_points * self.sample_step,
                    'window_start_time': window_time[0],
                    'window_end_time': window_time[-1],
                    'popt': tau_fitter.tau_off_popt,
                    'fitter': tau_fitter
                }
            }
            return result
        except Exception as e:
            # 拟合失败，返回None
            return None

    def _process_window_wrapper(self, window_params, interp, points_after_interp):
        """
        一个可序列化的普通函数，用于包装_process_window方法
        
        参数:
        -----
        window_params : tuple
            窗口参数 (window_points, start_idx)
        interp : bool
            是否使用插值
        points_after_interp : int
            插值后的点数
            
        返回:
        -----
        dict
            拟合结果字典，包含on和off的拟合结果
        """
        return self._process_window(window_params, interp, points_after_interp)

    def fit_tau_on_and_off(self, interp=True, points_after_interp=100):
        """
        使用并行处理同时拟合开启和关闭过程的tau值
        
        参数:
        -----
        interp : bool, optional
            是否使用插值，默认为True
        points_after_interp : int, optional
            插值后的点数，默认为100
            
        返回:
        -----
        tuple
            (tau_on_popt, tau_on_r_squared_adj, tau_off_popt, tau_off_r_squared_adj)
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
        
        # 创建所有需要处理的窗口参数列表
        window_params_list = []
        for window_points in range(min_window_points, max_window_points + 1, self.window_points_step):
            max_start_idx = len(self.time) - window_points
            for start_idx in range(0, max_start_idx, self.window_start_idx_step):
                window_params_list.append((window_points, start_idx))
        
        # 显示总任务数
        total_tasks = len(window_params_list)
        if self.show_progress:
            print(f"总共需要处理 {total_tasks} 个窗口拟合任务")
        
        # 使用ProcessPoolExecutor并行处理所有窗口
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 使用一个可序列化的普通函数，而不是lambda
            process_function = self._process_window_wrapper
            
            if self.show_progress:
                # 使用tqdm显示进度
                results = list(tqdm(
                    executor.map(
                        process_function, 
                        window_params_list,
                        [interp] * len(window_params_list),
                        [points_after_interp] * len(window_params_list)
                    ),
                    total=total_tasks,
                    desc="并行拟合进度"
                ))
            else:
                # 不显示进度
                results = list(executor.map(
                    process_function,
                    window_params_list,
                    [interp] * len(window_params_list),
                    [points_after_interp] * len(window_params_list)
                ))
        
        # 处理结果，找到最佳拟合
        for result in results:
            if result is None:
                continue
                
            # 检查on拟合结果
            on_result = result['on']
            if on_result['r_squared_adj'] > best_tau_on['r_squared_adj']:
                best_tau_on = on_result
            
            # 检查off拟合结果
            off_result = result['off']
            if off_result['r_squared_adj'] > best_tau_off['r_squared_adj']:
                best_tau_off = off_result
        
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


class ParallelCyclesAutoTauFitter:
    """
    CyclesAutoTauFitter的并行版本，使用多进程加速多个周期的处理
    
    利用多核CPU并行处理不同周期的拟合任务，大幅提升多周期数据的处理速度
    """
    
    def __init__(self, time, signal, period, sample_rate, **kwargs):
        """
        初始化并行版CyclesAutoTauFitter
        
        参数:
        -----
        time : array-like
            时间数据
        signal : array-like
            信号数据
        period : float
            信号周期(s)
        sample_rate : float
            采样率(Hz)
        **kwargs :
            传递给AutoTauFitter的额外参数，如:
            window_scalar_min, window_scalar_max, window_points_step, window_start_idx_step, 
            normalize, language, show_progress, max_workers等
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.auto_tau_fitter_params = kwargs
        self.max_workers = kwargs.get('max_workers', multiprocessing.cpu_count())
        self.show_progress = kwargs.get('show_progress', False)
        self.language = kwargs.get('language', 'en')
        
        # 结果存储
        self.cycle_results = []
        self.refitted_cycles = []
        self.last_r_squared_threshold = 0.95

        # 窗口参数
        self.window_on_offset = None
        self.window_off_offset = None
        self.window_on_size = None
        self.window_off_size = None
        self.initial_auto_fitter = None
        
        # 确保传递给下层类的参数中删除max_workers
        if 'max_workers' in self.auto_tau_fitter_params:
            self.parallel_params = self.auto_tau_fitter_params.copy()
        else:
            self.parallel_params = self.auto_tau_fitter_params

        # 语言字典
        self.text = {
            'cn': {
                'no_results': '没有可用的周期结果。请先运行fit_all_cycles()。',
                'tau_on': 'Tau On',
                'tau_off': 'Tau Off',
                'refitted_tau_on': '重新拟合 Tau On',
                'refitted_tau_off': '重新拟合 Tau Off',
                'cycle': '周期',
                'tau_s': 'Tau (s)',
                'tau_values_per_cycle': '每个周期的Tau值',
                'r_squared_on': 'R² On',
                'r_squared_off': 'R² Off',
                'refitted_on': '重新拟合 On',
                'refitted_off': '重新拟合 Off',
                'threshold': '阈值',
                'r_squared_values_per_cycle': '每个周期的R²值',
                'r_squared': 'R²',
                'windows_not_determined': '尚未确定窗口。请先运行find_best_windows()或fit_all_cycles()。',
                'signal': '信号',
                'full_signal_with_windows': '完整信号与开启和关闭窗口',
                'signal_with_windows_cycles': '信号与开启和关闭窗口(周期{}至{})',
                'start_cycle_exceeds_available': '起始周期{}超过可用周期{}',
                'time_s': '时间 (s)',
                'on_window': '开启窗口',
                'off_window': '关闭窗口',
                'invalid_start_cycle': '无效的start_cycle',
                'must_be_between': '必须介于0和',
                'data': '数据',
                'fit': '拟合',
                'cycle_on_fit': '周期{} - 开启拟合 (τ = {:.5f} s, R² = {:.3f})',
                'refitted': ' [已重新拟合]',
                'cycle_off_fit': '周期{} - 关闭拟合 (τ = {:.5f} s, R² = {:.3f})',
                'fit_results_for_cycles': '周期{}-{}的拟合结果',
            },
            'en': {
                'no_results': 'No cycle results available. Please run fit_all_cycles() first.',
                'tau_on': 'Tau On',
                'tau_off': 'Tau Off',
                'refitted_tau_on': 'Refitted Tau On',
                'refitted_tau_off': 'Refitted Tau Off',
                'cycle': 'Cycle',
                'tau_s': 'Tau (s)',
                'tau_values_per_cycle': 'Tau Values per Cycle',
                'r_squared_on': 'R² On',
                'r_squared_off': 'R² Off',
                'refitted_on': 'Refitted On',
                'refitted_off': 'Refitted Off',
                'threshold': 'Threshold',
                'r_squared_values_per_cycle': 'R² Values per Cycle',
                'r_squared': 'R²',
                'windows_not_determined': 'Windows not determined yet. Please run find_best_windows() or fit_all_cycles() first.',
                'signal': 'Signal',
                'full_signal_with_windows': 'Full Signal with On and Off Windows',
                'signal_with_windows_cycles': 'Signal with On and Off Windows (Cycles {} to {})',
                'start_cycle_exceeds_available': 'Start cycle {} exceeds available cycles {}',
                'time_s': 'Time (s)',
                'on_window': 'On Window',
                'off_window': 'Off Window',
                'invalid_start_cycle': 'Invalid start_cycle',
                'must_be_between': 'must be between 0 and',
                'data': 'Data',
                'fit': 'Fit',
                'cycle_on_fit': 'Cycle {} - On Fit (τ = {:.5f} s, R² = {:.3f})',
                'refitted': ' [Refitted]',
                'cycle_off_fit': 'Cycle {} - Off Fit (τ = {:.5f} s, R² = {:.3f})',
                'fit_results_for_cycles': 'Fit Results for Cycles {}-{}',
            }
        }
    
    def find_best_windows(self, interp=True, points_after_interp=100):
        """
        使用并行版AutoTauFitter从前两个周期找到最佳拟合窗口
        
        参数:
        -----
        interp : bool, optional
            是否使用插值，默认为True
        points_after_interp : int, optional
            插值后的点数，默认为100
            
        返回:
        -----
        ParallelAutoTauFitter
            用于找到最佳窗口的ParallelAutoTauFitter实例
        """
        # 提取前两个周期
        two_period_mask = (self.time <= self.time[0] + 2 * self.period)
        time_subset = self.time[two_period_mask]
        signal_subset = self.signal[two_period_mask]

        # 使用并行版AutoTauFitter
        auto_fitter = ParallelAutoTauFitter(
            time_subset,
            signal_subset,
            sample_step=1/self.sample_rate,
            period=self.period,
            max_workers=self.max_workers,
            **{k: v for k, v in self.auto_tau_fitter_params.items() 
               if k not in ['max_workers']}
        )
        
        auto_fitter.fit_tau_on_and_off(interp=interp, points_after_interp=points_after_interp)

        # 存储最佳窗口参数（取模确保偏移量在单个周期范围内）
        self.window_on_offset = (auto_fitter.best_tau_on_window_start_time - self.time[0]) % self.period
        self.window_off_offset = (auto_fitter.best_tau_off_window_start_time - self.time[0]) % self.period
        self.window_on_size = auto_fitter.best_tau_on_window_size
        self.window_off_size = auto_fitter.best_tau_off_window_size
        self.initial_auto_fitter = auto_fitter

        return auto_fitter
    
    def _process_cycle(self, cycle_data):
        """
        处理单个周期的拟合任务，被并行调用
        
        参数:
        -----
        cycle_data : dict
            周期数据，包含:
            - cycle_index: 周期索引
            - t_on_idx: 开启窗口 [开始时间, 结束时间]
            - t_off_idx: 关闭窗口 [开始时间, 结束时间]
            - time: 时间数据
            - signal: 信号数据
            - interp: 是否使用插值
            - points_after_interp: 插值后点数
            - r_squared_threshold: R²阈值
            
        返回:
        -----
        dict
            周期处理结果
        """
        i = cycle_data['cycle_index']
        on_window_start, on_window_end = cycle_data['t_on_idx']
        off_window_start, off_window_end = cycle_data['t_off_idx']
        time = cycle_data['time']
        signal = cycle_data['signal']
        interp = cycle_data['interp']
        points_after_interp = cycle_data['points_after_interp']
        r_squared_threshold = cycle_data['r_squared_threshold']
        auto_tau_fitter_params = cycle_data['auto_tau_fitter_params']
        cycle_start_time = cycle_data['cycle_start_time']
        
        # 创建窗口的掩码
        on_mask = (time >= on_window_start) & (time <= on_window_end)
        off_mask = (time >= off_window_start) & (time <= off_window_end)

        # 检查是否有足够的数据点
        if np.sum(on_mask) < 3 or np.sum(off_mask) < 3:
            return {
                'status': 'skipped',
                'message': f"周期{i+1}的数据点不足。"
            }

        # 为此周期创建一个TauFitter
        tau_fitter = TauFitter(
            time,
            signal,
            t_on_idx=[on_window_start, on_window_end],
            t_off_idx=[off_window_start, off_window_end],
            **{k: v for k, v in auto_tau_fitter_params.items() if k in ['normalize', 'language']}
        )

        # 用指定的插值设置拟合数据
        tau_fitter.fit_tau_on(interp=interp, points_after_interp=points_after_interp)
        tau_fitter.fit_tau_off(interp=interp, points_after_interp=points_after_interp)

        # 检查拟合是否足够好(R² > 阈值)
        r_squared_on = tau_fitter.tau_on_r_squared
        r_squared_off = tau_fitter.tau_off_r_squared

        needs_refitting = False
        refit_type = []
        refit_info = None

        if r_squared_on < r_squared_threshold:
            needs_refitting = True
            refit_type.append('on')

        if r_squared_off < r_squared_threshold:
            needs_refitting = True
            refit_type.append('off')

        # 如果拟合质量不佳，尝试为此特定周期找到更好的窗口
        if needs_refitting:
            # 定义包含此周期加上一些余量的时间窗口
            cycle_start = cycle_start_time
            cycle_end = cycle_start_time + cycle_data['period']

            # 确保不超出边界
            cycle_start = max(cycle_start, time[0])
            cycle_end = min(cycle_end, time[-1])

            # 提取此周期的数据
            cycle_mask = (time >= cycle_start) & (time <= cycle_end)
            cycle_time = time[cycle_mask]
            cycle_signal = signal[cycle_mask]

            # 确保有足够的数据点
            if len(cycle_time) < 10:  # 最小值
                # 使用原始拟合
                pass
            else:
                # 记录重新拟合信息
                refit_info = {
                    'cycle': i + 1,
                    'original_r_squared_on': r_squared_on,
                    'original_r_squared_off': r_squared_off,
                    'refit_types': refit_type,
                }

                # 使用AutoTauFitter为这个周期找到更好的窗口
                cycle_auto_fitter = AutoTauFitter(
                    cycle_time,
                    cycle_signal,
                    sample_step=1/cycle_data['sample_rate'],
                    period=cycle_data['period'],
                    **{k: v for k, v in auto_tau_fitter_params.items() 
                       if k not in ['max_workers']}
                )

                cycle_auto_fitter.fit_tau_on_and_off(interp=interp, points_after_interp=points_after_interp)

                # 检查并应用更好的开启拟合（如果需要）
                if 'on' in refit_type:
                    new_r_squared_on = cycle_auto_fitter.best_tau_on_fitter.tau_on_r_squared if cycle_auto_fitter.best_tau_on_fitter else 0

                    if new_r_squared_on > r_squared_on:
                        tau_fitter.tau_on_popt = cycle_auto_fitter.best_tau_on_fitter.tau_on_popt
                        tau_fitter.tau_on_pcov = cycle_auto_fitter.best_tau_on_fitter.tau_on_pcov
                        tau_fitter.tau_on_r_squared = new_r_squared_on
                        tau_fitter.tau_on_r_squared_adj = cycle_auto_fitter.best_tau_on_fitter.tau_on_r_squared_adj
                        refit_info['new_r_squared_on'] = new_r_squared_on
                    else:
                        refit_info['new_r_squared_on'] = r_squared_on

                # 检查并应用更好的关闭拟合（如果需要）
                if 'off' in refit_type:
                    new_r_squared_off = cycle_auto_fitter.best_tau_off_fitter.tau_off_r_squared if cycle_auto_fitter.best_tau_off_fitter else 0

                    if new_r_squared_off > r_squared_off:
                        tau_fitter.tau_off_popt = cycle_auto_fitter.best_tau_off_fitter.tau_off_popt
                        tau_fitter.tau_off_pcov = cycle_auto_fitter.best_tau_off_fitter.tau_off_pcov
                        tau_fitter.tau_off_r_squared = new_r_squared_off
                        tau_fitter.tau_off_r_squared_adj = cycle_auto_fitter.best_tau_off_fitter.tau_off_r_squared_adj
                        refit_info['new_r_squared_off'] = new_r_squared_off
                    else:
                        refit_info['new_r_squared_off'] = r_squared_off

        # 存储结果
        result = {
            'status': 'success',
            'cycle': i + 1,
            'cycle_start_time': cycle_start_time,
            'tau_on': tau_fitter.get_tau_on(),
            'tau_off': tau_fitter.get_tau_off(),
            'tau_on_popt': tau_fitter.tau_on_popt,
            'tau_off_popt': tau_fitter.tau_off_popt,
            'tau_on_r_squared': tau_fitter.tau_on_r_squared,
            'tau_off_r_squared': tau_fitter.tau_off_r_squared,
            'tau_on_r_squared_adj': tau_fitter.tau_on_r_squared_adj,
            'tau_off_r_squared_adj': tau_fitter.tau_off_r_squared_adj,
            'fitter': tau_fitter,
            'was_refitted': needs_refitting,
            'refit_info': refit_info
        }

        return result
    
    def fit_all_cycles(self, interp=True, points_after_interp=100, r_squared_threshold=0.95):
        """
        使用并行处理拟合所有周期的tau值
        
        参数:
        -----
        interp : bool, optional
            是否使用插值，默认为True
        points_after_interp : int, optional
            插值后的点数，默认为100
        r_squared_threshold : float, optional
            R²阈值，如果低于此值，将尝试找到更好的拟合(默认: 0.95)
            
        返回:
        -----
        list of dict
            包含每个周期拟合结果的列表
        """
        # 存储阈值以供后续参考
        self.last_r_squared_threshold = r_squared_threshold

        if self.window_on_offset is None or self.window_off_offset is None:
            # 如果尚未找到窗口，找到它们
            self.find_best_windows(interp=interp, points_after_interp=points_after_interp)

        # 计算数据中完整周期的数量
        total_time = self.time[-1] - self.time[0]
        num_cycles = int(total_time / self.period)

        # 准备每个周期的处理参数
        cycles_data = []
        for i in range(num_cycles):
            cycle_start_time = self.time[0] + i * self.period

            # 计算此周期的窗口开始和结束时间
            on_window_start = cycle_start_time + self.window_on_offset
            on_window_end = on_window_start + self.window_on_size

            off_window_start = cycle_start_time + self.window_off_offset
            off_window_end = off_window_start + self.window_off_size

            # 创建周期数据字典
            cycle_data = {
                'cycle_index': i,
                'cycle_start_time': cycle_start_time,
                't_on_idx': [on_window_start, on_window_end],
                't_off_idx': [off_window_start, off_window_end],
                'time': self.time,
                'signal': self.signal,
                'interp': interp,
                'points_after_interp': points_after_interp,
                'r_squared_threshold': r_squared_threshold,
                'auto_tau_fitter_params': self.auto_tau_fitter_params,
                'sample_rate': self.sample_rate,
                'period': self.period
            }
            cycles_data.append(cycle_data)

        # 重置结果存储
        self.cycle_results = []
        self.refitted_cycles = []

        # 使用ProcessPoolExecutor并行处理所有周期
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            if self.show_progress:
                # 使用tqdm显示进度
                futures = list(tqdm(
                    executor.map(self._process_cycle, cycles_data),
                    total=len(cycles_data),
                    desc="并行处理周期"
                ))
            else:
                # 不显示进度
                futures = list(executor.map(self._process_cycle, cycles_data))

        # 处理结果
        for result in futures:
            if result['status'] == 'skipped':
                if self.show_progress:
                    print(result['message'])
                continue
            
            # 添加到结果列表
            self.cycle_results.append(result)
            
            # 如果重新拟合过，添加到重新拟合列表
            if result['was_refitted'] and result['refit_info'] is not None:
                self.refitted_cycles.append(result['refit_info'])
                
        # 按周期排序结果
        self.cycle_results.sort(key=lambda x: x['cycle'])
                
        return self.cycle_results

    # 后续方法与CyclesAutoTauFitter相同，可直接复用
    def get_summary_data(self):
        """
        返回拟合结果的摘要DataFrame
        
        返回:
        -----
        pandas.DataFrame
            包含周期号、开始时间、tau值和R平方值的DataFrame
        """
        if not self.cycle_results:
            print("没有可用的周期结果。请先运行fit_all_cycles()。")
            return None

        data = {
            'cycle': [],
            'cycle_start_time': [],
            'tau_on': [],
            'tau_off': [],
            'r_squared_on': [],
            'r_squared_off': [],
            'r_squared_adj_on': [],
            'r_squared_adj_off': [],
            'was_refitted': []
        }

        for res in self.cycle_results:
            data['cycle'].append(res['cycle'])
            data['cycle_start_time'].append(res['cycle_start_time'])
            data['tau_on'].append(res['tau_on'])
            data['tau_off'].append(res['tau_off'])
            data['r_squared_on'].append(res['tau_on_r_squared'])
            data['r_squared_off'].append(res['tau_off_r_squared'])
            data['r_squared_adj_on'].append(res['tau_on_r_squared_adj'])
            data['r_squared_adj_off'].append(res['tau_off_r_squared_adj'])
            data['was_refitted'].append(res.get('was_refitted', False))

        return pd.DataFrame(data)
        
    def get_refitted_cycles_info(self):
        """
        获取关于需要重新拟合的周期的详细信息
        
        返回:
        -----
        pandas.DataFrame
            包含有关重新拟合周期的信息的DataFrame，包括原始和新的R²值
        """
        if not hasattr(self, 'refitted_cycles') or not self.refitted_cycles:
            print("没有周期被重新拟合。")
            return None

        return pd.DataFrame(self.refitted_cycles)

    def plot_cycle_results(self, figsize=(10, 6)):
        """
        绘制所有周期的tau值
        
        参数:
        -----
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸
        """
        if not self.cycle_results:
            print(self.text[self.language]['no_results'])
            return

        cycles = [res['cycle'] for res in self.cycle_results]
        tau_on_values = [res['tau_on'] for res in self.cycle_results]
        tau_off_values = [res['tau_off'] for res in self.cycle_results]

        # 查找重新拟合的周期
        refitted_indices = [i for i, res in enumerate(self.cycle_results) if res.get('was_refitted', False)]
        refitted_cycles = [cycles[i] for i in refitted_indices]
        refitted_tau_on = [tau_on_values[i] for i in refitted_indices]
        refitted_tau_off = [tau_off_values[i] for i in refitted_indices]

        plt.figure(figsize=figsize)
        # 绘制所有周期
        plt.plot(cycles, tau_on_values, 'o-', label=self.text[self.language]['tau_on'], color='blue')
        plt.plot(cycles, tau_off_values, 'o-', label=self.text[self.language]['tau_off'], color='red')

        # 突出显示重新拟合的周期
        if refitted_cycles:
            plt.scatter(refitted_cycles, refitted_tau_on, s=100, facecolors='none', edgecolors='blue',
                        linewidth=2, label=self.text[self.language]['refitted_tau_on'])
            plt.scatter(refitted_cycles, refitted_tau_off, s=100, facecolors='none', edgecolors='red',
                        linewidth=2, label=self.text[self.language]['refitted_tau_off'])

        plt.xlabel(self.text[self.language]['cycle'])
        plt.ylabel(self.text[self.language]['tau_s'])
        plt.title(self.text[self.language]['tau_values_per_cycle'])
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_r_squared_values(self, figsize=(10, 6)):
        """
        绘制所有周期的R²值，突出显示重新拟合的周期
        
        参数:
        -----
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸
        """
        if not self.cycle_results:
            print(self.text[self.language]['no_results'])
            return

        summary = self.get_summary_data()

        plt.figure(figsize=figsize)

        # 创建x位置
        cycles = summary['cycle']

        # 绘制R²值
        plt.plot(cycles, summary['r_squared_on'], 'o-', label=self.text[self.language]['r_squared_on'], color='blue')
        plt.plot(cycles, summary['r_squared_off'], 'o-', label=self.text[self.language]['r_squared_off'], color='red')

        # 突出显示重新拟合的周期
        refitted = summary[summary['was_refitted']]
        if not refitted.empty:
            plt.scatter(refitted['cycle'], refitted['r_squared_on'],
                        s=100, facecolors='none', edgecolors='blue', linewidth=2,
                        label=self.text[self.language]['refitted_on'])
            plt.scatter(refitted['cycle'], refitted['r_squared_off'],
                        s=100, facecolors='none', edgecolors='red', linewidth=2,
                        label=self.text[self.language]['refitted_off'])

        # 在阈值处添加水平线(假设为0.95，如果未提供)
        if hasattr(self, 'last_r_squared_threshold'):
            threshold = self.last_r_squared_threshold
        else:
            threshold = 0.95

        plt.axhline(y=threshold, color='green', linestyle='--',
                    label=f'{self.text[self.language]["threshold"]} ({threshold})')

        plt.xlabel(self.text[self.language]['cycle'])
        plt.ylabel(self.text[self.language]['r_squared'])
        plt.title(self.text[self.language]['r_squared_values_per_cycle'])
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.show()

    def plot_windows_on_signal(self, plot_full_signal=False, start_cycle=0, num_cycles=5, figsize=(12, 6)):
        """
        绘制原始信号，突出显示开启和关闭过渡的窗口
        
        参数:
        -----
        plot_full_signal : bool, optional
            如果为True，则无论num_cycles如何，都绘制整个信号。默认为False。
        start_cycle : int, optional
            要绘制的第一个周期(从0开始索引)。默认为0。
        num_cycles : int, optional
            要绘制的周期数。默认为5。如果plot_full_signal=True，则忽略此参数。
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸。
        """
        if self.window_on_offset is None or self.window_off_offset is None:
            print(self.text[self.language]['windows_not_determined'])
            return

        # 创建图形
        plt.figure(figsize=figsize)

        if plot_full_signal:
            # 绘制整个信号
            plt.plot(self.time, self.signal, '-', label=self.text[self.language]['signal'])

            # 计算数据中有多少个完整周期
            total_cycles = int((self.time[-1] - self.time[0]) / self.period)

            # 为数据中的每个完整周期添加窗口
            for i in range(total_cycles):
                self._plot_cycle_windows(i, i==0)  # 仅将第一个周期包含在图例中
        else:
            # 计算有效周期范围
            max_cycle = int((self.time[-1] - self.time[0]) / self.period)
            if start_cycle >= max_cycle:
                print(self.text[self.language]['start_cycle_exceeds_available'].format(start_cycle, max_cycle))
                return

            # 计算要显示的周期范围
            end_cycle = min(start_cycle + num_cycles, max_cycle)
            actual_cycles = end_cycle - start_cycle

            # 计算要显示的时间范围
            start_time = self.time[0] + start_cycle * self.period
            end_time = self.time[0] + end_cycle * self.period

            # 过滤出选定时间范围内的数据
            mask = (self.time >= start_time) & (self.time <= end_time)
            plt.plot(self.time[mask], self.signal[mask], '-', label=self.text[self.language]['signal'])

            # 为选定的周期添加窗口
            for i in range(start_cycle, end_cycle):
                self._plot_cycle_windows(i, i==start_cycle)  # 仅将第一个周期包含在图例中

        plt.xlabel(self.text[self.language]['time_s'])
        plt.ylabel(self.text[self.language]['signal'])

        if plot_full_signal:
            plt.title(self.text[self.language]['full_signal_with_windows'])
        else:
            plt.title(self.text[self.language]['signal_with_windows_cycles'].format(start_cycle+1, start_cycle+actual_cycles))

        plt.legend()
        plt.grid(True)
        plt.show()

    def _plot_cycle_windows(self, cycle_index, include_in_legend=False):
        """
        在信号图上为指定周期绘制开启和关闭窗口
        
        参数:
        -----
        cycle_index : int
            要绘制窗口的周期索引(从0开始)
        include_in_legend : bool, optional
            是否将此周期的窗口包含在图例中
        """
        cycle_start_time = self.time[0] + cycle_index * self.period
        
        # 计算窗口时间
        on_window_start = cycle_start_time + self.window_on_offset
        on_window_end = on_window_start + self.window_on_size
        
        off_window_start = cycle_start_time + self.window_off_offset
        off_window_end = off_window_start + self.window_off_size
        
        # 绘制窗口
        if include_in_legend:
            plt.axvspan(on_window_start, on_window_end, alpha=0.2, color='green', label=f'{self.text[self.language]["on_window"]}')
            plt.axvspan(off_window_start, off_window_end, alpha=0.2, color='red', label=f'{self.text[self.language]["off_window"]}')
        else:
            plt.axvspan(on_window_start, on_window_end, alpha=0.2, color='green')
            plt.axvspan(off_window_start, off_window_end, alpha=0.2, color='red')

    def plot_all_fits(self, start_cycle=0, num_cycles=None, figsize=(15, 10)):
        """
        绘制所有或选定周期的拟合结果
        
        参数:
        -----
        start_cycle : int, optional
            要绘制的第一个周期的索引(从0开始)。默认为0。
        num_cycles : int, optional
            要绘制的周期数。如果为None，则绘制从start_cycle开始的所有周期
            (限制为10，以避免图形过大)。
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸。
        """
        if not self.cycle_results:
            print(self.text[self.language]['no_results'])
            return

        # 验证start_cycle
        if start_cycle < 0 or start_cycle >= len(self.cycle_results):
            print(f"{self.text[self.language]['invalid_start_cycle']}: {start_cycle}. {self.text[self.language]['must_be_between']} 0 and {len(self.cycle_results)-1}")
            return

        # 计算要绘制多少个周期
        cycles_remaining = len(self.cycle_results) - start_cycle

        if num_cycles is None:
            # 如果为None，绘制所有剩余周期(限制为10)
            num_cycles = min(cycles_remaining, 10)
        else:
            # 限制为可用周期
            num_cycles = min(num_cycles, cycles_remaining)

        end_cycle = start_cycle + num_cycles

        # 计算子图网格的行和列
        n_cols = 2  # 开和关在单独的列中
        n_rows = num_cycles

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # 处理单行的情况
        if n_rows == 1:
            axes = np.array([axes])

        for i in range(num_cycles):
            cycle_idx = start_cycle + i
            actual_cycle_num = self.cycle_results[cycle_idx]['cycle']  # 使用实际的周期号

            # 获取周期结果
            result = self.cycle_results[cycle_idx]
            fitter = result['fitter']
            was_refitted = result.get('was_refitted', False)

            # 绘制开启拟合
            ax_on = axes[i, 0]

            mask_on = (self.time >= fitter.t_on_idx[0]) & (self.time <= fitter.t_on_idx[1])
            t_on = self.time[mask_on]
            s_on = self.signal[mask_on]

            ax_on.plot(t_on, s_on, 'o', label=self.text[self.language]['data'])
            t_fit = np.linspace(t_on[0], t_on[-1], 100)
            ax_on.plot(t_fit, fitter.exp_rise(t_fit - t_fit[0], *fitter.tau_on_popt), '-', label=self.text[self.language]['fit'])

            # 添加标题和重新拟合信息
            title = self.text[self.language]['cycle_on_fit'].format(actual_cycle_num, fitter.get_tau_on(), fitter.tau_on_r_squared)
            if was_refitted and result['refit_info'] and 'on' in result['refit_info']['refit_types']:
                title += self.text[self.language]['refitted']
            ax_on.set_title(title)

            ax_on.set_xlabel(self.text[self.language]['time_s'])
            ax_on.set_ylabel(self.text[self.language]['signal'])
            ax_on.legend()
            ax_on.grid(True)

            # 绘制关闭拟合
            ax_off = axes[i, 1]

            mask_off = (self.time >= fitter.t_off_idx[0]) & (self.time <= fitter.t_off_idx[1])
            t_off = self.time[mask_off]
            s_off = self.signal[mask_off]

            ax_off.plot(t_off, s_off, 'o', label=self.text[self.language]['data'])
            t_fit = np.linspace(t_off[0], t_off[-1], 100)
            ax_off.plot(t_fit, fitter.exp_decay(t_fit - t_fit[0], *fitter.tau_off_popt), '-', label=self.text[self.language]['fit'])

            # 添加标题和重新拟合信息
            title = self.text[self.language]['cycle_off_fit'].format(actual_cycle_num, fitter.get_tau_off(), fitter.tau_off_r_squared)
            if was_refitted and result['refit_info'] and 'off' in result['refit_info']['refit_types']:
                title += self.text[self.language]['refitted']
            ax_off.set_title(title)

            ax_off.set_xlabel(self.text[self.language]['time_s'])
            ax_off.set_ylabel(self.text[self.language]['signal'])
            ax_off.legend()
            ax_off.grid(True)

        plt.suptitle(self.text[self.language]['fit_results_for_cycles'].format(start_cycle+1, end_cycle), fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
