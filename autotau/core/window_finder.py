import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from .tau_fitter import TauFitter


class WindowFinder:
    """
    自动搜索最佳拟合窗口的类，但不进行拟合

    使用并行处理加速窗口搜索，返回最佳窗口参数供后续使用
    """

    def __init__(self, time, signal, sample_step, period, window_scalar_min=1/5, window_scalar_max=1/3,
                 window_points_step=10, window_start_idx_step=1, normalize=False, language='en',
                 show_progress=False, max_workers=None):
        """
        初始化WindowFinder

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

        # 最佳窗口参数
        self.best_on_window = None
        self.best_off_window = None

        # 语言字典
        self.text = {
            'cn': {
                'searching': '正在搜索最佳窗口...',
                'found': '找到最佳窗口参数',
                'on_window': '开启窗口',
                'off_window': '关闭窗口',
                'offset': '偏移量',
                'size': '大小',
                'r_squared': 'R²'
            },
            'en': {
                'searching': 'Searching for best windows...',
                'found': 'Found best window parameters',
                'on_window': 'On Window',
                'off_window': 'Off Window',
                'offset': 'Offset',
                'size': 'Size',
                'r_squared': 'R²'
            }
        }

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
            窗口评估结果字典
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
                    'r_squared': tau_fitter.tau_on_r_squared if tau_fitter.tau_on_r_squared is not None else 0,
                    'window_size': window_points * self.sample_step,
                    'window_start_time': window_time[0],
                    'window_end_time': window_time[-1],
                },
                'off': {
                    'r_squared_adj': tau_fitter.tau_off_r_squared_adj if tau_fitter.tau_off_r_squared_adj is not None else 0,
                    'r_squared': tau_fitter.tau_off_r_squared if tau_fitter.tau_off_r_squared is not None else 0,
                    'window_size': window_points * self.sample_step,
                    'window_start_time': window_time[0],
                    'window_end_time': window_time[-1],
                }
            }
            return result
        except Exception as e:
            # 拟合失败，返回None
            return None

    def _process_window_wrapper(self, window_params, interp, points_after_interp):
        """
        包装方法用于并行处理
        """
        return self._process_window(window_params, interp, points_after_interp)

    def find_best_windows(self, interp=True, points_after_interp=100):
        """
        搜索最佳的on和off窗口参数

        参数:
        -----
        interp : bool, optional
            是否使用插值，默认为True
        points_after_interp : int, optional
            插值后的点数，默认为100

        返回:
        -----
        dict
            包含最佳窗口参数的字典，格式为:
            {
                'on': {
                    'offset': float,  # 相对于数据起始时间的偏移量
                    'size': float,    # 窗口大小
                    'start_time': float,
                    'end_time': float,
                    'r_squared': float,
                    'r_squared_adj': float
                },
                'off': {
                    'offset': float,
                    'size': float,
                    'start_time': float,
                    'end_time': float,
                    'r_squared': float,
                    'r_squared_adj': float
                }
            }
        """
        if self.show_progress:
            print(self.text[self.language]['searching'])

        # 初始化最佳窗口结果
        best_on = {
            'r_squared_adj': 0,
            'r_squared': 0,
            'window_size': 0,
            'window_start_time': 0,
            'window_end_time': 0
        }

        best_off = {
            'r_squared_adj': 0,
            'r_squared': 0,
            'window_size': 0,
            'window_start_time': 0,
            'window_end_time': 0
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
            print(f"Total windows to evaluate: {total_tasks}")

        # 使用ProcessPoolExecutor并行处理所有窗口
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
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
                    desc="Window search progress"
                ))
            else:
                # 不显示进度
                results = list(executor.map(
                    process_function,
                    window_params_list,
                    [interp] * len(window_params_list),
                    [points_after_interp] * len(window_params_list)
                ))

        # 处理结果，找到最佳窗口
        for result in results:
            if result is None:
                continue

            # 检查on窗口结果
            on_result = result['on']
            if on_result['r_squared_adj'] > best_on['r_squared_adj']:
                best_on = on_result

            # 检查off窗口结果
            off_result = result['off']
            if off_result['r_squared_adj'] > best_off['r_squared_adj']:
                best_off = off_result

        # 计算相对于数据起始时间的偏移量
        time_start = self.time[0]

        self.best_on_window = {
            'offset': best_on['window_start_time'] - time_start,
            'size': best_on['window_size'],
            'start_time': best_on['window_start_time'],
            'end_time': best_on['window_end_time'],
            'r_squared': best_on['r_squared'],
            'r_squared_adj': best_on['r_squared_adj']
        }

        self.best_off_window = {
            'offset': best_off['window_start_time'] - time_start,
            'size': best_off['window_size'],
            'start_time': best_off['window_start_time'],
            'end_time': best_off['window_end_time'],
            'r_squared': best_off['r_squared'],
            'r_squared_adj': best_off['r_squared_adj']
        }

        if self.show_progress:
            print(f"\n{self.text[self.language]['found']}:")
            print(f"{self.text[self.language]['on_window']}:")
            print(f"  {self.text[self.language]['offset']}: {self.best_on_window['offset']:.4f} s")
            print(f"  {self.text[self.language]['size']}: {self.best_on_window['size']:.4f} s")
            print(f"  {self.text[self.language]['r_squared']}: {self.best_on_window['r_squared']:.4f}")
            print(f"{self.text[self.language]['off_window']}:")
            print(f"  {self.text[self.language]['offset']}: {self.best_off_window['offset']:.4f} s")
            print(f"  {self.text[self.language]['size']}: {self.best_off_window['size']:.4f} s")
            print(f"  {self.text[self.language]['r_squared']}: {self.best_off_window['r_squared']:.4f}")

        return {
            'on': self.best_on_window,
            'off': self.best_off_window
        }

    def get_window_params(self):
        """
        获取窗口参数（在调用find_best_windows之后使用）

        返回:
        -----
        dict
            窗口参数字典
        """
        if self.best_on_window is None or self.best_off_window is None:
            raise ValueError("Please run find_best_windows() first")

        return {
            'on': self.best_on_window,
            'off': self.best_off_window
        }
