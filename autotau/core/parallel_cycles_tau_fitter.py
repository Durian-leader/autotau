import numpy as np
import concurrent.futures
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from .tau_fitter import TauFitter


class ParallelCyclesTauFitter:
    """
    手动指定窗口参数，并行拟合多个周期信号的tau值

    该类接受预定义的窗口参数，使用多进程并行处理所有周期的拟合任务
    """

    def __init__(self, time, signal, period, sample_rate, window_on_offset, window_on_size,
                 window_off_offset, window_off_size, normalize=False, language='en',
                 show_progress=False, max_workers=None):
        """
        初始化ParallelCyclesTauFitter类

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
        window_on_offset : float
            开启窗口相对于周期起始的偏移量(s)
        window_on_size : float
            开启窗口大小(s)
        window_off_offset : float
            关闭窗口相对于周期起始的偏移量(s)
        window_off_size : float
            关闭窗口大小(s)
        normalize : bool, optional
            是否归一化信号(默认: False)
        language : str, optional
            界面语言('cn'或'en', 默认: 'en')
        show_progress : bool, optional
            是否显示进度条
        max_workers : int, optional
            最大工作进程数，默认为None，表示使用系统CPU核心数
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.window_on_offset = window_on_offset
        self.window_on_size = window_on_size
        self.window_off_offset = window_off_offset
        self.window_off_size = window_off_size
        self.normalize = normalize
        self.language = language
        self.show_progress = show_progress
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()

        # 结果存储
        self.cycle_results = []

        # 语言字典
        self.text = {
            'cn': {
                'no_results': '没有可用的周期结果。请先运行fit_all_cycles()。',
                'tau_on': 'Tau On',
                'tau_off': 'Tau Off',
                'cycle': '周期',
                'tau_s': 'Tau (s)',
                'tau_values_per_cycle': '每个周期的Tau值',
                'invalid_start_cycle': '无效的start_cycle',
                'must_be_between': '必须介于0和',
                'data': '数据',
                'fit': '拟合',
                'cycle_on_fit': '周期{} - 开启拟合 (τ = {:.5f} s, R² = {:.3f})',
                'time_s': '时间 (s)',
                'signal': '信号',
                'cycle_off_fit': '周期{} - 关闭拟合 (τ = {:.5f} s, R² = {:.3f})',
                'fit_results_for_cycles': '周期{}-{}的拟合结果',
                'r_squared_on': 'R² On',
                'r_squared_off': 'R² Off',
                'threshold': '阈值',
                'r_squared_values_per_cycle': '每个周期的R²值',
                'r_squared': 'R²',
                'full_signal_with_windows': '完整信号与开启和关闭窗口',
                'signal_with_windows_cycles': '信号与开启和关闭窗口(周期{}至{})',
                'on_window': '开启窗口',
                'off_window': '关闭窗口',
                'warning_insufficient_data': '警告: 周期{}的数据点不足。跳过。',
                'start_cycle_exceeds_available': '起始周期{}超过可用周期{}',
                'tau_on_y_label': 'Tau On (s)',
                'tau_off_y_label': 'Tau Off (s)'
            },
            'en': {
                'no_results': 'No cycle results available. Please run fit_all_cycles() first.',
                'tau_on': 'Tau On',
                'tau_off': 'Tau Off',
                'cycle': 'Cycle',
                'tau_s': 'Tau (s)',
                'tau_values_per_cycle': 'Tau Values per Cycle',
                'invalid_start_cycle': 'Invalid start_cycle',
                'must_be_between': 'must be between 0 and',
                'data': 'Data',
                'fit': 'Fit',
                'cycle_on_fit': 'Cycle {} - On Fit (τ = {:.5f} s, R² = {:.3f})',
                'time_s': 'Time (s)',
                'signal': 'Signal',
                'cycle_off_fit': 'Cycle {} - Off Fit (τ = {:.5f} s, R² = {:.3f})',
                'fit_results_for_cycles': 'Fit Results for Cycles {}-{}',
                'r_squared_on': 'R² On',
                'r_squared_off': 'R² Off',
                'threshold': 'Threshold',
                'r_squared_values_per_cycle': 'R² Values per Cycle',
                'r_squared': 'R²',
                'full_signal_with_windows': 'Full Signal with On and Off Windows',
                'signal_with_windows_cycles': 'Signal with On and Off Windows (Cycles {} to {})',
                'on_window': 'On Window',
                'off_window': 'Off Window',
                'warning_insufficient_data': 'Warning: Insufficient data points for cycle {}. Skipping.',
                'start_cycle_exceeds_available': 'Start cycle {} exceeds available cycles {}',
                'tau_on_y_label': 'Tau On (s)',
                'tau_off_y_label': 'Tau Off (s)'
            }
        }

    def _process_cycle(self, cycle_data):
        """
        处理单个周期的拟合任务，被并行调用

        参数:
        -----
        cycle_data : dict
            周期数据，包含所有必要信息

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
        cycle_start_time = cycle_data['cycle_start_time']

        # 创建窗口的掩码
        on_mask = (time >= on_window_start) & (time <= on_window_end)
        off_mask = (time >= off_window_start) & (time <= off_window_end)

        # 检查是否有足够的数据点
        if np.sum(on_mask) < 3 or np.sum(off_mask) < 3:
            return {
                'status': 'skipped',
                'message': f"Cycle {i+1}: Insufficient data points."
            }

        # 为此周期创建一个TauFitter
        tau_fitter = TauFitter(
            time,
            signal,
            t_on_idx=[on_window_start, on_window_end],
            t_off_idx=[off_window_start, off_window_end],
            normalize=cycle_data['normalize'],
            language=cycle_data['language']
        )

        # 用指定的插值设置拟合数据
        tau_fitter.fit_tau_on(interp=interp, points_after_interp=points_after_interp)
        tau_fitter.fit_tau_off(interp=interp, points_after_interp=points_after_interp)

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
            'fitter': tau_fitter
        }

        return result

    def fit_all_cycles(self, interp=True, points_after_interp=100):
        """
        使用并行处理拟合所有周期的tau值

        参数:
        -----
        interp : bool, optional
            是否在拟合过程中使用插值(默认: True)
        points_after_interp : int, optional
            插值后的点数(默认: 100)

        返回:
        -----
        list of dict
            包含每个周期拟合结果的列表
        """
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
                'normalize': self.normalize,
                'language': self.language
            }
            cycles_data.append(cycle_data)

        # 重置结果存储
        self.cycle_results = []

        # 使用ProcessPoolExecutor并行处理所有周期
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            if self.show_progress:
                # 使用tqdm显示进度
                futures = list(tqdm(
                    executor.map(self._process_cycle, cycles_data),
                    total=len(cycles_data),
                    desc="Parallel fitting cycles"
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

        # 按周期排序结果
        self.cycle_results.sort(key=lambda x: x['cycle'])

        return self.cycle_results

    def plot_cycle_results(self, figsize=(10, 6), dual_y_axis=True):
        """
        绘制所有周期的tau值

        参数:
        -----
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸
        dual_y_axis : bool, optional
            是否使用双y轴, 默认为True
        """
        if not self.cycle_results:
            print(self.text[self.language]['no_results'])
            return

        cycles = [res['cycle'] for res in self.cycle_results]
        tau_on_values = [res['tau_on'] for res in self.cycle_results]
        tau_off_values = [res['tau_off'] for res in self.cycle_results]

        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.set_xlabel(self.text[self.language]['cycle'])
        ax1.set_ylabel(self.text[self.language]['tau_on_y_label'], color='blue')
        ax1.plot(cycles, tau_on_values, 'o-', label=self.text[self.language]['tau_on'], color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(axis='x')

        if dual_y_axis:
            ax2 = ax1.twinx()
            ax2.set_ylabel(self.text[self.language]['tau_off_y_label'], color='red')
            ax2.plot(cycles, tau_off_values, 'o-', label=self.text[self.language]['tau_off'], color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.grid(False)
            fig.tight_layout()
        else:
            ax1.plot(cycles, tau_off_values, 'o-', label=self.text[self.language]['tau_off'], color='red')
            ax1.set_ylabel(self.text[self.language]['tau_s'])
            ax1.grid(True)

        plt.title(self.text[self.language]['tau_values_per_cycle'])
        lines, labels = ax1.get_legend_handles_labels()
        if dual_y_axis:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax1.legend(lines, labels, loc='best')

        plt.show()

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
            actual_cycle_num = self.cycle_results[cycle_idx]['cycle']

            # 获取周期结果
            result = self.cycle_results[cycle_idx]
            fitter = result['fitter']

            # 绘制开启拟合
            ax_on = axes[i, 0]

            mask_on = (self.time >= fitter.t_on_idx[0]) & (self.time <= fitter.t_on_idx[1])
            t_on = self.time[mask_on]
            s_on = self.signal[mask_on]

            ax_on.plot(t_on, s_on, 'o', label=self.text[self.language]['data'])
            t_fit = np.linspace(t_on[0], t_on[-1], 100)
            ax_on.plot(t_fit, fitter.exp_rise(t_fit - t_fit[0], *fitter.tau_on_popt), '-', label=self.text[self.language]['fit'])

            title = self.text[self.language]['cycle_on_fit'].format(actual_cycle_num, fitter.get_tau_on(), fitter.tau_on_r_squared)
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

            title = self.text[self.language]['cycle_off_fit'].format(actual_cycle_num, fitter.get_tau_off(), fitter.tau_off_r_squared)
            ax_off.set_title(title)

            ax_off.set_xlabel(self.text[self.language]['time_s'])
            ax_off.set_ylabel(self.text[self.language]['signal'])
            ax_off.legend()
            ax_off.grid(True)

        plt.suptitle(self.text[self.language]['fit_results_for_cycles'].format(start_cycle+1, end_cycle), fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    def get_summary_data(self):
        """
        返回拟合结果的摘要DataFrame

        返回:
        -----
        pandas.DataFrame
            包含周期号、开始时间、tau值和R平方值的DataFrame
        """
        if not self.cycle_results:
            print(self.text[self.language]['no_results'])
            return None

        data = {
            'cycle': [],
            'cycle_start_time': [],
            'tau_on': [],
            'tau_off': [],
            'r_squared_on': [],
            'r_squared_off': [],
            'r_squared_adj_on': [],
            'r_squared_adj_off': []
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

        return pd.DataFrame(data)

    def plot_r_squared_values(self, figsize=(10, 6), threshold=None):
        """
        绘制所有周期的R²值

        参数:
        -----
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸
        threshold : float, optional
            R²阈值线，如果提供则在图中绘制水平线
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

        # 在阈值处添加水平线
        if threshold is not None:
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
        # 创建图形
        plt.figure(figsize=figsize)

        if plot_full_signal:
            # 绘制整个信号
            plt.plot(self.time, self.signal, '-', label=self.text[self.language]['signal'])

            # 计算数据中有多少个完整周期
            total_cycles = int((self.time[-1] - self.time[0]) / self.period)

            # 为数据中的每个完整周期添加窗口
            for i in range(total_cycles):
                self._plot_cycle_windows(i, i==0)
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
                self._plot_cycle_windows(i, i==start_cycle)

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
        绘制特定周期窗口的辅助方法

        参数:
        -----
        cycle_index : int
            要绘制的周期索引
        include_in_legend : bool
            是否将此周期的窗口包含在图例中
        """
        cycle_start_time = self.time[0] + cycle_index * self.period

        # 开启窗口
        on_window_start = cycle_start_time + self.window_on_offset
        on_window_end = on_window_start + self.window_on_size

        # 关闭窗口
        off_window_start = cycle_start_time + self.window_off_offset
        off_window_end = off_window_start + self.window_off_size

        # 突出显示开启窗口
        plt.axvspan(on_window_start, on_window_end, alpha=0.2, color='green',
                    label=self.text[self.language]['on_window'] if include_in_legend else "")

        # 突出显示关闭窗口
        plt.axvspan(off_window_start, off_window_end, alpha=0.2, color='red',
                    label=self.text[self.language]['off_window'] if include_in_legend else "")
