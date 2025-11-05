import numpy as np
import concurrent.futures
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import repeat
from .tau_fitter import TauFitter
from .parallel import (
    _get_mp_context,
    _auto_chunk_size,
    _chunk_sequence,
    _init_cycle_worker,
    _process_cycles_chunk,
)


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
        # 对偏移量取模，确保在 [0, period) 范围内，避免窗口位置偏移
        self.window_on_offset = window_on_offset % period
        self.window_on_size = window_on_size
        self.window_off_offset = window_off_offset % period
        self.window_off_size = window_off_size
        self.normalize = normalize
        self.language = language
        self.show_progress = show_progress
        self.max_workers = max_workers if max_workers else multiprocessing.cpu_count()
        # 使用与并行模块一致的多进程上下文（优先 fork，有利于性能）
        self._mp_context = _get_mp_context()

        # 结果存储（DataFrame 或记录列表）。默认使用 DataFrame。
        self.cycle_results = None

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

    def fit_all_cycles(self, interp=True, points_after_interp=100, return_format='dataframe'):
        """
        使用分块并行处理拟合所有周期的tau值（每个进程处理一组周期，降低调度与序列化开销）

        参数:
        -----
        interp : bool, optional
            是否在拟合过程中使用插值(默认: True)
        points_after_interp : int, optional
            插值后的点数(默认: 100)

        返回:
        -----
        - return_format='dataframe'（默认）：返回 pandas.DataFrame，列包括：
          ['cycle','cycle_start_time','tau_on','tau_off',
           'r_squared_on','r_squared_off','r_squared_adj_on','r_squared_adj_off',
           'window_on_start_time','window_on_end_time','window_off_start_time','window_off_end_time',
           'window_on_start_idx','window_on_end_idx','window_off_start_idx','window_off_end_idx']
        - return_format='records'：返回与旧版兼容的记录列表（不含每周期的 'fitter' 对象）。
        """
        # 计算完整周期数量
        total_time = float(self.time[-1] - self.time[0])
        num_cycles = int(total_time / float(self.period))

        if num_cycles <= 0:
            empty_df = pd.DataFrame(columns=[
                'cycle','cycle_start_time','tau_on','tau_off',
                'r_squared_on','r_squared_off','r_squared_adj_on','r_squared_adj_off',
                'window_on_start_time','window_on_end_time','window_off_start_time','window_off_end_time',
                'window_on_start_idx','window_on_end_idx','window_off_start_idx','window_off_end_idx'
            ])
            self.cycle_results = empty_df
            return empty_df if return_format == 'dataframe' else []

        # 计算合理的分块大小（与 parallel.py 保持一致的策略）
        chunk_size = _auto_chunk_size(num_cycles, self.max_workers)
        chunks = list(_chunk_sequence(num_cycles, chunk_size))

        # 轻量列容器，避免构建巨型嵌套对象
        col_cycle = []
        col_cycle_start_time = []
        col_tau_on = []
        col_tau_off = []
        col_r2_on = []
        col_r2_off = []
        col_r2a_on = []
        col_r2a_off = []
        col_on_st = []
        col_on_et = []
        col_off_st = []
        col_off_et = []
        col_on_si = []
        col_on_ei = []
        col_off_si = []
        col_off_ei = []

        # 初始化每个进程的只读共享状态，避免为每个周期重复序列化大数组
        base_time = float(self.time[0])
        init_args = (
            self.time,
            self.signal,
            float(self.sample_rate),
            float(self.period),
            self.language,
            bool(self.normalize),
            float(self.window_on_offset),
            float(self.window_on_size),
            float(self.window_off_offset),
            float(self.window_off_size),
            {},                 # auto_tau_params（此类不启用自动重拟合）
            base_time,
        )

        # 使用 ProcessPoolExecutor + initializer, 每个任务是一组周期索引
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=self._mp_context,
            initializer=_init_cycle_worker,
            initargs=init_args,
        ) as executor:
            mapped = executor.map(
                _process_cycles_chunk,
                chunks,
                repeat(bool(interp)),
                repeat(int(points_after_interp)),
                repeat(-1.0),  # r_squared_threshold（给负值可有效关闭重拟合）
            )

            # 进度包装
            if self.show_progress:
                mapped = tqdm(mapped, total=len(chunks), desc="Parallel fitting cycles (chunked)")

            # 累积轻量列数据
            for chunk_results in mapped:
                for r in chunk_results:
                    if r.get('status') == 'skipped':
                        if self.show_progress and r.get('message'):
                            print(r['message'])
                        continue
                    col_cycle.append(int(r['cycle']))
                    col_cycle_start_time.append(float(r.get('cycle_start_time', np.nan)))
                    col_tau_on.append(r.get('tau_on'))
                    col_tau_off.append(r.get('tau_off'))
                    col_r2_on.append(r.get('tau_on_r_squared'))
                    col_r2_off.append(r.get('tau_off_r_squared'))
                    col_r2a_on.append(r.get('tau_on_r_squared_adj'))
                    col_r2a_off.append(r.get('tau_off_r_squared_adj'))
                    col_on_st.append(r.get('window_on_start_time'))
                    col_on_et.append(r.get('window_on_end_time'))
                    col_off_st.append(r.get('window_off_start_time'))
                    col_off_et.append(r.get('window_off_end_time'))
                    col_on_si.append(r.get('window_on_start_idx'))
                    col_on_ei.append(r.get('window_on_end_idx'))
                    col_off_si.append(r.get('window_off_start_idx'))
                    col_off_ei.append(r.get('window_off_end_idx'))

        df = pd.DataFrame({
            'cycle': col_cycle,
            'cycle_start_time': col_cycle_start_time,
            'tau_on': col_tau_on,
            'tau_off': col_tau_off,
            'r_squared_on': col_r2_on,
            'r_squared_off': col_r2_off,
            'r_squared_adj_on': col_r2a_on,
            'r_squared_adj_off': col_r2a_off,
            'window_on_start_time': col_on_st,
            'window_on_end_time': col_on_et,
            'window_off_start_time': col_off_st,
            'window_off_end_time': col_off_et,
            'window_on_start_idx': col_on_si,
            'window_on_end_idx': col_on_ei,
            'window_off_start_idx': col_off_si,
            'window_off_end_idx': col_off_ei,
        })
        df.sort_values('cycle', inplace=True)
        df.reset_index(drop=True, inplace=True)

        self.cycle_results = df
        if return_format == 'dataframe':
            return df
        elif return_format == 'records':
            return df.to_dict(orient='records')
        else:
            raise ValueError("return_format must be 'dataframe' or 'records'")

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
        if self.cycle_results is None or (isinstance(self.cycle_results, list) and not self.cycle_results) or (hasattr(self.cycle_results, 'empty') and self.cycle_results.empty):
            print(self.text[self.language]['no_results'])
            return
        if isinstance(self.cycle_results, pd.DataFrame):
            cycles = self.cycle_results['cycle'].to_numpy()
            tau_on_values = self.cycle_results['tau_on'].to_numpy()
            tau_off_values = self.cycle_results['tau_off'].to_numpy()
        else:
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
        if self.cycle_results is None or (isinstance(self.cycle_results, list) and not self.cycle_results) or (hasattr(self.cycle_results, 'empty') and self.cycle_results.empty):
            print(self.text[self.language]['no_results'])
            return

        # 验证start_cycle
        total_n = len(self.cycle_results) if isinstance(self.cycle_results, list) else len(self.cycle_results.index)
        if start_cycle < 0 or start_cycle >= total_n:
            print(f"{self.text[self.language]['invalid_start_cycle']}: {start_cycle}. {self.text[self.language]['must_be_between']} 0 and {total_n-1}")
            return

        # 计算要绘制多少个周期
        cycles_remaining = total_n - start_cycle

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
            if isinstance(self.cycle_results, pd.DataFrame):
                row = self.cycle_results.iloc[cycle_idx]
                actual_cycle_num = int(row['cycle'])
                fitter = self._build_fitter_from_row(row, interp=True, points=100)
                if fitter is None or fitter.tau_on_popt is None or fitter.tau_off_popt is None:
                    axes[i, 0].axis('off')
                    axes[i, 1].axis('off')
                    continue
            else:
                actual_cycle_num = self.cycle_results[cycle_idx]['cycle']
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

            title = self.text[self.language]['cycle_on_fit'].format(actual_cycle_num, fitter.get_tau_on(), getattr(fitter, 'tau_on_r_squared', None))
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

            title = self.text[self.language]['cycle_off_fit'].format(actual_cycle_num, fitter.get_tau_off(), getattr(fitter, 'tau_off_r_squared', None))
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
        if self.cycle_results is None or (isinstance(self.cycle_results, list) and not self.cycle_results) or (hasattr(self.cycle_results, 'empty') and self.cycle_results.empty):
            print(self.text[self.language]['no_results'])
            return None

        if isinstance(self.cycle_results, pd.DataFrame):
            cols = ['cycle','cycle_start_time','tau_on','tau_off','r_squared_on','r_squared_off','r_squared_adj_on','r_squared_adj_off']
            existing = [c for c in cols if c in self.cycle_results.columns]
            return self.cycle_results[existing].copy()
        else:
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
                data['r_squared_on'].append(res.get('tau_on_r_squared') or res.get('r_squared_on'))
                data['r_squared_off'].append(res.get('tau_off_r_squared') or res.get('r_squared_off'))
                data['r_squared_adj_on'].append(res.get('tau_on_r_squared_adj') or res.get('r_squared_adj_on'))
                data['r_squared_adj_off'].append(res.get('tau_off_r_squared_adj') or res.get('r_squared_adj_off'))
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
        if self.cycle_results is None or (isinstance(self.cycle_results, list) and not self.cycle_results) or (hasattr(self.cycle_results, 'empty') and self.cycle_results.empty):
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

    def _build_fitter_from_row(self, row, interp=True, points=100):
        """根据 DataFrame 行信息按需构建 TauFitter（仅用于小规模可视化）。

        如果行中不包含 popt，将在对应窗口内执行一次拟合。
        """
        try:
            on_start = float(row['window_on_start_time'])
            on_end = float(row['window_on_end_time'])
            off_start = float(row['window_off_start_time'])
            off_end = float(row['window_off_end_time'])
        except Exception:
            return None

        fitter = TauFitter(
            self.time,
            self.signal,
            t_on_idx=[on_start, on_end],
            t_off_idx=[off_start, off_end],
            normalize=self.normalize,
            language=self.language,
        )
        # 拟合（仅在可视化时小量调用）
        fitter.fit_tau_on(interp=interp, points_after_interp=points)
        fitter.fit_tau_off(interp=interp, points_after_interp=points)
        return fitter
