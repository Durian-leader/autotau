import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .tau_fitter import TauFitter
from .auto_tau_fitter import AutoTauFitter

class CyclesAutoTauFitter:
    """
    自动拟合多个周期信号的tau值
    
    该类使用AutoTauFitter找到前两个周期的最佳拟合窗口，然后根据周期长度将这些窗口应用于后续周期。
    """

    def __init__(self, time, signal, period, sample_rate, **kwargs):
        """
        初始化CyclesAutoTauFitter类
        
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
            window_scalar_min : float, optional
                窗口大小最小值占周期的比例(默认: 1/5)
            window_scalar_max : float, optional
                窗口大小最大值占周期的比例(默认: 1/3)
            window_points_step : int, optional
                窗口点数搜索步长(默认: 10)
            window_start_idx_step : int, optional
                窗口起始位置搜索步长(默认: 1)
            normalize : bool, optional
                是否归一化信号(默认: False)
            language : str, optional
                界面语言('cn'或'en', 默认: 'cn')
            show_progress : bool, optional
                是否显示进度条(默认: False)
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.auto_tau_fitter_params = kwargs

        # 结果存储
        self.cycle_results = []
        self.refitted_cycles = []  # 存储需要重新拟合的周期信息
        self.last_r_squared_threshold = 0.95  # 存储上次使用的阈值

        self.initial_auto_fitter = None
        self.window_on_offset = None  # 开窗口的时间偏移
        self.window_off_offset = None  # 关窗口的时间偏移
        self.window_on_size = None  # 开窗口的大小
        self.window_off_size = None  # 关窗口的大小

    def find_best_windows(self, interp=True, points_after_interp=100):
        """
        使用前两个周期找到开启和关闭过程的最佳窗口
        
        参数:
        -----
        interp : bool, optional
            是否在拟合过程中使用插值(默认: True)
        points_after_interp : int, optional
            插值后的点数(默认: 100)
            
        返回:
        -----
        AutoTauFitter
            用于找到最佳窗口的AutoTauFitter实例
        """
        # 提取前两个周期
        two_period_mask = (self.time <= self.time[0] + 2 * self.period)
        time_subset = self.time[two_period_mask]
        signal_subset = self.signal[two_period_mask]

        # 对前两个周期使用AutoTauFitter
        auto_fitter = AutoTauFitter(
            time_subset,
            signal_subset,
            sample_step=1/self.sample_rate,
            period=self.period,
            **self.auto_tau_fitter_params
        )
        auto_fitter.fit_tau_on_and_off(interp=interp, points_after_interp=points_after_interp)

        # 存储最佳窗口参数（取模确保偏移量在单个周期范围内）
        self.window_on_offset = (auto_fitter.best_tau_on_window_start_time - self.time[0]) % self.period
        self.window_off_offset = (auto_fitter.best_tau_off_window_start_time - self.time[0]) % self.period
        self.window_on_size = auto_fitter.best_tau_on_window_size
        self.window_off_size = auto_fitter.best_tau_off_window_size
        self.initial_auto_fitter = auto_fitter

        return auto_fitter

    def fit_all_cycles(self, interp=True, points_after_interp=100, r_squared_threshold=0.95):
        """
        将find_best_windows()找到的窗口应用于数据中的所有周期
        
        该方法:
        1. 使用前两个周期找到开启/关闭过程的最佳窗口(如果尚未完成)
        2. 计算数据中完整周期的数量
        3. 对于每个周期，通过根据周期移动窗口来应用窗口
        4. 使用TauFitter拟合每个周期的tau值
        5. 如果R²低于阈值，尝试为该特定周期找到更好的拟合
        6. 存储结果
        
        参数:
        -----
        interp : bool, optional
            是否在拟合过程中使用插值(默认: True)
        points_after_interp : int, optional
            插值后的点数(默认: 100)
        r_squared_threshold : float, optional
            R²阈值。如果低于此值，将尝试找到更好的拟合(默认: 0.95)
            
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

        self.cycle_results = []
        self.refitted_cycles = []  # 重置重新拟合的周期列表

        for i in range(num_cycles):
            cycle_start_time = self.time[0] + i * self.period

            # 计算此周期的窗口开始和结束时间
            on_window_start = cycle_start_time + self.window_on_offset
            on_window_end = on_window_start + self.window_on_size

            off_window_start = cycle_start_time + self.window_off_offset
            off_window_end = off_window_start + self.window_off_size

            # 创建窗口的掩码
            on_mask = (self.time >= on_window_start) & (self.time <= on_window_end)
            off_mask = (self.time >= off_window_start) & (self.time <= off_window_end)

            # 检查是否有足够的数据点
            if np.sum(on_mask) < 3 or np.sum(off_mask) < 3:
                print(f"警告: 周期{i+1}的数据点不足。跳过。")
                continue

            # 为此周期创建一个TauFitter
            tau_fitter = TauFitter(
                self.time,
                self.signal,
                t_on_idx=[on_window_start, on_window_end],
                t_off_idx=[off_window_start, off_window_end],
                **{k: v for k, v in self.auto_tau_fitter_params.items() if k in ['normalize', 'language']}
            )

            # 用指定的插值设置拟合数据
            tau_fitter.fit_tau_on(interp=interp, points_after_interp=points_after_interp)
            tau_fitter.fit_tau_off(interp=interp, points_after_interp=points_after_interp)

            # 检查拟合是否足够好(R² > 阈值)
            r_squared_on = tau_fitter.tau_on_r_squared
            r_squared_off = tau_fitter.tau_off_r_squared

            needs_refitting = False
            refit_type = []

            if r_squared_on < r_squared_threshold:
                needs_refitting = True
                refit_type.append('on')

            if r_squared_off < r_squared_threshold:
                needs_refitting = True
                refit_type.append('off')

            # 如果拟合质量不佳，尝试为此特定周期找到更好的窗口
            if needs_refitting:
                # 定义包含此周期加上一些余量的时间窗口
                # 我们将使用完整的周期长度作为窗口
                cycle_start = cycle_start_time
                cycle_end = cycle_start_time + self.period

                # 确保不超出边界
                cycle_start = max(cycle_start, self.time[0])
                cycle_end = min(cycle_end, self.time[-1])

                # 提取此周期的数据
                cycle_mask = (self.time >= cycle_start) & (self.time <= cycle_end)
                cycle_time = self.time[cycle_mask]
                cycle_signal = self.signal[cycle_mask]

                # 确保有足够的数据点
                if len(cycle_time) < 10:  # 最小值
                    print(f"警告: 重新拟合周期{i+1}的数据点不足。使用原始拟合。")
                else:
                    print(f"周期{i+1}拟合不佳(R² on: {r_squared_on:.3f}, off: {r_squared_off:.3f})。尝试寻找更好的拟合...")

                    # 使用AutoTauFitter为这个周期找到更好的窗口
                    cycle_auto_fitter = AutoTauFitter(
                        cycle_time,
                        cycle_signal,
                        sample_step=1/self.sample_rate,
                        period=self.period,
                        **self.auto_tau_fitter_params
                    )

                    cycle_auto_fitter.fit_tau_on_and_off(interp=interp, points_after_interp=points_after_interp)

                    # 记录重新拟合信息
                    refit_info = {
                        'cycle': i + 1,
                        'original_r_squared_on': r_squared_on,
                        'original_r_squared_off': r_squared_off,
                        'refit_types': refit_type,
                    }

                    # 检查并应用更好的开启拟合（如果需要）
                    if 'on' in refit_type:
                        new_r_squared_on = cycle_auto_fitter.best_tau_on_fitter.tau_on_r_squared if cycle_auto_fitter.best_tau_on_fitter else 0

                        if new_r_squared_on > r_squared_on:
                            print(f"  找到更好的开启过渡拟合: R²从{r_squared_on:.3f}提高到{new_r_squared_on:.3f}")
                            tau_fitter.tau_on_popt = cycle_auto_fitter.best_tau_on_fitter.tau_on_popt
                            tau_fitter.tau_on_pcov = cycle_auto_fitter.best_tau_on_fitter.tau_on_pcov
                            tau_fitter.tau_on_r_squared = new_r_squared_on
                            tau_fitter.tau_on_r_squared_adj = cycle_auto_fitter.best_tau_on_fitter.tau_on_r_squared_adj
                            refit_info['new_r_squared_on'] = new_r_squared_on
                        else:
                            print(f"  无法找到更好的开启过渡拟合。保留原始拟合。")
                            refit_info['new_r_squared_on'] = r_squared_on

                    # 检查并应用更好的关闭拟合（如果需要）
                    if 'off' in refit_type:
                        new_r_squared_off = cycle_auto_fitter.best_tau_off_fitter.tau_off_r_squared if cycle_auto_fitter.best_tau_off_fitter else 0

                        if new_r_squared_off > r_squared_off:
                            print(f"  找到更好的关闭过渡拟合: R²从{r_squared_off:.3f}提高到{new_r_squared_off:.3f}")
                            tau_fitter.tau_off_popt = cycle_auto_fitter.best_tau_off_fitter.tau_off_popt
                            tau_fitter.tau_off_pcov = cycle_auto_fitter.best_tau_off_fitter.tau_off_pcov
                            tau_fitter.tau_off_r_squared = new_r_squared_off
                            tau_fitter.tau_off_r_squared_adj = cycle_auto_fitter.best_tau_off_fitter.tau_off_r_squared_adj
                            refit_info['new_r_squared_off'] = new_r_squared_off
                        else:
                            print(f"  无法找到更好的关闭过渡拟合。保留原始拟合。")
                            refit_info['new_r_squared_off'] = r_squared_off

                    self.refitted_cycles.append(refit_info)

            # 存储结果
            cycle_result = {
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
                'was_refitted': needs_refitting
            }

            self.cycle_results.append(cycle_result)

        return self.cycle_results

    def plot_cycle_results(self, figsize=(10, 6)):
        """
        绘制所有周期的tau值
        
        参数:
        -----
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸
        """
        if not self.cycle_results:
            print("没有可用的周期结果。请先运行fit_all_cycles()。")
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
        plt.plot(cycles, tau_on_values, 'o-', label='Tau On', color='blue')
        plt.plot(cycles, tau_off_values, 'o-', label='Tau Off', color='red')

        # 突出显示重新拟合的周期
        if refitted_cycles:
            plt.scatter(refitted_cycles, refitted_tau_on, s=100, facecolors='none', edgecolors='blue',
                        linewidth=2, label='Refitted Tau On')
            plt.scatter(refitted_cycles, refitted_tau_off, s=100, facecolors='none', edgecolors='red',
                        linewidth=2, label='Refitted Tau Off')

        plt.xlabel('周期')
        plt.ylabel('Tau (s)')
        plt.title('每个周期的Tau值')
        plt.legend()
        plt.grid(True)
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
            print("没有可用的周期结果。请先运行fit_all_cycles()。")
            return

        # 验证start_cycle
        if start_cycle < 0 or start_cycle >= len(self.cycle_results):
            print(f"无效的start_cycle: {start_cycle}。必须介于0和{len(self.cycle_results)-1}之间")
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
            actual_cycle_num = cycle_idx + 1  # 显示时从1开始索引

            # 获取周期结果
            result = self.cycle_results[cycle_idx]
            fitter = result['fitter']
            was_refitted = result.get('was_refitted', False)

            # 绘制开启拟合
            ax_on = axes[i, 0]

            mask_on = (self.time >= fitter.t_on_idx[0]) & (self.time <= fitter.t_on_idx[1])
            t_on = self.time[mask_on]
            s_on = self.signal[mask_on]

            ax_on.plot(t_on, s_on, 'o', label='数据')
            t_fit = np.linspace(t_on[0], t_on[-1], 100)
            ax_on.plot(t_fit, fitter.exp_rise(t_fit - t_fit[0], *fitter.tau_on_popt), '-', label='拟合')

            # 添加标题和重新拟合信息
            title = f'周期{actual_cycle_num} - 开启拟合 (τ = {fitter.get_tau_on():.5f} s, R² = {fitter.tau_on_r_squared:.3f})'
            if was_refitted and 'on' in [info['refit_types'] for info in self.refitted_cycles if info['cycle'] == actual_cycle_num][0]:
                title += ' [已重新拟合]'
            ax_on.set_title(title)

            ax_on.set_xlabel('时间 (s)')
            ax_on.set_ylabel('信号')
            ax_on.legend()
            ax_on.grid(True)

            # 绘制关闭拟合
            ax_off = axes[i, 1]

            mask_off = (self.time >= fitter.t_off_idx[0]) & (self.time <= fitter.t_off_idx[1])
            t_off = self.time[mask_off]
            s_off = self.signal[mask_off]

            ax_off.plot(t_off, s_off, 'o', label='数据')
            t_fit = np.linspace(t_off[0], t_off[-1], 100)
            ax_off.plot(t_fit, fitter.exp_decay(t_fit - t_fit[0], *fitter.tau_off_popt), '-', label='拟合')

            # 添加标题和重新拟合信息
            title = f'周期{actual_cycle_num} - 关闭拟合 (τ = {fitter.get_tau_off():.5f} s, R² = {fitter.tau_off_r_squared:.3f})'
            if was_refitted and 'off' in [info['refit_types'] for info in self.refitted_cycles if info['cycle'] == actual_cycle_num][0]:
                title += ' [已重新拟合]'
            ax_off.set_title(title)

            ax_off.set_xlabel('时间 (s)')
            ax_off.set_ylabel('信号')
            ax_off.legend()
            ax_off.grid(True)

        plt.suptitle(f'周期{start_cycle+1}-{end_cycle}的拟合结果', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为suptitle留出空间
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

    def plot_r_squared_values(self, figsize=(10, 6)):
        """
        绘制所有周期的R²值，突出显示重新拟合的周期
        
        参数:
        -----
        figsize : tuple, optional
            图形大小(宽度, 高度)，单位为英寸
        """
        if not self.cycle_results:
            print("没有可用的周期结果。请先运行fit_all_cycles()。")
            return

        summary = self.get_summary_data()

        plt.figure(figsize=figsize)

        # 创建x位置
        cycles = summary['cycle']

        # 绘制R²值
        plt.plot(cycles, summary['r_squared_on'], 'o-', label='R² On', color='blue')
        plt.plot(cycles, summary['r_squared_off'], 'o-', label='R² Off', color='red')

        # 突出显示重新拟合的周期
        refitted = summary[summary['was_refitted']]
        if not refitted.empty:
            plt.scatter(refitted['cycle'], refitted['r_squared_on'],
                        s=100, facecolors='none', edgecolors='blue', linewidth=2,
                        label='重新拟合 On')
            plt.scatter(refitted['cycle'], refitted['r_squared_off'],
                        s=100, facecolors='none', edgecolors='red', linewidth=2,
                        label='重新拟合 Off')

        # 在阈值处添加水平线(假设为0.95，如果未提供)
        if hasattr(self, 'last_r_squared_threshold'):
            threshold = self.last_r_squared_threshold
        else:
            threshold = 0.95

        plt.axhline(y=threshold, color='green', linestyle='--',
                    label=f'阈值 ({threshold})')

        plt.xlabel('周期')
        plt.ylabel('R²')
        plt.title('每个周期的R²值')
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
            print("尚未确定窗口。请先运行find_best_windows()或fit_all_cycles()。")
            return

        # 创建图形
        plt.figure(figsize=figsize)

        if plot_full_signal:
            # 绘制整个信号
            plt.plot(self.time, self.signal, '-', label='信号')

            # 计算数据中有多少个完整周期
            total_cycles = int((self.time[-1] - self.time[0]) / self.period)

            # 为数据中的每个完整周期添加窗口
            for i in range(total_cycles):
                self._plot_cycle_windows(i, i==0)  # 仅将第一个周期包含在图例中
        else:
            # 计算有效周期范围
            max_cycle = int((self.time[-1] - self.time[0]) / self.period)
            if start_cycle >= max_cycle:
                print(f"起始周期{start_cycle}超过可用周期{max_cycle}")
                return

            # 计算要显示的周期范围
            end_cycle = min(start_cycle + num_cycles, max_cycle)
            actual_cycles = end_cycle - start_cycle

            # 计算要显示的时间范围
            start_time = self.time[0] + start_cycle * self.period
            end_time = self.time[0] + end_cycle * self.period

            # 过滤出选定时间范围内的数据
            mask = (self.time >= start_time) & (self.time <= end_time)
            plt.plot(self.time[mask], self.signal[mask], '-', label='信号')

            # 为选定的周期添加窗口
            for i in range(start_cycle, end_cycle):
                self._plot_cycle_windows(i, i==start_cycle)  # 仅将第一个周期包含在图例中

        plt.xlabel('时间 (s)')
        plt.ylabel('信号')

        if plot_full_signal:
            plt.title('完整信号与开启和关闭窗口')
        else:
            plt.title(f'信号与开启和关闭窗口(周期{start_cycle+1}至{start_cycle+actual_cycles})')

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
                    label='开启窗口' if include_in_legend else "")

        # 突出显示关闭窗口
        plt.axvspan(off_window_start, off_window_end, alpha=0.2, color='red',
                    label='关闭窗口' if include_in_legend else "")
