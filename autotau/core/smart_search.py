"""
Smart Window Search Strategy using Global Optimization

Phase 2.2 优化：智能窗口搜索
- 使用 differential_evolution 代替网格搜索
- 全局优化算法（遗传算法变种）
- 搜索次数：50-200 次 vs 10,000-50,000 次（网格搜索）
- 预期加速：50-250x
"""

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from typing import Optional, Tuple, Dict, Any
from .tau_fitter import TauFitter


class SmartWindowSearchFitter:
    """
    使用全局优化的智能窗口搜索

    替代传统的网格搜索（exhaustive search），使用 scipy 的
    differential_evolution 算法进行全局优化。

    优势：
    1. 搜索次数大幅减少：50-200 次 vs 10,000-50,000 次
    2. 更好的全局最优保证
    3. 自适应搜索步长
    4. 早停机制

    预期加速：50-250x（窗口搜索阶段）
    """

    def __init__(self,
                 time: np.ndarray,
                 signal: np.ndarray,
                 sample_step: float,
                 period: float,
                 window_scalar_min: float = 0.1,
                 window_scalar_max: float = 0.4,
                 normalize: bool = False,
                 language: str = 'en',
                 maxiter: int = 50,
                 popsize: int = 15,
                 atol: float = 0.01,
                 tol: float = 0.001):
        """
        初始化 SmartWindowSearchFitter

        参数:
        -----
        time : np.ndarray
            时间数据
        signal : np.ndarray
            信号数据
        sample_step : float
            采样步长
        period : float
            周期
        window_scalar_min : float, optional
            窗口最小倍数（相对于周期）
        window_scalar_max : float, optional
            窗口最大倍数
        normalize : bool, optional
            是否归一化
        language : str, optional
            语言设置
        maxiter : int, optional
            最大迭代次数（differential_evolution）
            - 默认 50 次迭代
            - 每次迭代评估 popsize * 2 个点
            - 总评估次数 ≈ 50 * 15 * 2 = 1500 次（远少于网格搜索的10k-50k次）
        popsize : int, optional
            种群大小（differential_evolution）
            - 默认 15（scipy 推荐 15-20）
        atol : float, optional
            绝对容差（早停阈值）
            - 当 R² 改进 < atol 时停止
        tol : float, optional
            相对容差（早停阈值）
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.sample_step = sample_step
        self.period = period
        self.normalize = normalize
        self.language = language

        # 窗口搜索范围
        self.window_length_min = window_scalar_min * self.period
        self.window_length_max = window_scalar_max * self.period
        self.min_window_points = max(3, int(self.window_length_min / self.sample_step))
        self.max_window_points = int(self.window_length_max / self.sample_step)

        # 优化参数
        self.maxiter = maxiter
        self.popsize = popsize
        self.atol = atol
        self.tol = tol

        # 最佳拟合结果
        self.best_tau_on_fitter = None
        self.best_tau_off_fitter = None
        self.best_tau_on_window_start_time = None
        self.best_tau_off_window_start_time = None
        self.best_tau_on_window_end_time = None
        self.best_tau_off_window_end_time = None
        self.best_tau_on_window_size = None
        self.best_tau_off_window_size = None

        # 统计信息
        self.eval_count = 0
        self.optimization_result_on: Optional[OptimizeResult] = None
        self.optimization_result_off: Optional[OptimizeResult] = None

    def _objective_on(self, window_params: np.ndarray) -> float:
        """
        On 过程的目标函数（最小化负R²）

        参数:
        -----
        window_params : np.ndarray
            [window_points, start_idx]（连续值，会被四舍五入）

        返回:
        -----
        float
            负 R²（因为 differential_evolution 是最小化）
        """
        self.eval_count += 1

        window_points = int(np.round(window_params[0]))
        start_idx = int(np.round(window_params[1]))
        end_idx = start_idx + window_points

        # 边界检查
        if end_idx > len(self.time):
            return 1.0  # 惩罚值（对应 R² = -1）

        # 提取窗口数据
        window_time = self.time[start_idx:end_idx]
        window_signal = self.signal[start_idx:end_idx]

        if len(window_time) < 3:
            return 1.0

        try:
            # 拟合
            tau_fitter = TauFitter(
                window_time,
                window_signal,
                t_on_idx=[window_time[0], window_time[-1]],
                t_off_idx=[window_time[0], window_time[-1]],
                language=self.language,
                normalize=self.normalize
            )
            tau_fitter.fit_tau_on(interp=True, points_after_interp=100)

            r_squared_adj = tau_fitter.tau_on_r_squared_adj
            if r_squared_adj is None:
                return 1.0

            # 返回负 R²（最小化）
            return -r_squared_adj

        except Exception as e:
            return 1.0

    def _objective_off(self, window_params: np.ndarray) -> float:
        """
        Off 过程的目标函数（最小化负R²）

        参数:
        -----
        window_params : np.ndarray
            [window_points, start_idx]

        返回:
        -----
        float
            负 R²
        """
        self.eval_count += 1

        window_points = int(np.round(window_params[0]))
        start_idx = int(np.round(window_params[1]))
        end_idx = start_idx + window_points

        if end_idx > len(self.time):
            return 1.0

        window_time = self.time[start_idx:end_idx]
        window_signal = self.signal[start_idx:end_idx]

        if len(window_time) < 3:
            return 1.0

        try:
            tau_fitter = TauFitter(
                window_time,
                window_signal,
                t_on_idx=[window_time[0], window_time[-1]],
                t_off_idx=[window_time[0], window_time[-1]],
                language=self.language,
                normalize=self.normalize
            )
            tau_fitter.fit_tau_off(interp=True, points_after_interp=100)

            r_squared_adj = tau_fitter.tau_off_r_squared_adj
            if r_squared_adj is None:
                return 1.0

            return -r_squared_adj

        except Exception as e:
            return 1.0

    def fit_tau_on_and_off(self, interp: bool = True, points_after_interp: int = 100) -> Tuple:
        """
        使用智能搜索同时拟合 On 和 Off 过程

        返回:
        -----
        tuple
            (tau_on_popt, tau_on_r_squared_adj, tau_off_popt, tau_off_r_squared_adj)
        """
        # 定义搜索空间
        max_start_idx = len(self.time) - self.min_window_points

        bounds = [
            (self.min_window_points, self.max_window_points),  # window_points
            (0, max_start_idx)  # start_idx
        ]

        # 重置评估计数
        self.eval_count = 0

        # ========== 优化 On 窗口 ==========
        result_on = differential_evolution(
            self._objective_on,
            bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            atol=self.atol,
            tol=self.tol,
            seed=42,  # 可复现性
            workers=1,  # 不使用内部并行（避免嵌套）
            updating='deferred',  # 延迟更新（更快）
            polish=False  # 不进行局部精炼（节省时间）
        )

        self.optimization_result_on = result_on

        # 提取最佳 On 窗口
        best_window_points_on = int(np.round(result_on.x[0]))
        best_start_idx_on = int(np.round(result_on.x[1]))
        best_end_idx_on = best_start_idx_on + best_window_points_on

        # 使用最佳窗口重新拟合（获取完整结果）
        window_time_on = self.time[best_start_idx_on:best_end_idx_on]
        window_signal_on = self.signal[best_start_idx_on:best_end_idx_on]

        tau_fitter_on = TauFitter(
            window_time_on,
            window_signal_on,
            t_on_idx=[window_time_on[0], window_time_on[-1]],
            t_off_idx=[window_time_on[0], window_time_on[-1]],
            language=self.language,
            normalize=self.normalize
        )
        tau_fitter_on.fit_tau_on(interp=interp, points_after_interp=points_after_interp)

        # 保存 On 结果
        self.best_tau_on_fitter = tau_fitter_on
        self.best_tau_on_window_start_time = window_time_on[0]
        self.best_tau_on_window_end_time = window_time_on[-1]
        self.best_tau_on_window_size = best_window_points_on * self.sample_step

        # ========== 优化 Off 窗口 ==========
        result_off = differential_evolution(
            self._objective_off,
            bounds,
            maxiter=self.maxiter,
            popsize=self.popsize,
            atol=self.atol,
            tol=self.tol,
            seed=43,
            workers=1,
            updating='deferred',
            polish=False
        )

        self.optimization_result_off = result_off

        # 提取最佳 Off 窗口
        best_window_points_off = int(np.round(result_off.x[0]))
        best_start_idx_off = int(np.round(result_off.x[1]))
        best_end_idx_off = best_start_idx_off + best_window_points_off

        window_time_off = self.time[best_start_idx_off:best_end_idx_off]
        window_signal_off = self.signal[best_start_idx_off:best_end_idx_off]

        tau_fitter_off = TauFitter(
            window_time_off,
            window_signal_off,
            t_on_idx=[window_time_off[0], window_time_off[-1]],
            t_off_idx=[window_time_off[0], window_time_off[-1]],
            language=self.language,
            normalize=self.normalize
        )
        tau_fitter_off.fit_tau_off(interp=interp, points_after_interp=points_after_interp)

        # 保存 Off 结果
        self.best_tau_off_fitter = tau_fitter_off
        self.best_tau_off_window_start_time = window_time_off[0]
        self.best_tau_off_window_end_time = window_time_off[-1]
        self.best_tau_off_window_size = best_window_points_off * self.sample_step

        # 返回结果
        return (
            tau_fitter_on.tau_on_popt,
            tau_fitter_on.tau_on_r_squared_adj,
            tau_fitter_off.tau_off_popt,
            tau_fitter_off.tau_off_r_squared_adj
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取优化统计信息

        返回:
        -----
        dict
            统计信息字典
        """
        stats = {
            'total_evaluations': self.eval_count,
            'maxiter': self.maxiter,
            'popsize': self.popsize
        }

        if self.optimization_result_on:
            stats['on_iterations'] = self.optimization_result_on.nit
            stats['on_success'] = self.optimization_result_on.success
            stats['on_message'] = self.optimization_result_on.message

        if self.optimization_result_off:
            stats['off_iterations'] = self.optimization_result_off.nit
            stats['off_success'] = self.optimization_result_off.success
            stats['off_message'] = self.optimization_result_off.message

        return stats
