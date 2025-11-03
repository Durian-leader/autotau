"""
Cached Window Search Strategy for AutoTauFitter

Phase 2.1 优化：跨步窗口缓存
- 首步全搜索
- 后续步复用窗口
- 定期重新验证
- 预期加速：5-10x
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable
from .auto_tau_fitter import AutoTauFitter
from .tau_fitter import TauFitter


class CachedAutoTauFitter:
    """
    带窗口缓存的 AutoTauFitter

    核心优化：
    1. 首个步骤进行完整窗口搜索
    2. 后续步骤优先使用缓存的窗口参数
    3. 如果缓存窗口拟合质量 < 阈值，则重新搜索
    4. 每 N 步强制重新验证

    预期加速：5-10x（避免 80-95% 的窗口搜索）
    """

    def __init__(self,
                 base_fitter_factory: Optional[Callable] = None,
                 validation_threshold: float = 0.95,
                 revalidation_interval: int = 500):
        """
        初始化 CachedAutoTauFitter

        参数:
        -----
        base_fitter_factory : callable, optional
            创建 AutoTauFitter 的工厂函数
            - None: 使用默认串行 AutoTauFitter
            - callable: 返回配置好的 AutoTauFitter 实例
        validation_threshold : float, optional
            缓存窗口验证阈值（R² 阈值）
            - 如果缓存窗口的 R² < threshold，则重新搜索
            - 默认 0.95（95% 拟合质量）
        revalidation_interval : int, optional
            强制重新验证间隔（步数）
            - 每 N 步强制重新搜索一次窗口
            - 默认 500 步
        """
        self.base_fitter_factory = base_fitter_factory
        self.validation_threshold = validation_threshold
        self.revalidation_interval = revalidation_interval

        # 缓存存储
        self.cached_window_on: Optional[Dict[str, Any]] = None
        self.cached_window_off: Optional[Dict[str, Any]] = None
        self.last_full_search_step = -1

        # 统计信息
        self.stats = {
            'total_steps': 0,
            'full_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'revalidations': 0
        }

    def fit_step(self,
                 time: np.ndarray,
                 signal: np.ndarray,
                 sample_step: float,
                 period: float,
                 step_index: int = 0,
                 force_full_search: bool = False,
                 **kwargs) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        拟合单个步骤（带缓存策略）

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
        step_index : int, optional
            当前步骤索引（用于缓存逻辑）
        force_full_search : bool, optional
            强制进行完整窗口搜索
        **kwargs :
            传递给 AutoTauFitter 的参数

        返回:
        -----
        tuple
            (tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2)
        """
        self.stats['total_steps'] += 1

        # 决策：是否需要完整搜索
        need_full_search = (
            force_full_search or
            self.cached_window_on is None or  # 首次
            self.cached_window_off is None or
            (step_index - self.last_full_search_step >= self.revalidation_interval)  # 定期重验
        )

        if need_full_search:
            # 执行完整窗口搜索
            return self._full_search(time, signal, sample_step, period, step_index, **kwargs)

        else:
            # 尝试使用缓存窗口
            result = self._try_cached_windows(time, signal, sample_step, period, step_index, **kwargs)

            if result is not None:
                # 缓存命中
                self.stats['cache_hits'] += 1
                return result
            else:
                # 缓存失败，回退到完整搜索
                self.stats['cache_misses'] += 1
                return self._full_search(time, signal, sample_step, period, step_index, **kwargs)

    def _full_search(self,
                     time: np.ndarray,
                     signal: np.ndarray,
                     sample_step: float,
                     period: float,
                     step_index: int,
                     **kwargs) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        执行完整窗口搜索并更新缓存

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
        step_index : int
            当前步骤索引
        **kwargs :
            传递给 AutoTauFitter 的参数

        返回:
        -----
        tuple
            (tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2)
        """
        self.stats['full_searches'] += 1

        # 创建 AutoTauFitter
        if self.base_fitter_factory:
            fitter = self.base_fitter_factory(time, signal, sample_step=sample_step, period=period, **kwargs)
        else:
            fitter = AutoTauFitter(time, signal, sample_step=sample_step, period=period, **kwargs)

        # 执行完整搜索
        tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2 = fitter.fit_tau_on_and_off()

        # 更新缓存
        self.cached_window_on = {
            'start_time': fitter.best_tau_on_window_start_time,
            'end_time': fitter.best_tau_on_window_end_time,
            'size': fitter.best_tau_on_window_size,
            'start_offset': fitter.best_tau_on_window_start_time - time[0],  # 相对偏移
            'duration': fitter.best_tau_on_window_end_time - fitter.best_tau_on_window_start_time
        }

        self.cached_window_off = {
            'start_time': fitter.best_tau_off_window_start_time,
            'end_time': fitter.best_tau_off_window_end_time,
            'size': fitter.best_tau_off_window_size,
            'start_offset': fitter.best_tau_off_window_start_time - time[0],
            'duration': fitter.best_tau_off_window_end_time - fitter.best_tau_off_window_start_time
        }

        self.last_full_search_step = step_index

        return tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2

    def _try_cached_windows(self,
                            time: np.ndarray,
                            signal: np.ndarray,
                            sample_step: float,
                            period: float,
                            step_index: int,
                            **kwargs) -> Optional[Tuple[np.ndarray, float, np.ndarray, float]]:
        """
        尝试使用缓存的窗口参数进行拟合

        如果缓存窗口的 R² >= validation_threshold，则返回结果
        否则返回 None（触发完整搜索）

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
        step_index : int
            当前步骤索引
        **kwargs :
            传递给 TauFitter 的参数

        返回:
        -----
        tuple or None
            (tau_on_popt, tau_on_r2, tau_off_popt, tau_off_r2) 或 None（缓存失败）
        """
        try:
            # 应用缓存窗口（使用相对偏移）
            t_start = time[0]

            # On window
            on_start_time = t_start + self.cached_window_on['start_offset']
            on_duration = self.cached_window_on['duration']
            on_end_time = on_start_time + on_duration

            # Off window
            off_start_time = t_start + self.cached_window_off['start_offset']
            off_duration = self.cached_window_off['duration']
            off_end_time = off_start_time + off_duration

            # 提取窗口数据
            on_mask = (time >= on_start_time) & (time <= on_end_time)
            off_mask = (time >= off_start_time) & (time <= off_end_time)

            if on_mask.sum() < 3 or off_mask.sum() < 3:
                # 窗口数据点不足
                return None

            time_on = time[on_mask]
            signal_on = signal[on_mask]
            time_off = time[off_mask]
            signal_off = signal[off_mask]

            # 拟合 On
            normalize = kwargs.get('normalize', False)
            language = kwargs.get('language', 'en')
            interp = kwargs.get('interp', True)
            points_after_interp = kwargs.get('points_after_interp', 100)

            tau_fitter_on = TauFitter(
                time_on, signal_on,
                t_on_idx=[time_on[0], time_on[-1]],
                t_off_idx=[time_on[0], time_on[-1]],
                language=language,
                normalize=normalize
            )
            tau_fitter_on.fit_tau_on(interp=interp, points_after_interp=points_after_interp)

            # 拟合 Off
            tau_fitter_off = TauFitter(
                time_off, signal_off,
                t_on_idx=[time_off[0], time_off[-1]],
                t_off_idx=[time_off[0], time_off[-1]],
                language=language,
                normalize=normalize
            )
            tau_fitter_off.fit_tau_off(interp=interp, points_after_interp=points_after_interp)

            # 验证拟合质量
            r2_on = tau_fitter_on.tau_on_r_squared_adj
            r2_off = tau_fitter_off.tau_off_r_squared_adj

            if r2_on is None or r2_off is None:
                return None

            if r2_on < self.validation_threshold or r2_off < self.validation_threshold:
                # 拟合质量不足，需要重新搜索
                return None

            # 缓存窗口拟合成功
            return (
                tau_fitter_on.tau_on_popt,
                r2_on,
                tau_fitter_off.tau_off_popt,
                r2_off
            )

        except Exception as e:
            # 缓存拟合失败
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        返回:
        -----
        dict
            统计信息字典
        """
        if self.stats['total_steps'] == 0:
            cache_hit_rate = 0.0
        else:
            cache_hit_rate = self.stats['cache_hits'] / self.stats['total_steps']

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'search_reduction': f"{cache_hit_rate * 100:.1f}%",
            'estimated_speedup': f"{1 / (1 - cache_hit_rate):.1f}x" if cache_hit_rate < 1 else "∞"
        }

    def reset_cache(self):
        """重置缓存"""
        self.cached_window_on = None
        self.cached_window_off = None
        self.last_full_search_step = -1
        self.stats = {
            'total_steps': 0,
            'full_searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'revalidations': 0
        }
