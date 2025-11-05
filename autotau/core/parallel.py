import math
import os

# Ensure heavy numeric libraries do not oversubscribe CPU cores before NumPy loads.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from itertools import repeat

import numpy as np
import concurrent.futures
import multiprocessing
import pandas as pd
from tqdm import tqdm
import warnings
from .tau_fitter import TauFitter
from .auto_tau_fitter import AutoTauFitter
import matplotlib.pyplot as plt


def _get_mp_context():
    """Return a multiprocessing context that favours fork when available."""
    try:
        return multiprocessing.get_context('fork')
    except ValueError:  # Windows / platforms without fork support
        return multiprocessing.get_context('spawn')


def _auto_chunk_size(total_items, max_workers, *, min_chunk=256, max_chunk=20000, target_chunks_per_worker=8):
    """Compute a reasonable chunk size for distributing work evenly."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if total_items <= 0:
        return min_chunk

    estimated = max(total_items // (max_workers * target_chunks_per_worker), 1)
    estimated = max(min_chunk, estimated)
    return min(max_chunk, estimated)


def _chunk_sequence(sequence_length, chunk_size):
    """Yield tuple chunks of monotonically increasing indices."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for start in range(0, sequence_length, chunk_size):
        end = min(sequence_length, start + chunk_size)
        yield tuple(range(start, end))


def _chunk_pairs(pair_iterable, chunk_size):
    """Yield tuple chunks from an iterable of (window_points, start_idx)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch = []
    for pair in pair_iterable:
        batch.append(pair)
        if len(batch) >= chunk_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


_WINDOW_WORKER_STATE = {}
_CYCLE_WORKER_STATE = {}


def _init_window_worker(time, signal, sample_step, language, normalize):
    """Initialise per-process state for window-based workers."""
    time = np.asarray(time)
    signal = np.asarray(signal)
    time.setflags(write=False)
    signal.setflags(write=False)

    _WINDOW_WORKER_STATE.clear()
    _WINDOW_WORKER_STATE.update({
        'time': time,
        'signal': signal,
        'sample_step': float(sample_step),
        'language': language,
        'normalize': bool(normalize),
    })


def _init_cycle_worker(time, signal, sample_rate, period, language, normalize,
                       window_on_offset, window_on_size, window_off_offset, window_off_size,
                       auto_tau_params, base_time):
    """Initialise per-process state for cycle-based workers."""
    time = np.asarray(time)
    signal = np.asarray(signal)
    time.setflags(write=False)
    signal.setflags(write=False)

    _CYCLE_WORKER_STATE.clear()
    _CYCLE_WORKER_STATE.update({
        'time': time,
        'signal': signal,
        'sample_rate': float(sample_rate),
        'period': float(period),
        'language': language,
        'normalize': bool(normalize),
        'window_on_offset': float(window_on_offset),
        'window_on_size': float(window_on_size),
        'window_off_offset': float(window_off_offset),
        'window_off_size': float(window_off_size),
        'auto_tau_params': auto_tau_params,
        'base_time': float(base_time),
    })


def _process_window_chunk(chunk, interp, points_after_interp):
    """Process a chunk of window configurations and return the best fits."""
    state = _WINDOW_WORKER_STATE
    time = state['time']
    signal = state['signal']
    sample_step = state['sample_step']
    language = state['language']
    normalize = state['normalize']

    best_on = None
    best_off = None

    for window_points, start_idx in chunk:
        end_idx = start_idx + window_points
        if end_idx > time.shape[0]:
            continue

        window_time = time[start_idx:end_idx]
        window_signal = signal[start_idx:end_idx]
        if window_time.size < 3:
            continue

        try:
            tau_fitter = TauFitter(
                window_time,
                window_signal,
                t_on_idx=[float(window_time[0]), float(window_time[-1])],
                t_off_idx=[float(window_time[0]), float(window_time[-1])],
                language=language,
                normalize=normalize,
            )

            on_popt, on_pcov, on_r2, on_r2_adj = tau_fitter.fit_tau_on(
                interp=interp,
                points_after_interp=points_after_interp,
            )
            off_popt, off_pcov, off_r2, off_r2_adj = tau_fitter.fit_tau_off(
                interp=interp,
                points_after_interp=points_after_interp,
            )
        except Exception:
            continue

        on_candidate = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_points': window_points,
            'window_start_time': float(window_time[0]),
            'window_end_time': float(window_time[-1]),
            'window_size': window_points * sample_step,
            'tau_popt': tuple(map(float, on_popt)),
            'tau_pcov': on_pcov.tolist() if on_pcov is not None else None,
            'r_squared': float(on_r2),
            'r_squared_adj': float(on_r2_adj),
        }

        if best_on is None or on_candidate['r_squared_adj'] > best_on['r_squared_adj']:
            best_on = on_candidate

        off_candidate = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'window_points': window_points,
            'window_start_time': float(window_time[0]),
            'window_end_time': float(window_time[-1]),
            'window_size': window_points * sample_step,
            'tau_popt': tuple(map(float, off_popt)),
            'tau_pcov': off_pcov.tolist() if off_pcov is not None else None,
            'r_squared': float(off_r2),
            'r_squared_adj': float(off_r2_adj),
        }

        if best_off is None or off_candidate['r_squared_adj'] > best_off['r_squared_adj']:
            best_off = off_candidate

    return {'on': best_on, 'off': best_off}


def _time_to_index(time_value, base_time, sample_rate, *, is_end=False, max_index=None):
    """Convert a timestamp to a discrete sample index using the sample rate."""
    delta = (time_value - base_time) * sample_rate
    if is_end:
        idx = math.ceil(delta)
    else:
        idx = math.floor(delta)
    idx = max(0, int(idx))
    if max_index is not None:
        idx = min(max_index, idx)
    return idx


def _process_cycles_chunk(cycle_indices, interp, points_after_interp, r_squared_threshold):
    """Process a chunk of cycle indices and return fitting summaries."""
    state = _CYCLE_WORKER_STATE
    time = state['time']
    signal = state['signal']
    sample_rate = state['sample_rate']
    period = state['period']
    language = state['language']
    normalize = state['normalize']
    window_on_offset = state['window_on_offset']
    window_on_size = state['window_on_size']
    window_off_offset = state['window_off_offset']
    window_off_size = state['window_off_size']
    auto_tau_params = state['auto_tau_params']
    base_time = state['base_time']

    max_index = time.shape[0]
    sample_step = 1.0 / sample_rate if sample_rate != 0 else (time[1] - time[0])

    results = []

    for cycle_index in cycle_indices:
        cycle_start_time = base_time + cycle_index * period
        cycle_end_time = cycle_start_time + period

        cycle_start_idx = _time_to_index(cycle_start_time, base_time, sample_rate, max_index=max_index)
        cycle_end_idx = _time_to_index(cycle_end_time, base_time, sample_rate, is_end=True, max_index=max_index)

        cycle_start_idx = min(cycle_start_idx, max_index - 1)
        cycle_end_idx = max(cycle_end_idx, cycle_start_idx + 1)
        cycle_end_idx = min(cycle_end_idx, max_index)

        cycle_time = time[cycle_start_idx:cycle_end_idx]
        cycle_signal = signal[cycle_start_idx:cycle_end_idx]

        if cycle_time.size < 3:
            results.append({
                'status': 'skipped',
                'cycle': cycle_index + 1,
                'message': f"周期{cycle_index + 1}的数据点不足。"
            })
            continue

        on_start_time = cycle_start_time + window_on_offset
        on_end_time = on_start_time + window_on_size
        off_start_time = cycle_start_time + window_off_offset
        off_end_time = off_start_time + window_off_size

        on_start_idx = _time_to_index(on_start_time, base_time, sample_rate, max_index=max_index)
        on_end_idx = _time_to_index(on_end_time, base_time, sample_rate, is_end=True, max_index=max_index)
        off_start_idx = _time_to_index(off_start_time, base_time, sample_rate, max_index=max_index)
        off_end_idx = _time_to_index(off_end_time, base_time, sample_rate, is_end=True, max_index=max_index)

        on_start_idx_rel = max(0, on_start_idx - cycle_start_idx)
        on_end_idx_rel = max(on_start_idx_rel + 1, min(cycle_time.size, on_end_idx - cycle_start_idx))
        off_start_idx_rel = max(0, off_start_idx - cycle_start_idx)
        off_end_idx_rel = max(off_start_idx_rel + 1, min(cycle_time.size, off_end_idx - cycle_start_idx))

        if (on_end_idx_rel - on_start_idx_rel) < 3 or (off_end_idx_rel - off_start_idx_rel) < 3:
            results.append({
                'status': 'skipped',
                'cycle': cycle_index + 1,
                'message': f"周期{cycle_index + 1}的数据点不足。"
            })
            continue

        tau_fitter = TauFitter(
            cycle_time,
            cycle_signal,
            t_on_idx=[float(on_start_time), float(on_end_time)],
            t_off_idx=[float(off_start_time), float(off_end_time)],
            language=language,
            normalize=normalize,
        )

        tau_fitter.fit_tau_on(interp=interp, points_after_interp=points_after_interp)
        tau_fitter.fit_tau_off(interp=interp, points_after_interp=points_after_interp)

        r_squared_on = tau_fitter.tau_on_r_squared
        r_squared_off = tau_fitter.tau_off_r_squared

        needs_refit_on = r_squared_on is None or r_squared_on < r_squared_threshold
        needs_refit_off = r_squared_off is None or r_squared_off < r_squared_threshold

        was_refitted = False
        refit_info = None

        if needs_refit_on or needs_refit_off:
            if cycle_time.size >= 10:
                refit_info = {
                    'cycle': cycle_index + 1,
                    'original_r_squared_on': r_squared_on,
                    'original_r_squared_off': r_squared_off,
                    'refit_types': [],
                }

                refit_params = dict(auto_tau_params)
                refit_params.setdefault('show_progress', False)

                cycle_auto_fitter = AutoTauFitter(
                    cycle_time,
                    cycle_signal,
                    sample_step=sample_step,
                    period=period,
                    **refit_params,
                )
                cycle_auto_fitter.fit_tau_on_and_off(
                    interp=interp,
                    points_after_interp=points_after_interp,
                )

                if needs_refit_on and cycle_auto_fitter.best_tau_on_fitter is not None:
                    new_fitter = cycle_auto_fitter.best_tau_on_fitter
                    if new_fitter.tau_on_r_squared is not None and (
                        r_squared_on is None or new_fitter.tau_on_r_squared > r_squared_on
                    ):
                        tau_fitter.tau_on_popt = new_fitter.tau_on_popt
                        tau_fitter.tau_on_pcov = new_fitter.tau_on_pcov
                        tau_fitter.tau_on_r_squared = new_fitter.tau_on_r_squared
                        tau_fitter.tau_on_r_squared_adj = new_fitter.tau_on_r_squared_adj
                        tau_fitter.t_on_idx = new_fitter.t_on_idx
                        r_squared_on = tau_fitter.tau_on_r_squared
                        refit_info['refit_types'].append('on')
                        refit_info['new_r_squared_on'] = tau_fitter.tau_on_r_squared

                if needs_refit_off and cycle_auto_fitter.best_tau_off_fitter is not None:
                    new_fitter = cycle_auto_fitter.best_tau_off_fitter
                    if new_fitter.tau_off_r_squared is not None and (
                        r_squared_off is None or new_fitter.tau_off_r_squared > r_squared_off
                    ):
                        tau_fitter.tau_off_popt = new_fitter.tau_off_popt
                        tau_fitter.tau_off_pcov = new_fitter.tau_off_pcov
                        tau_fitter.tau_off_r_squared = new_fitter.tau_off_r_squared
                        tau_fitter.tau_off_r_squared_adj = new_fitter.tau_off_r_squared_adj
                        tau_fitter.t_off_idx = new_fitter.t_off_idx
                        r_squared_off = tau_fitter.tau_off_r_squared
                        refit_info['refit_types'].append('off')
                        refit_info['new_r_squared_off'] = tau_fitter.tau_off_r_squared

                if refit_info['refit_types']:
                    was_refitted = True
                else:
                    refit_info = None

        tau_on_value = tau_fitter.get_tau_on()
        tau_off_value = tau_fitter.get_tau_off()

        result = {
            'status': 'success',
            'cycle': cycle_index + 1,
            'cycle_start_time': float(cycle_start_time),
            'cycle_start_idx': cycle_start_idx,
            'cycle_end_idx': cycle_end_idx,
            'tau_on': float(tau_on_value) if tau_on_value is not None else None,
            'tau_off': float(tau_off_value) if tau_off_value is not None else None,
            'tau_on_popt': tuple(map(float, tau_fitter.tau_on_popt)) if tau_fitter.tau_on_popt is not None else None,
            'tau_off_popt': tuple(map(float, tau_fitter.tau_off_popt)) if tau_fitter.tau_off_popt is not None else None,
            'tau_on_pcov': tau_fitter.tau_on_pcov.tolist() if tau_fitter.tau_on_pcov is not None else None,
            'tau_off_pcov': tau_fitter.tau_off_pcov.tolist() if tau_fitter.tau_off_pcov is not None else None,
            'tau_on_r_squared': float(tau_fitter.tau_on_r_squared) if tau_fitter.tau_on_r_squared is not None else None,
            'tau_off_r_squared': float(tau_fitter.tau_off_r_squared) if tau_fitter.tau_off_r_squared is not None else None,
            'tau_on_r_squared_adj': float(tau_fitter.tau_on_r_squared_adj) if tau_fitter.tau_on_r_squared_adj is not None else None,
            'tau_off_r_squared_adj': float(tau_fitter.tau_off_r_squared_adj) if tau_fitter.tau_off_r_squared_adj is not None else None,
            'window_on_start_time': float(tau_fitter.t_on_idx[0]) if tau_fitter.t_on_idx else float(on_start_time),
            'window_on_end_time': float(tau_fitter.t_on_idx[1]) if tau_fitter.t_on_idx else float(on_end_time),
            'window_off_start_time': float(tau_fitter.t_off_idx[0]) if tau_fitter.t_off_idx else float(off_start_time),
            'window_off_end_time': float(tau_fitter.t_off_idx[1]) if tau_fitter.t_off_idx else float(off_end_time),
            'window_on_start_idx': _time_to_index(tau_fitter.t_on_idx[0], base_time, sample_rate, max_index=max_index) if tau_fitter.t_on_idx else on_start_idx,
            'window_on_end_idx': _time_to_index(tau_fitter.t_on_idx[1], base_time, sample_rate, is_end=True, max_index=max_index) if tau_fitter.t_on_idx else on_end_idx,
            'window_off_start_idx': _time_to_index(tau_fitter.t_off_idx[0], base_time, sample_rate, max_index=max_index) if tau_fitter.t_off_idx else off_start_idx,
            'window_off_end_idx': _time_to_index(tau_fitter.t_off_idx[1], base_time, sample_rate, is_end=True, max_index=max_index) if tau_fitter.t_off_idx else off_end_idx,
            'was_refitted': was_refitted,
            'refit_info': refit_info,
        }

        results.append(result)

    return results

class ParallelAutoTauFitter:
    """
    AutoTauFitter的并行版本，使用多进程加速窗口搜索

    ⚠️ DEPRECATED (v0.3.0): 此类已被废弃
    ----------------------------------------
    请改用 AutoTauFitter(..., executor=ProcessPoolExecutor(max_workers=N))

    废弃原因：
    - 硬编码的并行策略导致嵌套并行问题
    - 无法与上层框架（如 features_v2）的并行策略协调
    - 新架构支持更灵活的并行配置

    迁移示例：
        # 旧代码（废弃）
        from autotau import ParallelAutoTauFitter
        fitter = ParallelAutoTauFitter(..., max_workers=8)

        # 新代码（推荐）
        from autotau import AutoTauFitter
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8) as executor:
            fitter = AutoTauFitter(..., executor=executor)
            result = fitter.fit_tau_on_and_off()

    利用多核CPU并行处理不同窗口大小和位置的拟合任务，大幅提升滑动窗口搜索速度
    """

    def __init__(self, time, signal, sample_step, period, window_scalar_min=1/5, window_scalar_max=1/3,
                 window_points_step=10, window_start_idx_step=1, normalize=False, language='en',
                 show_progress=False, max_workers=None):
        """
        初始化并行版AutoTauFitter

        ⚠️ DEPRECATED: 请改用 AutoTauFitter(..., executor=ProcessPoolExecutor(...))
        
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
        # ⚠️ 发出废弃警告
        warnings.warn(
            "\n"
            "=" * 70 + "\n"
            "⚠️  ParallelAutoTauFitter 已被废弃 (v0.3.0)\n"
            "=" * 70 + "\n"
            "请改用灵活的新 API：\n\n"
            "  from autotau import AutoTauFitter\n"
            "  from concurrent.futures import ProcessPoolExecutor\n\n"
            "  with ProcessPoolExecutor(max_workers=8) as executor:\n"
            "      fitter = AutoTauFitter(..., executor=executor)\n"
            "      result = fitter.fit_tau_on_and_off()\n\n"
            "新架构优势：\n"
            "  ✓ 避免嵌套并行问题\n"
            "  ✓ 与 features_v2 等上层框架完美集成\n"
            "  ✓ 更灵活的并行策略控制\n"
            "=" * 70,
            DeprecationWarning,
            stacklevel=2
        )

        self.time = np.array(time)
        self.signal = np.array(signal)
        self.sample_step = sample_step
        self.period = period
        self.normalize = normalize
        self.language = language
        self.window_length_min = window_scalar_min * self.period
        self.window_length_max = window_scalar_max * self.period
        self.show_progress = show_progress
        context = _get_mp_context()
        self._mp_context = context
        cpu_total = context.cpu_count() if hasattr(context, 'cpu_count') else None
        if cpu_total is None:
            cpu_total = max(1, os.cpu_count() or 1)
        self.max_workers = max_workers if max_workers is not None else cpu_total

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
        # 初始化状态
        best_tau_on = None
        best_tau_off = None
        self.best_tau_on_fitter = None
        self.best_tau_off_fitter = None
        self.best_tau_on_window_start_time = None
        self.best_tau_on_window_end_time = None
        self.best_tau_on_window_size = None
        self.best_tau_off_window_start_time = None
        self.best_tau_off_window_end_time = None
        self.best_tau_off_window_size = None
        
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

        if not window_params_list:
            return (None, None, None, None)
        
        # 使用ProcessPoolExecutor并行处理所有窗口
        chunk_size = _auto_chunk_size(len(window_params_list), self.max_workers)
        total_chunks = math.ceil(len(window_params_list) / chunk_size) if chunk_size else 0
        chunks_iter = _chunk_pairs(window_params_list, chunk_size) if chunk_size else ()

        executor_kwargs = {
            'max_workers': self.max_workers,
            'mp_context': self._mp_context,
            'initializer': _init_window_worker,
            'initargs': (self.time, self.signal, self.sample_step, self.language, self.normalize),
        }

        with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
            mapped = executor.map(
                _process_window_chunk,
                chunks_iter,
                repeat(interp),
                repeat(points_after_interp),
                chunksize=1,
            )

            if self.show_progress:
                mapped = tqdm(mapped, total=total_chunks, desc="并行拟合进度")

            for result in mapped:
                if not result:
                    continue

                on_result = result.get('on')
                if on_result is not None and (
                    best_tau_on is None or on_result['r_squared_adj'] > best_tau_on['r_squared_adj']
                ):
                    best_tau_on = on_result

                off_result = result.get('off')
                if off_result is not None and (
                    best_tau_off is None or off_result['r_squared_adj'] > best_tau_off['r_squared_adj']
                ):
                    best_tau_off = off_result

        if best_tau_on is not None:
            start_idx = best_tau_on['start_idx']
            end_idx = best_tau_on['end_idx']
            window_time = self.time[start_idx:end_idx]
            window_signal = self.signal[start_idx:end_idx]
            best_fitter = TauFitter(
                window_time,
                window_signal,
                t_on_idx=[best_tau_on['window_start_time'], best_tau_on['window_end_time']],
                t_off_idx=[best_tau_on['window_start_time'], best_tau_on['window_end_time']],
                language=self.language,
                normalize=self.normalize,
            )
            best_fitter.tau_on_popt = np.array(best_tau_on['tau_popt']) if best_tau_on['tau_popt'] is not None else None
            best_fitter.tau_on_pcov = np.array(best_tau_on['tau_pcov']) if best_tau_on['tau_pcov'] is not None else None
            best_fitter.tau_on_r_squared = best_tau_on['r_squared']
            best_fitter.tau_on_r_squared_adj = best_tau_on['r_squared_adj']
            self.best_tau_on_fitter = best_fitter

            self.best_tau_on_window_start_time = best_tau_on['window_start_time']
            self.best_tau_on_window_end_time = best_tau_on['window_end_time']
            self.best_tau_on_window_size = best_tau_on['window_size']
        else:
            self.best_tau_on_window_start_time = None
            self.best_tau_on_window_end_time = None
            self.best_tau_on_window_size = None

        if best_tau_off is not None:
            start_idx = best_tau_off['start_idx']
            end_idx = best_tau_off['end_idx']
            window_time = self.time[start_idx:end_idx]
            window_signal = self.signal[start_idx:end_idx]
            best_fitter = TauFitter(
                window_time,
                window_signal,
                t_on_idx=[best_tau_off['window_start_time'], best_tau_off['window_end_time']],
                t_off_idx=[best_tau_off['window_start_time'], best_tau_off['window_end_time']],
                language=self.language,
                normalize=self.normalize,
            )
            best_fitter.tau_off_popt = np.array(best_tau_off['tau_popt']) if best_tau_off['tau_popt'] is not None else None
            best_fitter.tau_off_pcov = np.array(best_tau_off['tau_pcov']) if best_tau_off['tau_pcov'] is not None else None
            best_fitter.tau_off_r_squared = best_tau_off['r_squared']
            best_fitter.tau_off_r_squared_adj = best_tau_off['r_squared_adj']
            self.best_tau_off_fitter = best_fitter

            self.best_tau_off_window_start_time = best_tau_off['window_start_time']
            self.best_tau_off_window_end_time = best_tau_off['window_end_time']
            self.best_tau_off_window_size = best_tau_off['window_size']
        else:
            self.best_tau_off_window_start_time = None
            self.best_tau_off_window_end_time = None
            self.best_tau_off_window_size = None

        return (
            np.array(best_tau_on['tau_popt']) if best_tau_on is not None else None,
            best_tau_on['r_squared_adj'] if best_tau_on is not None else None,
            np.array(best_tau_off['tau_popt']) if best_tau_off is not None else None,
            best_tau_off['r_squared_adj'] if best_tau_off is not None else None,
        )


class ParallelCyclesAutoTauFitter:
    """
    CyclesAutoTauFitter的并行版本，使用多进程加速多个周期的处理

    ⚠️ DEPRECATED (v0.3.0): 此类已被废弃
    ----------------------------------------
    请改用 CyclesAutoTauFitter(..., fitter_factory=...)

    废弃原因：
    - 硬编码的并行策略导致嵌套并行问题
    - 无法与上层框架（如 features_v2）的并行策略协调
    - 新架构支持更灵活的并行配置

    迁移示例：
        # 旧代码（废弃）
        from autotau import ParallelCyclesAutoTauFitter
        fitter = ParallelCyclesAutoTauFitter(..., max_workers=8)

        # 新代码（推荐 - 默认串行）
        from autotau import CyclesAutoTauFitter
        fitter = CyclesAutoTauFitter(...)  # 默认串行，适合上层框架调用

        # 新代码（可选 - 窗口搜索并行）
        from autotau import CyclesAutoTauFitter, AutoTauFitter
        from concurrent.futures import ProcessPoolExecutor

        executor = ProcessPoolExecutor(max_workers=8)
        fitter_factory = lambda time, signal, **kw: AutoTauFitter(
            time, signal, executor=executor, **kw
        )
        fitter = CyclesAutoTauFitter(..., fitter_factory=fitter_factory)

    利用多核CPU并行处理不同周期的拟合任务，大幅提升多周期数据的处理速度
    """

    def __init__(self, time, signal, period, sample_rate, **kwargs):
        """
        初始化并行版CyclesAutoTauFitter

        ⚠️ DEPRECATED: 请改用 CyclesAutoTauFitter(..., fitter_factory=...)

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
        # ⚠️ 发出废弃警告
        warnings.warn(
            "\n"
            "=" * 70 + "\n"
            "⚠️  ParallelCyclesAutoTauFitter 已被废弃 (v0.3.0)\n"
            "=" * 70 + "\n"
            "请改用灵活的新 API：\n\n"
            "  from autotau import CyclesAutoTauFitter\n\n"
            "  # 默认串行（推荐，适合 features_v2 调用）\n"
            "  fitter = CyclesAutoTauFitter(...)\n\n"
            "  # 或者使用自定义 factory 实现并行窗口搜索\n"
            "  from autotau import AutoTauFitter\n"
            "  from concurrent.futures import ProcessPoolExecutor\n\n"
            "  executor = ProcessPoolExecutor(max_workers=8)\n"
            "  factory = lambda t, s, **kw: AutoTauFitter(t, s, executor=executor, **kw)\n"
            "  fitter = CyclesAutoTauFitter(..., fitter_factory=factory)\n\n"
            "新架构优势：\n"
            "  ✓ 避免嵌套并行问题\n"
            "  ✓ 与 features_v2 等上层框架完美集成\n"
            "  ✓ 更灵活的并行策略控制\n"
            "=" * 70,
            DeprecationWarning,
            stacklevel=2
        )

        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.auto_tau_fitter_params = dict(kwargs)
        context = _get_mp_context()
        self._mp_context = context
        cpu_total = context.cpu_count() if hasattr(context, 'cpu_count') else None
        if cpu_total is None:
            cpu_total = max(1, os.cpu_count() or 1)
        max_workers_value = self.auto_tau_fitter_params.pop('max_workers', kwargs.get('max_workers', None))
        self.max_workers = max_workers_value if max_workers_value is not None else cpu_total
        self.show_progress = self.auto_tau_fitter_params.get('show_progress', kwargs.get('show_progress', False))
        self.language = self.auto_tau_fitter_params.get('language', kwargs.get('language', 'en'))
        self.normalize = self.auto_tau_fitter_params.get('normalize', kwargs.get('normalize', False))
        
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
                'tau_on_y_label': 'Tau On (s)',
                'tau_off_y_label': 'Tau Off (s)'
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
                'tau_on_y_label': 'Tau On (s)',
                'tau_off_y_label': 'Tau Off (s)'
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

        if any(value is None for value in [
            self.window_on_offset,
            self.window_on_size,
            self.window_off_offset,
            self.window_off_size,
        ]):
            raise RuntimeError("Failed to determine valid on/off windows before cycle fitting.")

        # 计算数据中完整周期的数量
        total_time = self.time[-1] - self.time[0]
        num_cycles = int(total_time / self.period)

        if num_cycles <= 0:
            self.cycle_results = []
            self.refitted_cycles = []
            return self.cycle_results

        chunk_size = _auto_chunk_size(num_cycles, self.max_workers, min_chunk=32)
        total_chunks = math.ceil(num_cycles / chunk_size)
        chunk_indices_iter = _chunk_sequence(num_cycles, chunk_size)

        self.cycle_results = []
        self.refitted_cycles = []

        executor_kwargs = {
            'max_workers': self.max_workers,
            'mp_context': self._mp_context,
            'initializer': _init_cycle_worker,
            'initargs': (
                self.time,
                self.signal,
                self.sample_rate,
                self.period,
                self.language,
                self.normalize,
                self.window_on_offset,
                self.window_on_size,
                self.window_off_offset,
                self.window_off_size,
                dict(self.auto_tau_fitter_params),
                float(self.time[0]),
            ),
        }

        with concurrent.futures.ProcessPoolExecutor(**executor_kwargs) as executor:
            mapped = executor.map(
                _process_cycles_chunk,
                chunk_indices_iter,
                repeat(interp),
                repeat(points_after_interp),
                repeat(r_squared_threshold),
                chunksize=1,
            )

            if self.show_progress:
                mapped = tqdm(mapped, total=total_chunks, desc="并行处理周期")

            for batch in mapped:
                if not batch:
                    continue
                for result in batch:
                    if result['status'] == 'skipped':
                        if self.show_progress and result.get('message'):
                            print(result['message'])
                        continue

                    self.cycle_results.append(result)

                    if result.get('was_refitted') and result.get('refit_info') is not None:
                        self.refitted_cycles.append(result['refit_info'])

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
            data['tau_on'].append(res['tau_on'] if res['tau_on'] is not None else np.nan)
            data['tau_off'].append(res['tau_off'] if res['tau_off'] is not None else np.nan)
            data['r_squared_on'].append(res['tau_on_r_squared'] if res['tau_on_r_squared'] is not None else np.nan)
            data['r_squared_off'].append(res['tau_off_r_squared'] if res['tau_off_r_squared'] is not None else np.nan)
            data['r_squared_adj_on'].append(res['tau_on_r_squared_adj'] if res['tau_on_r_squared_adj'] is not None else np.nan)
            data['r_squared_adj_off'].append(res['tau_off_r_squared_adj'] if res['tau_off_r_squared_adj'] is not None else np.nan)
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
        tau_on_values = [res['tau_on'] if res['tau_on'] is not None else np.nan for res in self.cycle_results]
        tau_off_values = [res['tau_off'] if res['tau_off'] is not None else np.nan for res in self.cycle_results]

        refitted_indices = [i for i, res in enumerate(self.cycle_results) if res.get('was_refitted', False)]
        refitted_cycles = [cycles[i] for i in refitted_indices]
        refitted_tau_on = [tau_on_values[i] for i in refitted_indices]
        refitted_tau_off = [tau_off_values[i] for i in refitted_indices]

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

        if refitted_cycles:
            if dual_y_axis:
                ax1.scatter(refitted_cycles, refitted_tau_on, s=100, facecolors='none', edgecolors='blue', linewidth=2, label=self.text[self.language]['refitted_tau_on'])
                ax2.scatter(refitted_cycles, refitted_tau_off, s=100, facecolors='none', edgecolors='red', linewidth=2, label=self.text[self.language]['refitted_tau_off'])
            else:
                ax1.scatter(refitted_cycles, refitted_tau_on, s=100, facecolors='none', edgecolors='blue', linewidth=2, label=self.text[self.language]['refitted_tau_on'])
                ax1.scatter(refitted_cycles, refitted_tau_off, s=100, facecolors='none', edgecolors='red', linewidth=2, label=self.text[self.language]['refitted_tau_off'])

        plt.title(self.text[self.language]['tau_values_per_cycle'])
        lines, labels = ax1.get_legend_handles_labels()
        if dual_y_axis:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        ax1.legend(lines, labels, loc='best')
        
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

    def _build_cycle_fitter_from_result(self, result):
        """Construct a TauFitter instance for plotting based on stored summary data."""
        cycle_start_idx = result.get('cycle_start_idx')
        cycle_end_idx = result.get('cycle_end_idx')
        if cycle_start_idx is None or cycle_end_idx is None:
            return None

        cycle_time = self.time[cycle_start_idx:cycle_end_idx]
        cycle_signal = self.signal[cycle_start_idx:cycle_end_idx]

        if cycle_time.size < 3:
            return None

        window_on_start_time = result.get('window_on_start_time')
        window_on_end_time = result.get('window_on_end_time')
        window_off_start_time = result.get('window_off_start_time')
        window_off_end_time = result.get('window_off_end_time')

        tau_fitter = TauFitter(
            cycle_time,
            cycle_signal,
            t_on_idx=[window_on_start_time, window_on_end_time] if window_on_start_time is not None and window_on_end_time is not None else None,
            t_off_idx=[window_off_start_time, window_off_end_time] if window_off_start_time is not None and window_off_end_time is not None else None,
            language=self.language,
            normalize=self.normalize,
        )

        if result.get('tau_on_popt') is not None:
            tau_fitter.tau_on_popt = np.array(result['tau_on_popt'])
        if result.get('tau_off_popt') is not None:
            tau_fitter.tau_off_popt = np.array(result['tau_off_popt'])
        if result.get('tau_on_pcov') is not None:
            tau_fitter.tau_on_pcov = np.array(result['tau_on_pcov'])
        if result.get('tau_off_pcov') is not None:
            tau_fitter.tau_off_pcov = np.array(result['tau_off_pcov'])

        tau_fitter.tau_on_r_squared = result.get('tau_on_r_squared')
        tau_fitter.tau_on_r_squared_adj = result.get('tau_on_r_squared_adj')
        tau_fitter.tau_off_r_squared = result.get('tau_off_r_squared')
        tau_fitter.tau_off_r_squared_adj = result.get('tau_off_r_squared_adj')

        return tau_fitter

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
            fitter = self._build_cycle_fitter_from_result(result)
            if fitter is None or fitter.tau_on_popt is None or fitter.tau_off_popt is None:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                continue
            was_refitted = result.get('was_refitted', False)

            # 绘制开启拟合
            ax_on = axes[i, 0]

            on_start_idx = result.get('window_on_start_idx')
            on_end_idx = result.get('window_on_end_idx')
            if on_start_idx is None or on_end_idx is None or on_end_idx - on_start_idx < 2:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                continue
            t_on = self.time[on_start_idx:on_end_idx]
            s_on = self.signal[on_start_idx:on_end_idx]

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

            off_start_idx = result.get('window_off_start_idx')
            off_end_idx = result.get('window_off_end_idx')
            if off_start_idx is None or off_end_idx is None or off_end_idx - off_start_idx < 2:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                continue
            t_off = self.time[off_start_idx:off_end_idx]
            s_off = self.signal[off_start_idx:off_end_idx]

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
