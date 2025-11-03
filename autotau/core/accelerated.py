"""
Numba-accelerated Functions for AutoTau

Phase 3.1 优化：Numba JIT 编译
- 编译指数函数（exp_rise, exp_decay）
- 编译 R² 计算
- 预期加速：5-10x（热点函数）
"""

import numpy as np
from typing import Tuple

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback：使用装饰器占位符
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# Numba-accelerated Exponential Functions
# ============================================================================

@jit(nopython=True, cache=True)
def exp_rise_numba(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """
    指数上升函数（Numba 编译版本）

    I(t) = A * (1 - exp(-t / tau)) + C

    参数:
    -----
    t : np.ndarray
        时间数组
    A : float
        振幅
    tau : float
        时间常数
    C : float
        偏移量

    返回:
    -----
    np.ndarray
        计算结果
    """
    return A * (1.0 - np.exp(-t / tau)) + C


@jit(nopython=True, cache=True)
def exp_decay_numba(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """
    指数衰减函数（Numba 编译版本）

    I(t) = A * exp(-t / tau) + C

    参数:
    -----
    t : np.ndarray
        时间数组
    A : float
        振幅
    tau : float
        时间常数
    C : float
        偏移量

    返回:
    -----
    np.ndarray
        计算结果
    """
    return A * np.exp(-t / tau) + C


@jit(nopython=True, cache=True)
def compute_r_squared_numba(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    """
    计算 R²（Numba 编译版本）

    R² = 1 - (SS_res / SS_tot)
    其中:
        SS_res = Σ(y_data - y_fit)²
        SS_tot = Σ(y_data - mean(y_data))²

    参数:
    -----
    y_data : np.ndarray
        实际数据
    y_fit : np.ndarray
        拟合数据

    返回:
    -----
    float
        R² 值
    """
    # 计算残差平方和
    ss_res = np.sum((y_data - y_fit) ** 2)

    # 计算总平方和
    y_mean = np.mean(y_data)
    ss_tot = np.sum((y_data - y_mean) ** 2)

    # 计算 R²
    if ss_tot == 0:
        return 0.0

    r_squared = 1.0 - (ss_res / ss_tot)
    return r_squared


@jit(nopython=True, cache=True)
def compute_adjusted_r_squared_numba(y_data: np.ndarray,
                                      y_fit: np.ndarray,
                                      n_params: int) -> float:
    """
    计算调整 R²（Numba 编译版本）

    Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]
    其中:
        n = 数据点数
        p = 参数数量

    参数:
    -----
    y_data : np.ndarray
        实际数据
    y_fit : np.ndarray
        拟合数据
    n_params : int
        拟合参数数量

    返回:
    -----
    float
        调整 R² 值
    """
    n = len(y_data)

    if n <= n_params + 1:
        return 0.0

    r_squared = compute_r_squared_numba(y_data, y_fit)
    adjusted_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / (n - n_params - 1)

    return adjusted_r_squared


@jit(nopython=True, cache=True)
def batch_exp_rise_numba(t_batch: np.ndarray,
                         params_batch: np.ndarray) -> np.ndarray:
    """
    批量计算指数上升（Numba 编译版本）

    参数:
    -----
    t_batch : np.ndarray
        时间数组（1D）
    params_batch : np.ndarray
        参数数组（N×3）：[[A1, tau1, C1], [A2, tau2, C2], ...]

    返回:
    -----
    np.ndarray
        结果数组（N×len(t_batch)）
    """
    n_batch = params_batch.shape[0]
    n_points = len(t_batch)
    results = np.empty((n_batch, n_points), dtype=np.float64)

    for i in range(n_batch):
        A = params_batch[i, 0]
        tau = params_batch[i, 1]
        C = params_batch[i, 2]
        results[i, :] = exp_rise_numba(t_batch, A, tau, C)

    return results


@jit(nopython=True, cache=True)
def batch_r_squared_numba(y_data_batch: np.ndarray,
                           y_fit_batch: np.ndarray) -> np.ndarray:
    """
    批量计算 R²（Numba 编译版本）

    参数:
    -----
    y_data_batch : np.ndarray
        实际数据批次（N×M）
    y_fit_batch : np.ndarray
        拟合数据批次（N×M）

    返回:
    -----
    np.ndarray
        R² 数组（N）
    """
    n_batch = y_data_batch.shape[0]
    r_squared_array = np.empty(n_batch, dtype=np.float64)

    for i in range(n_batch):
        r_squared_array[i] = compute_r_squared_numba(y_data_batch[i], y_fit_batch[i])

    return r_squared_array


# ============================================================================
# Fallback Functions (Pure NumPy, if Numba not available)
# ============================================================================

def exp_rise_numpy(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """指数上升函数（NumPy 版本）"""
    return A * (1.0 - np.exp(-t / tau)) + C


def exp_decay_numpy(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    """指数衰减函数（NumPy 版本）"""
    return A * np.exp(-t / tau) + C


def compute_r_squared_numpy(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    """计算 R²（NumPy 版本）"""
    ss_res = np.sum((y_data - y_fit) ** 2)
    y_mean = np.mean(y_data)
    ss_tot = np.sum((y_data - y_mean) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


# ============================================================================
# Unified API (automatically selects Numba or NumPy)
# ============================================================================

if NUMBA_AVAILABLE:
    exp_rise = exp_rise_numba
    exp_decay = exp_decay_numba
    compute_r_squared = compute_r_squared_numba
    compute_adjusted_r_squared = compute_adjusted_r_squared_numba
    batch_exp_rise = batch_exp_rise_numba
    batch_r_squared = batch_r_squared_numba
    print("✓ Numba acceleration enabled for autotau")
else:
    exp_rise = exp_rise_numpy
    exp_decay = exp_decay_numpy
    compute_r_squared = compute_r_squared_numpy

    def compute_adjusted_r_squared(y_data, y_fit, n_params):
        n = len(y_data)
        if n <= n_params + 1:
            return 0.0
        r_squared = compute_r_squared_numpy(y_data, y_fit)
        return 1.0 - (1.0 - r_squared) * (n - 1) / (n - n_params - 1)

    batch_exp_rise = None  # Not available without Numba
    batch_r_squared = None

    print("⚠️  Numba not available, using pure NumPy (slower)")


# ============================================================================
# Utility Functions
# ============================================================================

def is_numba_available() -> bool:
    """检查 Numba 是否可用"""
    return NUMBA_AVAILABLE


def get_acceleration_status() -> str:
    """获取加速状态字符串"""
    if NUMBA_AVAILABLE:
        return "Numba JIT (5-10x speedup)"
    else:
        return "Pure NumPy (no acceleration)"
