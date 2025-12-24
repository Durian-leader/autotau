"""
Standalone fitting module for AutoTau GUI.

This module provides all fitting logic independently from the core module,
ensuring reliable execution in GUI context without multiprocessing issues.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class FitResult:
    """Result of a single exponential fit."""
    tau: float
    amplitude: float
    offset: float
    r_squared: float
    r_squared_adj: float
    popt: np.ndarray
    pcov: Optional[np.ndarray] = None


@dataclass
class CycleFitResult:
    """Result of fitting a single cycle."""
    cycle: int
    cycle_start_time: float
    tau_on: float
    tau_off: float
    r_squared_on: float
    r_squared_off: float
    r_squared_adj_on: float
    r_squared_adj_off: float
    on_fit: Optional[FitResult] = None
    off_fit: Optional[FitResult] = None
    was_refitted: bool = False


@dataclass
class WindowParameters:
    """Window parameters for fitting."""
    on_offset: float
    on_size: float
    off_offset: float
    off_size: float


class ExponentialFitter:
    """
    Simple exponential fitter for tau extraction.

    Provides static methods for exponential rise/decay fitting
    without any matplotlib or multiprocessing dependencies.
    """

    @staticmethod
    def exp_rise(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
        """Exponential rise: y = A * (1 - exp(-t/tau)) + C"""
        return A * (1 - np.exp(-t / tau)) + C

    @staticmethod
    def exp_decay(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
        """Exponential decay: y = A * exp(-t/tau) + C"""
        return A * np.exp(-t / tau) + C

    @staticmethod
    def compute_r_squared(
        x_data: np.ndarray,
        y_data: np.ndarray,
        popt: np.ndarray,
        func
    ) -> Tuple[float, float]:
        """Compute R-squared and adjusted R-squared."""
        y_fit = func(x_data, *popt)
        y_mean = np.mean(y_data)

        # RSS (Residual Sum of Squares)
        rss = np.sum((y_data - y_fit) ** 2)
        # TSS (Total Sum of Squares)
        tss = np.sum((y_data - y_mean) ** 2)

        # R-squared
        if tss < 1e-10:
            r_squared = 0.0
        else:
            r_squared = 1 - (rss / tss)

        # Adjusted R-squared
        n = len(y_data)
        p = len(popt)
        if n - p - 1 > 0:
            r_squared_adj = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        else:
            r_squared_adj = r_squared

        return r_squared, r_squared_adj

    @staticmethod
    def normalize_signal(signal: np.ndarray) -> np.ndarray:
        """Normalize signal to 0-1 range."""
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        if signal_max - signal_min > 1e-10:
            return (signal - signal_min) / (signal_max - signal_min)
        return np.zeros_like(signal)

    @classmethod
    def fit_rise(
        cls,
        time: np.ndarray,
        signal: np.ndarray,
        normalize: bool = False,
        interp: bool = True,
        points_after_interp: int = 100
    ) -> Optional[FitResult]:
        """
        Fit exponential rise to data.

        Args:
            time: Time array (will be shifted to start from 0)
            signal: Signal array
            normalize: Whether to normalize signal to 0-1
            interp: Whether to interpolate before fitting
            points_after_interp: Number of points after interpolation

        Returns:
            FitResult or None if fitting fails
        """
        if len(time) < 3:
            return None

        t = time - time[0]
        s = signal.copy()

        if normalize:
            s = cls.normalize_signal(s)

        if interp:
            t_dense = np.linspace(t[0], t[-1], points_after_interp)
            s_dense = np.interp(t_dense, t, s)
        else:
            t_dense = t
            s_dense = s

        try:
            popt, pcov = curve_fit(
                cls.exp_rise,
                t_dense,
                s_dense,
                maxfev=100_000,
                bounds=((0, 0, -np.inf), (np.inf, np.inf, np.inf))
            )
            r_squared, r_squared_adj = cls.compute_r_squared(t, s, popt, cls.exp_rise)

            return FitResult(
                tau=popt[1],
                amplitude=popt[0],
                offset=popt[2],
                r_squared=r_squared,
                r_squared_adj=r_squared_adj,
                popt=popt,
                pcov=pcov
            )
        except (RuntimeError, ValueError):
            return None

    @classmethod
    def fit_decay(
        cls,
        time: np.ndarray,
        signal: np.ndarray,
        normalize: bool = False,
        interp: bool = True,
        points_after_interp: int = 100
    ) -> Optional[FitResult]:
        """
        Fit exponential decay to data.

        Args:
            time: Time array (will be shifted to start from 0)
            signal: Signal array
            normalize: Whether to normalize signal to 0-1
            interp: Whether to interpolate before fitting
            points_after_interp: Number of points after interpolation

        Returns:
            FitResult or None if fitting fails
        """
        if len(time) < 3:
            return None

        t = time - time[0]
        s = signal.copy()

        if normalize:
            s = cls.normalize_signal(s)

        if interp:
            t_dense = np.linspace(t[0], t[-1], points_after_interp)
            s_dense = np.interp(t_dense, t, s)
        else:
            t_dense = t
            s_dense = s

        try:
            popt, pcov = curve_fit(
                cls.exp_decay,
                t_dense,
                s_dense,
                maxfev=100_000,
                bounds=((0, 0, -np.inf), (np.inf, np.inf, np.inf))
            )
            r_squared, r_squared_adj = cls.compute_r_squared(t, s, popt, cls.exp_decay)

            return FitResult(
                tau=popt[1],
                amplitude=popt[0],
                offset=popt[2],
                r_squared=r_squared,
                r_squared_adj=r_squared_adj,
                popt=popt,
                pcov=pcov
            )
        except (RuntimeError, ValueError):
            return None


class WindowSearcher:
    """
    Search for optimal fitting windows in periodic signals.

    This is a simplified, GUI-friendly implementation that searches
    for the best fitting windows without multiprocessing.
    """

    def __init__(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        period: float,
        sample_rate: float,
        normalize: bool = False,
        window_scalar_min: float = 0.2,
        window_scalar_max: float = 0.33,
        window_points_step: int = 10,
        window_start_idx_step: int = 5
    ):
        """
        Initialize window searcher.

        Args:
            time: Time array
            signal: Signal array
            period: Signal period in seconds
            sample_rate: Sample rate in Hz
            normalize: Whether to normalize signal
            window_scalar_min: Minimum window size as fraction of period
            window_scalar_max: Maximum window size as fraction of period
            window_points_step: Step size for window size search
            window_start_idx_step: Step size for window start position search
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.normalize = normalize

        # Window search parameters
        self.window_scalar_min = window_scalar_min
        self.window_scalar_max = window_scalar_max
        self.window_points_step = window_points_step
        self.window_start_idx_step = window_start_idx_step

        # Results
        self.best_on_result: Optional[FitResult] = None
        self.best_off_result: Optional[FitResult] = None
        self.best_on_window_start: float = 0.0
        self.best_on_window_size: float = 0.0
        self.best_off_window_start: float = 0.0
        self.best_off_window_size: float = 0.0

    def search(self, progress_callback=None) -> WindowParameters:
        """
        Search for optimal windows.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            WindowParameters with optimal window configuration
        """
        sample_step = 1.0 / self.sample_rate
        half_period = self.period / 2

        # Calculate window size range in points
        min_window_points = int(self.period * self.window_scalar_min * self.sample_rate)
        max_window_points = int(self.period * self.window_scalar_max * self.sample_rate)

        # Generate window sizes to try
        window_sizes = list(range(min_window_points, max_window_points + 1, self.window_points_step))
        if not window_sizes:
            window_sizes = [min_window_points]

        # Calculate maximum start position for on/off windows
        max_on_start_idx = int(half_period * self.sample_rate)
        max_off_start_idx = int(half_period * self.sample_rate)

        best_on_r2 = -1.0
        best_off_r2 = -1.0

        # Total iterations for progress
        total_iterations = len(window_sizes) * (max_on_start_idx // self.window_start_idx_step + 1) * 2
        current_iteration = 0

        # Search for best on-transition window (first half of period)
        for window_points in window_sizes:
            window_size = window_points * sample_step

            for start_idx in range(0, max_on_start_idx, self.window_start_idx_step):
                current_iteration += 1
                if progress_callback and current_iteration % 10 == 0:
                    progress_callback(current_iteration, total_iterations)

                window_start = self.time[0] + start_idx * sample_step
                window_end = window_start + window_size

                # Ensure window is within first period
                if window_end > self.time[0] + half_period:
                    continue

                # Extract window data
                mask = (self.time >= window_start) & (self.time <= window_end)
                if np.sum(mask) < 5:
                    continue

                t_window = self.time[mask]
                s_window = self.signal[mask]

                # Try fitting
                result = ExponentialFitter.fit_rise(
                    t_window, s_window,
                    normalize=self.normalize
                )

                if result and result.r_squared > best_on_r2:
                    best_on_r2 = result.r_squared
                    self.best_on_result = result
                    self.best_on_window_start = window_start
                    self.best_on_window_size = window_size

        # Search for best off-transition window (second half of period)
        for window_points in window_sizes:
            window_size = window_points * sample_step

            for start_idx in range(0, max_off_start_idx, self.window_start_idx_step):
                current_iteration += 1
                if progress_callback and current_iteration % 10 == 0:
                    progress_callback(current_iteration, total_iterations)

                window_start = self.time[0] + half_period + start_idx * sample_step
                window_end = window_start + window_size

                # Ensure window is within period
                if window_end > self.time[0] + self.period:
                    continue

                # Extract window data
                mask = (self.time >= window_start) & (self.time <= window_end)
                if np.sum(mask) < 5:
                    continue

                t_window = self.time[mask]
                s_window = self.signal[mask]

                # Try fitting
                result = ExponentialFitter.fit_decay(
                    t_window, s_window,
                    normalize=self.normalize
                )

                if result and result.r_squared > best_off_r2:
                    best_off_r2 = result.r_squared
                    self.best_off_result = result
                    self.best_off_window_start = window_start
                    self.best_off_window_size = window_size

        # Calculate offsets relative to start time
        on_offset = (self.best_on_window_start - self.time[0]) % self.period
        off_offset = (self.best_off_window_start - self.time[0]) % self.period

        return WindowParameters(
            on_offset=on_offset,
            on_size=self.best_on_window_size,
            off_offset=off_offset,
            off_size=self.best_off_window_size
        )


class CyclesFitter:
    """
    Fit multiple cycles using specified window parameters.

    This is a simplified, GUI-friendly implementation that processes
    cycles sequentially without multiprocessing.
    """

    def __init__(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        period: float,
        sample_rate: float,
        windows: WindowParameters,
        normalize: bool = False,
        r_squared_threshold: float = 0.95
    ):
        """
        Initialize cycles fitter.

        Args:
            time: Time array
            signal: Signal array
            period: Signal period in seconds
            sample_rate: Sample rate in Hz
            windows: Window parameters for fitting
            normalize: Whether to normalize signal
            r_squared_threshold: Threshold for refitting
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.windows = windows
        self.normalize = normalize
        self.r_squared_threshold = r_squared_threshold

        # Results
        self.results: List[CycleFitResult] = []

    def fit_all(self, progress_callback=None) -> List[CycleFitResult]:
        """
        Fit all cycles.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of CycleFitResult for each cycle
        """
        # Calculate number of complete cycles
        total_time = self.time[-1] - self.time[0]
        num_cycles = int(total_time / self.period)

        self.results = []

        for i in range(num_cycles):
            if progress_callback:
                progress_callback(i + 1, num_cycles)

            cycle_start_time = self.time[0] + i * self.period

            # Calculate window times for this cycle
            on_start = cycle_start_time + self.windows.on_offset
            on_end = on_start + self.windows.on_size
            off_start = cycle_start_time + self.windows.off_offset
            off_end = off_start + self.windows.off_size

            # Extract on-transition data
            on_mask = (self.time >= on_start) & (self.time <= on_end)
            off_mask = (self.time >= off_start) & (self.time <= off_end)

            # Check if we have enough data
            if np.sum(on_mask) < 3 or np.sum(off_mask) < 3:
                continue

            # Fit on-transition
            on_fit = ExponentialFitter.fit_rise(
                self.time[on_mask],
                self.signal[on_mask],
                normalize=self.normalize
            )

            # Fit off-transition
            off_fit = ExponentialFitter.fit_decay(
                self.time[off_mask],
                self.signal[off_mask],
                normalize=self.normalize
            )

            # Handle failed fits
            if on_fit is None:
                tau_on = 0.0
                r2_on = 0.0
                r2_adj_on = 0.0
            else:
                tau_on = on_fit.tau
                r2_on = on_fit.r_squared
                r2_adj_on = on_fit.r_squared_adj

            if off_fit is None:
                tau_off = 0.0
                r2_off = 0.0
                r2_adj_off = 0.0
            else:
                tau_off = off_fit.tau
                r2_off = off_fit.r_squared
                r2_adj_off = off_fit.r_squared_adj

            result = CycleFitResult(
                cycle=i + 1,
                cycle_start_time=cycle_start_time,
                tau_on=tau_on,
                tau_off=tau_off,
                r_squared_on=r2_on,
                r_squared_off=r2_off,
                r_squared_adj_on=r2_adj_on,
                r_squared_adj_off=r2_adj_off,
                on_fit=on_fit,
                off_fit=off_fit,
                was_refitted=False
            )

            self.results.append(result)

        return self.results

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get results as a DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = {
            'cycle': [r.cycle for r in self.results],
            'cycle_start_time': [r.cycle_start_time for r in self.results],
            'tau_on': [r.tau_on for r in self.results],
            'tau_off': [r.tau_off for r in self.results],
            'r_squared_on': [r.r_squared_on for r in self.results],
            'r_squared_off': [r.r_squared_off for r in self.results],
            'r_squared_adj_on': [r.r_squared_adj_on for r in self.results],
            'r_squared_adj_off': [r.r_squared_adj_off for r in self.results],
            'was_refitted': [r.was_refitted for r in self.results]
        }

        return pd.DataFrame(data)


class AutoCyclesFitter:
    """
    Automatically find optimal windows and fit all cycles.

    Combines WindowSearcher and CyclesFitter for a complete
    auto-fitting workflow.
    """

    def __init__(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        period: float,
        sample_rate: float,
        normalize: bool = False,
        r_squared_threshold: float = 0.95,
        window_scalar_min: float = 0.2,
        window_scalar_max: float = 0.33
    ):
        """
        Initialize auto cycles fitter.

        Args:
            time: Time array
            signal: Signal array
            period: Signal period in seconds
            sample_rate: Sample rate in Hz
            normalize: Whether to normalize signal
            r_squared_threshold: Threshold for refitting
            window_scalar_min: Minimum window size as fraction of period
            window_scalar_max: Maximum window size as fraction of period
        """
        self.time = np.array(time)
        self.signal = np.array(signal)
        self.period = period
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.r_squared_threshold = r_squared_threshold
        self.window_scalar_min = window_scalar_min
        self.window_scalar_max = window_scalar_max

        # Results
        self.windows: Optional[WindowParameters] = None
        self.results: List[CycleFitResult] = []
        self.searcher: Optional[WindowSearcher] = None
        self.fitter: Optional[CyclesFitter] = None

    def fit_all(self, progress_callback=None) -> Tuple[WindowParameters, List[CycleFitResult]]:
        """
        Find optimal windows and fit all cycles.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Tuple of (WindowParameters, List[CycleFitResult])
        """
        # Extract first two periods for window search
        two_period_mask = self.time <= self.time[0] + 2 * self.period
        time_subset = self.time[two_period_mask]
        signal_subset = self.signal[two_period_mask]

        # Search for optimal windows
        self.searcher = WindowSearcher(
            time_subset,
            signal_subset,
            self.period,
            self.sample_rate,
            normalize=self.normalize,
            window_scalar_min=self.window_scalar_min,
            window_scalar_max=self.window_scalar_max
        )

        def search_progress(cur, total):
            if progress_callback:
                # Window search is roughly 30% of total work
                progress_callback(int(cur * 0.3), 100)

        self.windows = self.searcher.search(progress_callback=search_progress)

        # Fit all cycles using found windows
        self.fitter = CyclesFitter(
            self.time,
            self.signal,
            self.period,
            self.sample_rate,
            self.windows,
            normalize=self.normalize,
            r_squared_threshold=self.r_squared_threshold
        )

        def fit_progress(cur, total):
            if progress_callback:
                # Cycle fitting is roughly 70% of total work
                progress_callback(30 + int(cur / total * 70), 100)

        self.results = self.fitter.fit_all(progress_callback=fit_progress)

        if progress_callback:
            progress_callback(100, 100)

        return self.windows, self.results

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get results as a DataFrame."""
        if self.fitter:
            return self.fitter.get_summary_dataframe()
        return pd.DataFrame()

    def get_window_dict(self) -> Dict[str, float]:
        """Get windows as a dictionary."""
        if self.windows is None:
            return {}

        on_r2 = self.searcher.best_on_result.r_squared if self.searcher and self.searcher.best_on_result else 0
        off_r2 = self.searcher.best_off_result.r_squared if self.searcher and self.searcher.best_off_result else 0

        return {
            'on_offset': self.windows.on_offset,
            'on_size': self.windows.on_size,
            'off_offset': self.windows.off_offset,
            'off_size': self.windows.off_size,
            'on_r2': on_r2,
            'off_r2': off_r2
        }
