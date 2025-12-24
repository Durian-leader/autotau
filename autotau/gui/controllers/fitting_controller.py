"""
Fitting controller for AutoTau GUI.

Handles fitting operations in background threads.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from autotau import (
    CyclesAutoTauFitter,
    CyclesTauFitter,
    AutoTauFitter
)


@dataclass
class FittingParameters:
    """Common fitting parameters."""
    period: float
    sample_rate: float
    normalize: bool = False
    r_squared_threshold: float = 0.95
    language: str = 'cn'
    max_workers: Optional[int] = None


@dataclass
class ManualWindowParameters:
    """Manual mode window parameters."""
    window_on_offset: float
    window_on_size: float
    window_off_offset: float
    window_off_size: float


class FittingWorker(QObject):
    """
    Worker for running fitting operations in background.

    Signals:
        started: Emitted when fitting starts
        progress: Emitted with (current, total) progress
        finished: Emitted with DataFrame results
        error: Emitted with error message
        windows_found: Emitted when auto mode finds optimal windows
    """
    started = pyqtSignal()
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)  # pd.DataFrame
    error = pyqtSignal(str)
    windows_found = pyqtSignal(dict)  # Window parameters

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False

    def cancel(self):
        """Request cancellation of current operation."""
        self._is_cancelled = True

    def run_auto_mode(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters
    ):
        """
        Run automatic fitting with CyclesAutoTauFitter.

        Args:
            time: Time array
            signal: Signal array
            params: Fitting parameters
        """
        self._is_cancelled = False
        self.started.emit()

        try:
            sample_step = 1.0 / params.sample_rate

            # Create fitter
            fitter = CyclesAutoTauFitter(
                time=time,
                signal=signal,
                period=params.period,
                sample_rate=params.sample_rate,
                normalize=params.normalize,
                language=params.language,
                show_progress=False
            )

            # Fit all cycles
            results = fitter.fit_all_cycles(
                r_squared_threshold=params.r_squared_threshold
            )

            if self._is_cancelled:
                return

            # Get summary data
            df = fitter.get_summary_data()

            # Emit window parameters found
            if hasattr(fitter, 'auto_fitter') and fitter.auto_fitter is not None:
                auto = fitter.auto_fitter
                windows = {
                    'on_offset': auto.best_tau_on_window_start_time - time[0],
                    'on_size': auto.best_tau_on_window_size,
                    'off_offset': auto.best_tau_off_window_start_time - time[0],
                    'off_size': auto.best_tau_off_window_size,
                }
                self.windows_found.emit(windows)

            # Store fitter for later use
            self._last_fitter = fitter

            self.finished.emit(df)

        except Exception as e:
            self.error.emit(str(e))

    def run_manual_mode(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters,
        window_params: ManualWindowParameters
    ):
        """
        Run manual fitting with CyclesTauFitter (serial, no multiprocessing).

        Args:
            time: Time array
            signal: Signal array
            params: Common fitting parameters
            window_params: Window parameters
        """
        self._is_cancelled = False
        self.started.emit()

        try:
            # Create fitter (serial version to avoid multiprocessing issues in GUI)
            fitter = CyclesTauFitter(
                time=time,
                signal=signal,
                period=params.period,
                sample_rate=params.sample_rate,
                window_on_offset=window_params.window_on_offset,
                window_on_size=window_params.window_on_size,
                window_off_offset=window_params.window_off_offset,
                window_off_size=window_params.window_off_size,
                normalize=params.normalize,
                language=params.language
            )

            if self._is_cancelled:
                return

            # Fit all cycles
            fitter.fit_all_cycles()

            # Get summary data as DataFrame
            df = fitter.get_summary_data()

            # Store fitter for later use
            self._last_fitter = fitter

            self.finished.emit(df)

        except Exception as e:
            self.error.emit(str(e))

    def run_window_search(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters
    ):
        """
        Search for optimal windows using AutoTauFitter (serial, no multiprocessing).

        Args:
            time: Time array
            signal: Signal array
            params: Fitting parameters
        """
        self._is_cancelled = False
        self.started.emit()

        try:
            sample_step = 1.0 / params.sample_rate

            # Use AutoTauFitter for window search (serial mode, no executor)
            auto_fitter = AutoTauFitter(
                time=time,
                signal=signal,
                sample_step=sample_step,
                period=params.period,
                normalize=params.normalize,
                language=params.language,
                show_progress=False,
                executor=None  # Serial mode to avoid multiprocessing issues
            )

            if self._is_cancelled:
                return

            # Find best windows by fitting
            auto_fitter.fit_tau_on_and_off()

            # Convert to window parameters format
            window_dict = {
                'on_offset': auto_fitter.best_tau_on_window_start_time - time[0],
                'on_size': auto_fitter.best_tau_on_window_size,
                'off_offset': auto_fitter.best_tau_off_window_start_time - time[0],
                'off_size': auto_fitter.best_tau_off_window_size,
                'on_r2': auto_fitter.best_tau_on_fitter.tau_on_r_squared_adj if auto_fitter.best_tau_on_fitter else 0,
                'off_r2': auto_fitter.best_tau_off_fitter.tau_off_r_squared_adj if auto_fitter.best_tau_off_fitter else 0,
            }

            self.windows_found.emit(window_dict)
            self.finished.emit(None)

        except Exception as e:
            self.error.emit(str(e))

    def get_last_fitter(self):
        """Get the last used fitter for plotting."""
        return getattr(self, '_last_fitter', None)


class FittingController(QObject):
    """
    Controller for managing fitting operations.

    Handles thread management and signal routing.
    """
    fitting_started = pyqtSignal()
    fitting_progress = pyqtSignal(int, int)
    fitting_completed = pyqtSignal(object)  # pd.DataFrame
    fitting_error = pyqtSignal(str)
    windows_found = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._worker = None
        self._last_results = None
        self._last_fitter = None

    def _setup_worker(self):
        """Set up worker and thread."""
        if self._thread is not None:
            self.cancel()

        self._thread = QThread()
        self._worker = FittingWorker()
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._worker.started.connect(self.fitting_started)
        self._worker.progress.connect(self.fitting_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.windows_found.connect(self.windows_found)

        self._thread.start()

    def _on_finished(self, df):
        """Handle fitting completion."""
        self._last_results = df
        self._last_fitter = self._worker.get_last_fitter()
        self.fitting_completed.emit(df)
        self._cleanup()

    def _on_error(self, message):
        """Handle fitting error."""
        self.fitting_error.emit(message)
        self._cleanup()

    def _cleanup(self):
        """Clean up thread."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def run_auto(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters
    ):
        """Run automatic fitting."""
        self._setup_worker()
        # Use QTimer.singleShot to call in the thread
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._worker.run_auto_mode(time, signal, params))

    def run_manual(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters,
        window_params: ManualWindowParameters
    ):
        """Run manual fitting."""
        self._setup_worker()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._worker.run_manual_mode(time, signal, params, window_params))

    def search_windows(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters
    ):
        """Search for optimal windows."""
        self._setup_worker()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self._worker.run_window_search(time, signal, params))

    def cancel(self):
        """Cancel current operation."""
        if self._worker is not None:
            self._worker.cancel()
        self._cleanup()

    def get_last_results(self) -> Optional[pd.DataFrame]:
        """Get last fitting results."""
        return self._last_results

    def get_last_fitter(self):
        """Get last fitter for plotting."""
        return self._last_fitter

    def is_running(self) -> bool:
        """Check if fitting is running."""
        return self._thread is not None and self._thread.isRunning()
