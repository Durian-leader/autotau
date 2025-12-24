"""
Fitting controller for AutoTau GUI.

Handles fitting operations in background threads using standalone fitters.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from ..fitters import (
    AutoCyclesFitter,
    CyclesFitter,
    WindowSearcher,
    WindowParameters
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

    This worker uses standalone fitters that don't depend on the core module,
    avoiding multiprocessing and matplotlib conflicts.

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
    windows_found = pyqtSignal(dict)

    # Internal signals for triggering work
    _do_auto = pyqtSignal(object, object, object)  # time, signal, params
    _do_manual = pyqtSignal(object, object, object, object)  # time, signal, params, window_params
    _do_search = pyqtSignal(object, object, object)  # time, signal, params

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False
        self._last_fitter = None
        self._last_windows = None

        # Connect internal signals to slots
        self._do_auto.connect(self._run_auto_mode)
        self._do_manual.connect(self._run_manual_mode)
        self._do_search.connect(self._run_window_search)

    def cancel(self):
        """Request cancellation of current operation."""
        self._is_cancelled = True

    def request_auto(self, time: np.ndarray, signal: np.ndarray, params: FittingParameters):
        """Request auto mode fitting (thread-safe)."""
        self._do_auto.emit(time, signal, params)

    def request_manual(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters,
        window_params: ManualWindowParameters
    ):
        """Request manual mode fitting (thread-safe)."""
        self._do_manual.emit(time, signal, params, window_params)

    def request_search(self, time: np.ndarray, signal: np.ndarray, params: FittingParameters):
        """Request window search (thread-safe)."""
        self._do_search.emit(time, signal, params)

    @pyqtSlot(object, object, object)
    def _run_auto_mode(self, time: np.ndarray, signal: np.ndarray, params: FittingParameters):
        """
        Run automatic fitting with AutoCyclesFitter.

        Args:
            time: Time array
            signal: Signal array
            params: Fitting parameters
        """
        self._is_cancelled = False
        self.started.emit()

        try:
            # Create auto fitter
            fitter = AutoCyclesFitter(
                time=time,
                signal=signal,
                period=params.period,
                sample_rate=params.sample_rate,
                normalize=params.normalize,
                r_squared_threshold=params.r_squared_threshold
            )

            def progress_cb(current, total):
                if not self._is_cancelled:
                    self.progress.emit(current, total)

            # Fit all cycles
            windows, results = fitter.fit_all(progress_callback=progress_cb)

            if self._is_cancelled:
                return

            # Get summary data
            df = fitter.get_summary_dataframe()

            # Emit window parameters found
            window_dict = fitter.get_window_dict()
            if window_dict:
                self.windows_found.emit(window_dict)

            # Store for later use
            self._last_fitter = fitter
            self._last_windows = windows

            self.finished.emit(df)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    @pyqtSlot(object, object, object, object)
    def _run_manual_mode(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters,
        window_params: ManualWindowParameters
    ):
        """
        Run manual fitting with CyclesFitter.

        Args:
            time: Time array
            signal: Signal array
            params: Common fitting parameters
            window_params: Window parameters
        """
        self._is_cancelled = False
        self.started.emit()

        try:
            # Convert window parameters
            windows = WindowParameters(
                on_offset=window_params.window_on_offset,
                on_size=window_params.window_on_size,
                off_offset=window_params.window_off_offset,
                off_size=window_params.window_off_size
            )

            # Create fitter
            fitter = CyclesFitter(
                time=time,
                signal=signal,
                period=params.period,
                sample_rate=params.sample_rate,
                windows=windows,
                normalize=params.normalize,
                r_squared_threshold=params.r_squared_threshold
            )

            def progress_cb(current, total):
                if not self._is_cancelled:
                    self.progress.emit(current, total)

            if self._is_cancelled:
                return

            # Fit all cycles
            fitter.fit_all(progress_callback=progress_cb)

            # Get summary data as DataFrame
            df = fitter.get_summary_dataframe()

            # Store for later use
            self._last_fitter = fitter
            self._last_windows = windows

            self.finished.emit(df)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    @pyqtSlot(object, object, object)
    def _run_window_search(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters
    ):
        """
        Search for optimal windows using WindowSearcher.

        Args:
            time: Time array
            signal: Signal array
            params: Fitting parameters
        """
        self._is_cancelled = False
        self.started.emit()

        try:
            # Extract first two periods for window search
            two_period_mask = time <= time[0] + 2 * params.period
            time_subset = time[two_period_mask]
            signal_subset = signal[two_period_mask]

            # Create searcher
            searcher = WindowSearcher(
                time_subset,
                signal_subset,
                params.period,
                params.sample_rate,
                normalize=params.normalize
            )

            def progress_cb(current, total):
                if not self._is_cancelled:
                    self.progress.emit(current, total)

            if self._is_cancelled:
                return

            # Search for best windows
            windows = searcher.search(progress_callback=progress_cb)

            # Convert to window parameters format
            on_r2 = searcher.best_on_result.r_squared if searcher.best_on_result else 0
            off_r2 = searcher.best_off_result.r_squared if searcher.best_off_result else 0

            window_dict = {
                'on_offset': windows.on_offset,
                'on_size': windows.on_size,
                'off_offset': windows.off_offset,
                'off_size': windows.off_size,
                'on_r2': on_r2,
                'off_r2': off_r2,
            }

            self._last_windows = windows

            self.windows_found.emit(window_dict)
            self.finished.emit(None)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    def get_last_fitter(self):
        """Get the last used fitter for plotting."""
        return self._last_fitter

    def get_last_windows(self):
        """Get the last found windows."""
        return self._last_windows


class FittingController(QObject):
    """
    Controller for managing fitting operations.

    Handles thread management and signal routing.
    Uses proper QThread pattern with worker moved to thread.
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
        """Set up worker and thread with proper signal connections."""
        if self._thread is not None:
            self.cancel()

        # Create thread and worker
        self._thread = QThread()
        self._worker = FittingWorker()

        # Move worker to thread
        self._worker.moveToThread(self._thread)

        # Connect worker signals to controller signals
        self._worker.started.connect(self.fitting_started)
        self._worker.progress.connect(self.fitting_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.windows_found.connect(self.windows_found)

        # Start the thread
        self._thread.start()

    def _on_finished(self, df):
        """Handle fitting completion."""
        self._last_results = df
        if self._worker:
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
        # Use worker's request method which emits internal signal
        self._worker.request_auto(time, signal, params)

    def run_manual(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters,
        window_params: ManualWindowParameters
    ):
        """Run manual fitting."""
        self._setup_worker()
        self._worker.request_manual(time, signal, params, window_params)

    def search_windows(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        params: FittingParameters
    ):
        """Search for optimal windows."""
        self._setup_worker()
        self._worker.request_search(time, signal, params)

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
