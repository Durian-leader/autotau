"""
Data controller for AutoTau GUI.

Manages data input, validation, and storage.
"""

import numpy as np
from typing import Optional, Tuple
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from ..utils.clipboard_parser import (
    parse_clipboard_data,
    estimate_sample_rate,
    estimate_period
)


class DataController(QObject):
    """
    Controller for managing input data.

    Signals:
        data_loaded: Emitted when data is successfully loaded
        data_cleared: Emitted when data is cleared
        error: Emitted when data loading fails
        estimates_ready: Emitted with estimated parameters
    """
    data_loaded = pyqtSignal(object, object)  # time, signal arrays
    data_cleared = pyqtSignal()
    error = pyqtSignal(str)
    estimates_ready = pyqtSignal(dict)  # Estimated parameters

    def __init__(self, parent=None):
        super().__init__(parent)
        self._time = None
        self._signal = None

    def load_from_clipboard(self) -> bool:
        """
        Load data from system clipboard.

        Returns:
            True if successful, False otherwise
        """
        try:
            clipboard = QApplication.clipboard()
            text = clipboard.text()

            if not text:
                self.error.emit("Clipboard is empty")
                return False

            time, signal = parse_clipboard_data(text)
            self._time = time
            self._signal = signal

            self.data_loaded.emit(time, signal)

            # Try to estimate parameters
            self._estimate_parameters()

            return True

        except ValueError as e:
            self.error.emit(str(e))
            return False
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
            return False

    def load_from_text(self, text: str) -> bool:
        """
        Load data from text string.

        Args:
            text: Two-column data text

        Returns:
            True if successful, False otherwise
        """
        try:
            time, signal = parse_clipboard_data(text)
            self._time = time
            self._signal = signal

            self.data_loaded.emit(time, signal)

            # Try to estimate parameters
            self._estimate_parameters()

            return True

        except ValueError as e:
            self.error.emit(str(e))
            return False

    def load_from_arrays(
        self,
        time: np.ndarray,
        signal: np.ndarray
    ) -> bool:
        """
        Load data from numpy arrays.

        Args:
            time: Time array
            signal: Signal array

        Returns:
            True if successful
        """
        if len(time) != len(signal):
            self.error.emit("Time and signal arrays must have same length")
            return False

        if len(time) < 10:
            self.error.emit("Need at least 10 data points")
            return False

        self._time = np.array(time)
        self._signal = np.array(signal)

        self.data_loaded.emit(self._time, self._signal)
        self._estimate_parameters()

        return True

    def _estimate_parameters(self):
        """Estimate sample rate and period from data."""
        if self._time is None:
            return

        estimates = {}

        # Estimate sample rate
        try:
            estimates['sample_rate'] = estimate_sample_rate(self._time)
        except Exception:
            estimates['sample_rate'] = None

        # Estimate period
        try:
            period = estimate_period(self._time, self._signal)
            if period is not None and period > 0:
                # Sanity check: period should be reasonable
                total_time = self._time[-1] - self._time[0]
                if period < total_time / 2:
                    estimates['period'] = period
                else:
                    estimates['period'] = None
            else:
                estimates['period'] = None
        except Exception:
            estimates['period'] = None

        # Data statistics
        estimates['num_points'] = len(self._time)
        estimates['time_range'] = (self._time[0], self._time[-1])
        estimates['signal_range'] = (self._signal.min(), self._signal.max())

        self.estimates_ready.emit(estimates)

    def clear(self):
        """Clear loaded data."""
        self._time = None
        self._signal = None
        self.data_cleared.emit()

    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self._time is not None and self._signal is not None

    def get_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get loaded data."""
        return self._time, self._signal

    def get_time(self) -> Optional[np.ndarray]:
        """Get time array."""
        return self._time

    def get_signal(self) -> Optional[np.ndarray]:
        """Get signal array."""
        return self._signal

    def get_num_points(self) -> int:
        """Get number of data points."""
        if self._time is None:
            return 0
        return len(self._time)

    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """Get time range."""
        if self._time is None:
            return None
        return (self._time[0], self._time[-1])

    def get_num_cycles(self, period: float) -> int:
        """
        Get estimated number of complete cycles.

        Args:
            period: Signal period

        Returns:
            Number of complete cycles
        """
        if self._time is None or period <= 0:
            return 0

        total_time = self._time[-1] - self._time[0]
        return int(total_time / period)
