"""
Main window for AutoTau GUI application.
"""

import numpy as np
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTabWidget, QMessageBox, QSplitter, QStatusBar,
    QLabel
)

from .widgets.sidebar import Sidebar
from .widgets.data_input import DataInputWidget
from .widgets.plot_widgets import ResultsView
from .controllers.fitting_controller import (
    FittingController, FittingParameters, ManualWindowParameters
)
from .controllers.data_controller import DataController


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoTau - Time Constant Fitting Tool")
        self.setMinimumSize(1200, 800)

        # Controllers
        self.data_controller = DataController(self)
        self.fitting_controller = FittingController(self)

        # State
        self._current_time = None
        self._current_signal = None
        self._current_results = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the main window UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar()
        main_layout.addWidget(self.sidebar)

        # Main content area
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)

        # Tab widget for data input and results
        self.tabs = QTabWidget()

        # Data input tab
        self.data_input = DataInputWidget()
        self.tabs.addTab(self.data_input, "Data Input")

        # Results tab
        self.results_view = ResultsView()
        self.tabs.addTab(self.results_view, "Results")

        content_layout.addWidget(self.tabs)

        main_layout.addWidget(content_widget, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self):
        """Connect signals and slots."""
        # Data controller signals
        self.data_controller.data_loaded.connect(self._on_data_loaded)
        self.data_controller.error.connect(self._on_data_error)
        self.data_controller.estimates_ready.connect(self._on_estimates_ready)
        self.data_controller.data_cleared.connect(self._on_data_cleared)

        # Fitting controller signals
        self.fitting_controller.fitting_started.connect(self._on_fitting_started)
        self.fitting_controller.fitting_completed.connect(self._on_fitting_completed)
        self.fitting_controller.fitting_error.connect(self._on_fitting_error)
        self.fitting_controller.windows_found.connect(self._on_windows_found)

        # Sidebar signals
        self.sidebar.fit_clicked.connect(self._on_fit_clicked)
        self.sidebar.export_clicked.connect(self._on_export_clicked)
        self.sidebar.clear_clicked.connect(self._on_clear_clicked)
        self.sidebar.mode_changed.connect(self._on_mode_changed)

        # Data input signals
        self.data_input.parse_requested.connect(self._on_parse_clipboard)
        self.data_input.load_file_requested.connect(self._on_load_file)
        self.data_input.parse_text_button.clicked.connect(self._on_parse_text)

    def _on_parse_clipboard(self):
        """Handle parse clipboard request."""
        self.status_bar.showMessage("Parsing clipboard data...")
        self.data_controller.load_from_clipboard()

    def _on_parse_text(self):
        """Handle parse text request."""
        text = self.data_input.get_input_text()
        if text.strip():
            self.status_bar.showMessage("Parsing input text...")
            self.data_controller.load_from_text(text)
        else:
            QMessageBox.warning(self, "No Data", "Please paste data in the input area first.")

    def _on_load_file(self, filepath: str):
        """Handle load file request."""
        try:
            with open(filepath, 'r') as f:
                text = f.read()
            self.data_input.set_input_text(text)
            self.status_bar.showMessage(f"Loaded file: {filepath}")
            self.data_controller.load_from_text(text)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {e}")

    def _on_data_loaded(self, time: np.ndarray, signal: np.ndarray):
        """Handle data loaded event."""
        self._current_time = time
        self._current_signal = signal

        # Update preview
        self.data_input.update_preview(time, signal)

        # Update raw data plot
        period_str = self.sidebar.get_period()
        period = float(period_str) if period_str else None

        window_params = None
        if self.sidebar.get_mode() == 'manual' and period:
            try:
                window_params = {
                    'on_offset': float(self.sidebar.get_on_offset() or 0),
                    'on_size': float(self.sidebar.get_on_size() or 0),
                    'off_offset': float(self.sidebar.get_off_offset() or 0),
                    'off_size': float(self.sidebar.get_off_size() or 0),
                }
            except ValueError:
                pass

        self.results_view.update_raw_plot(time, signal, period, window_params)

        # Enable fitting
        self.sidebar.set_fitting_enabled(True)

        self.status_bar.showMessage(f"Loaded {len(time)} data points")

    def _on_data_error(self, message: str):
        """Handle data error."""
        QMessageBox.warning(self, "Data Error", message)
        self.status_bar.showMessage("Data loading failed")

    def _on_estimates_ready(self, estimates: dict):
        """Handle parameter estimates."""
        # Auto-fill sample rate if estimated
        if estimates.get('sample_rate') is not None:
            if not self.sidebar.get_sample_rate():
                self.sidebar.set_sample_rate(estimates['sample_rate'])

        # Show estimates in status
        info_parts = []
        if estimates.get('num_points'):
            info_parts.append(f"{estimates['num_points']} points")
        if estimates.get('sample_rate'):
            info_parts.append(f"~{estimates['sample_rate']:.0f} Hz")
        if estimates.get('period'):
            info_parts.append(f"estimated period: {estimates['period']:.4g} s")

        if info_parts:
            self.sidebar.set_status(" | ".join(info_parts))

    def _on_data_cleared(self):
        """Handle data cleared event."""
        self._current_time = None
        self._current_signal = None
        self._current_results = None
        self.sidebar.set_fitting_enabled(False)
        self.sidebar.set_export_enabled(False)
        self.status_bar.showMessage("Data cleared")

    def _on_fit_clicked(self):
        """Handle fit button click."""
        if self._current_time is None or self._current_signal is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        # Validate common parameters
        is_valid, error = self.sidebar.validate_common_params()
        if not is_valid:
            QMessageBox.warning(self, "Invalid Parameters", error)
            return

        # Get parameters
        period = float(self.sidebar.get_period())
        sample_rate = float(self.sidebar.get_sample_rate())
        r2_threshold = float(self.sidebar.get_r2_threshold())
        normalize = self.sidebar.get_normalize()

        params = FittingParameters(
            period=period,
            sample_rate=sample_rate,
            normalize=normalize,
            r_squared_threshold=r2_threshold
        )

        # Check mode
        mode = self.sidebar.get_mode()

        if mode == 'manual':
            # Validate manual parameters
            is_valid, error = self.sidebar.validate_manual_params(period)
            if not is_valid:
                QMessageBox.warning(self, "Invalid Window Parameters", error)
                return

            window_params = ManualWindowParameters(
                window_on_offset=float(self.sidebar.get_on_offset()),
                window_on_size=float(self.sidebar.get_on_size()),
                window_off_offset=float(self.sidebar.get_off_offset()),
                window_off_size=float(self.sidebar.get_off_size())
            )

            self.fitting_controller.run_manual(
                self._current_time,
                self._current_signal,
                params,
                window_params
            )
        else:
            # Auto mode
            self.fitting_controller.run_auto(
                self._current_time,
                self._current_signal,
                params
            )

    def _on_fitting_started(self):
        """Handle fitting started event."""
        self.sidebar.set_fitting_enabled(False)
        self.sidebar.show_progress(True)
        self.status_bar.showMessage("Fitting in progress...")
        self.sidebar.set_status("Fitting...")

    def _on_fitting_completed(self, df):
        """Handle fitting completed event."""
        self._current_results = df
        self.sidebar.show_progress(False)
        self.sidebar.set_fitting_enabled(True)

        if df is not None and not df.empty:
            # Update results
            r2_threshold = float(self.sidebar.get_r2_threshold())
            self.results_view.update_results(df, r2_threshold)

            # Update raw plot with windows
            period = float(self.sidebar.get_period())

            # Get window params from results if available
            if self.sidebar.get_mode() == 'manual':
                window_params = {
                    'on_offset': float(self.sidebar.get_on_offset()),
                    'on_size': float(self.sidebar.get_on_size()),
                    'off_offset': float(self.sidebar.get_off_offset()),
                    'off_size': float(self.sidebar.get_off_size()),
                }
            else:
                # Try to get from auto-detected windows
                window_params = getattr(self, '_detected_windows', None)

            if window_params:
                self.results_view.update_raw_plot(
                    self._current_time,
                    self._current_signal,
                    period,
                    window_params
                )

            # Enable export
            self.sidebar.set_export_enabled(True)

            # Switch to results tab
            self.tabs.setCurrentIndex(1)

            # Update status
            n_cycles = len(df)
            self.status_bar.showMessage(f"Fitting completed: {n_cycles} cycles processed")
            self.sidebar.set_status(f"Completed: {n_cycles} cycles")
        else:
            self.status_bar.showMessage("Fitting completed (no results)")

    def _on_fitting_error(self, message: str):
        """Handle fitting error."""
        self.sidebar.show_progress(False)
        self.sidebar.set_fitting_enabled(True)
        QMessageBox.critical(self, "Fitting Error", message)
        self.status_bar.showMessage("Fitting failed")
        self.sidebar.set_status(f"Error: {message[:50]}...")

    def _on_windows_found(self, windows: dict):
        """Handle window parameters found (auto mode)."""
        self._detected_windows = windows

        # Auto-fill manual parameters for reference
        self.sidebar.set_window_params(
            on_offset=windows.get('on_offset', 0),
            on_size=windows.get('on_size', 0),
            off_offset=windows.get('off_offset', 0),
            off_size=windows.get('off_size', 0)
        )

        # Show in status
        on_r2 = windows.get('on_r2', 0)
        off_r2 = windows.get('off_r2', 0)
        self.sidebar.set_status(
            f"Windows found: On R2={on_r2:.3f}, Off R2={off_r2:.3f}"
        )

    def _on_export_clicked(self):
        """Handle export button click."""
        if self._current_results is None or self._current_results.empty:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return

        # The ResultsTableWidget handles export dialogs
        self.results_view.results_table._export_csv()

    def _on_clear_clicked(self):
        """Handle clear button click."""
        reply = QMessageBox.question(
            self, "Clear All",
            "Are you sure you want to clear all data and results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.data_controller.clear()
            self.data_input.clear()
            self.results_view.clear()
            self._detected_windows = None
            self.sidebar.set_status("")

    def _on_mode_changed(self, mode: str):
        """Handle mode change."""
        # Update raw plot if data is loaded
        if self._current_time is not None:
            period_str = self.sidebar.get_period()
            period = float(period_str) if period_str else None

            window_params = None
            if mode == 'manual' and period:
                try:
                    window_params = {
                        'on_offset': float(self.sidebar.get_on_offset() or 0),
                        'on_size': float(self.sidebar.get_on_size() or 0),
                        'off_offset': float(self.sidebar.get_off_offset() or 0),
                        'off_size': float(self.sidebar.get_off_size() or 0),
                    }
                except ValueError:
                    pass
            elif mode == 'auto':
                window_params = getattr(self, '_detected_windows', None)

            self.results_view.update_raw_plot(
                self._current_time,
                self._current_signal,
                period,
                window_params
            )
