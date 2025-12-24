"""
Matplotlib plot widgets for AutoTau GUI.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QTabWidget, QSplitter,
    QSpinBox, QComboBox
)

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure


class BasePlotWidget(QWidget):
    """Base class for matplotlib plot widgets."""

    def __init__(self, parent=None, figsize=(6, 4)):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.figure.tight_layout()

    def clear(self):
        """Clear the plot."""
        self.ax.clear()
        self.canvas.draw()


class RawDataPlotWidget(BasePlotWidget):
    """Widget for plotting raw signal data with window annotations."""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(8, 4))

    def plot_data(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        period: Optional[float] = None,
        window_params: Optional[Dict[str, float]] = None,
        num_cycles: int = 5
    ):
        """
        Plot raw signal with optional window annotations.

        Args:
            time: Time array
            signal: Signal array
            period: Signal period (for window annotations)
            window_params: Dict with on_offset, on_size, off_offset, off_size
            num_cycles: Number of cycles to show windows for
        """
        self.ax.clear()

        # Plot signal
        self.ax.plot(time, signal, 'b-', linewidth=0.5, label='Signal')

        # Add window annotations if parameters provided
        if period is not None and window_params is not None:
            on_offset = window_params.get('on_offset', 0)
            on_size = window_params.get('on_size', 0)
            off_offset = window_params.get('off_offset', 0)
            off_size = window_params.get('off_size', 0)

            # Calculate number of visible cycles
            time_range = time[-1] - time[0]
            total_cycles = int(time_range / period)
            show_cycles = min(num_cycles, total_cycles)

            for i in range(show_cycles):
                cycle_start = time[0] + i * period

                # On window
                if on_size > 0:
                    on_start = cycle_start + on_offset
                    on_end = on_start + on_size
                    self.ax.axvspan(
                        on_start, on_end,
                        alpha=0.2, color='green',
                        label='On Window' if i == 0 else None
                    )

                # Off window
                if off_size > 0:
                    off_start = cycle_start + off_offset
                    off_end = off_start + off_size
                    self.ax.axvspan(
                        off_start, off_end,
                        alpha=0.2, color='red',
                        label='Off Window' if i == 0 else None
                    )

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Signal')
        self.ax.set_title('Raw Signal Data')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def zoom_to_cycles(self, time: np.ndarray, period: float, start_cycle: int = 0, num_cycles: int = 5):
        """
        Zoom to show specific cycles.

        Args:
            time: Time array
            period: Signal period
            start_cycle: Starting cycle number
            num_cycles: Number of cycles to show
        """
        start_time = time[0] + start_cycle * period
        end_time = start_time + num_cycles * period
        self.ax.set_xlim(start_time, end_time)
        self.canvas.draw()


class TauEvolutionPlotWidget(BasePlotWidget):
    """Widget for plotting tau evolution over cycles."""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(8, 4))

    def plot_tau(self, df: pd.DataFrame, dual_axis: bool = True):
        """
        Plot tau values over cycles.

        Args:
            df: DataFrame with cycle results
            dual_axis: Whether to use dual Y-axis
        """
        self.ax.clear()

        if df is None or df.empty:
            self.canvas.draw()
            return

        # Get cycle numbers
        cycles = df['cycle'].values if 'cycle' in df.columns else np.arange(len(df))

        # Get tau values (handle different column names)
        tau_on = df.get('tau_on', df.get('tau_on_value', None))
        tau_off = df.get('tau_off', df.get('tau_off_value', None))

        if tau_on is not None:
            tau_on = tau_on.values
        if tau_off is not None:
            tau_off = tau_off.values

        # Plot tau_on
        color1 = 'tab:blue'
        self.ax.set_xlabel('Cycle')
        self.ax.set_ylabel('Tau On (s)', color=color1)
        if tau_on is not None:
            line1 = self.ax.plot(cycles, tau_on, 'o-', color=color1, label='Tau On', markersize=3)
            self.ax.tick_params(axis='y', labelcolor=color1)

        # Plot tau_off
        if dual_axis and tau_off is not None:
            ax2 = self.ax.twinx()
            color2 = 'tab:red'
            ax2.set_ylabel('Tau Off (s)', color=color2)
            line2 = ax2.plot(cycles, tau_off, 's-', color=color2, label='Tau Off', markersize=3)
            ax2.tick_params(axis='y', labelcolor=color2)

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            self.ax.legend(lines, labels, loc='upper right')
        elif tau_off is not None:
            self.ax.plot(cycles, tau_off, 's-', color='tab:red', label='Tau Off', markersize=3)
            self.ax.legend(loc='upper right')

        self.ax.set_title('Tau Evolution Over Cycles')
        self.ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


class RSquaredPlotWidget(BasePlotWidget):
    """Widget for plotting R-squared quality over cycles."""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(8, 4))

    def plot_r_squared(self, df: pd.DataFrame, threshold: float = 0.95):
        """
        Plot R-squared values over cycles.

        Args:
            df: DataFrame with cycle results
            threshold: R-squared threshold line
        """
        self.ax.clear()

        if df is None or df.empty:
            self.canvas.draw()
            return

        # Get cycle numbers
        cycles = df['cycle'].values if 'cycle' in df.columns else np.arange(len(df))

        # Get R2 values (handle different column names)
        r2_on = df.get('r_squared_on', df.get('tau_on_r_squared', df.get('r_squared_adj_on', None)))
        r2_off = df.get('r_squared_off', df.get('tau_off_r_squared', df.get('r_squared_adj_off', None)))

        if r2_on is not None:
            r2_on = r2_on.values
            self.ax.plot(cycles, r2_on, 'o-', color='tab:blue', label='R2 On', markersize=3)

        if r2_off is not None:
            r2_off = r2_off.values
            self.ax.plot(cycles, r2_off, 's-', color='tab:red', label='R2 Off', markersize=3)

        # Threshold line
        self.ax.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold ({threshold})')

        self.ax.set_xlabel('Cycle')
        self.ax.set_ylabel('R-squared')
        self.ax.set_title('Fitting Quality (R-squared)')
        self.ax.set_ylim(0, 1.05)
        self.ax.legend(loc='lower right')
        self.ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()


class ResultsTableWidget(QWidget):
    """Widget for displaying and exporting results table."""

    export_csv_requested = pyqtSignal(str)
    export_excel_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Header with stats
        header_layout = QHBoxLayout()
        self.stats_label = QLabel("No results")
        header_layout.addWidget(self.stats_label)
        header_layout.addStretch()

        # Export buttons
        self.csv_button = QPushButton("Export CSV")
        self.csv_button.clicked.connect(self._export_csv)
        self.csv_button.setEnabled(False)
        header_layout.addWidget(self.csv_button)

        self.excel_button = QPushButton("Export Excel")
        self.excel_button.clicked.connect(self._export_excel)
        self.excel_button.setEnabled(False)
        header_layout.addWidget(self.excel_button)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self._copy_to_clipboard)
        self.copy_button.setEnabled(False)
        header_layout.addWidget(self.copy_button)

        layout.addLayout(header_layout)

        # Table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        layout.addWidget(self.table)

    def set_results(self, df: pd.DataFrame):
        """
        Set results DataFrame.

        Args:
            df: Results DataFrame
        """
        self._df = df

        if df is None or df.empty:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.stats_label.setText("No results")
            self.csv_button.setEnabled(False)
            self.excel_button.setEnabled(False)
            self.copy_button.setEnabled(False)
            return

        # Set up table
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns.tolist())

        # Fill data
        for i, row in df.iterrows():
            for j, (col, value) in enumerate(row.items()):
                if isinstance(value, float):
                    text = f"{value:.6g}"
                else:
                    text = str(value)
                self.table.setItem(i, j, QTableWidgetItem(text))

        # Update stats
        n_cycles = len(df)
        tau_on_mean = df.get('tau_on', df.get('tau_on_value', pd.Series())).mean()
        tau_off_mean = df.get('tau_off', df.get('tau_off_value', pd.Series())).mean()

        stats_text = f"{n_cycles} cycles"
        if not np.isnan(tau_on_mean):
            stats_text += f" | Mean Tau On: {tau_on_mean:.4g} s"
        if not np.isnan(tau_off_mean):
            stats_text += f" | Mean Tau Off: {tau_off_mean:.4g} s"

        self.stats_label.setText(stats_text)

        # Enable buttons
        self.csv_button.setEnabled(True)
        self.excel_button.setEnabled(True)
        self.copy_button.setEnabled(True)

    def _export_csv(self):
        """Export to CSV file."""
        if self._df is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)"
        )
        if filepath:
            if not filepath.endswith('.csv'):
                filepath += '.csv'
            self._df.to_csv(filepath, index=False)
            self.export_csv_requested.emit(filepath)

    def _export_excel(self):
        """Export to Excel file."""
        if self._df is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Excel", "", "Excel Files (*.xlsx)"
        )
        if filepath:
            if not filepath.endswith('.xlsx'):
                filepath += '.xlsx'
            self._df.to_excel(filepath, index=False)
            self.export_excel_requested.emit(filepath)

    def _copy_to_clipboard(self):
        """Copy to clipboard."""
        if self._df is None:
            return

        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(self._df.to_csv(index=False, sep='\t'))

    def clear(self):
        """Clear the table."""
        self._df = None
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.stats_label.setText("No results")
        self.csv_button.setEnabled(False)
        self.excel_button.setEnabled(False)
        self.copy_button.setEnabled(False)


class ResultsView(QWidget):
    """Combined view for all result visualizations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main splitter
        splitter = QSplitter(Qt.Vertical)

        # Plot tabs
        plot_tabs = QTabWidget()

        # Raw data plot
        self.raw_plot = RawDataPlotWidget()
        plot_tabs.addTab(self.raw_plot, "Raw Data")

        # Tau evolution plot
        self.tau_plot = TauEvolutionPlotWidget()
        plot_tabs.addTab(self.tau_plot, "Tau Evolution")

        # R-squared plot
        self.r2_plot = RSquaredPlotWidget()
        plot_tabs.addTab(self.r2_plot, "R-squared")

        splitter.addWidget(plot_tabs)

        # Results table
        self.results_table = ResultsTableWidget()
        splitter.addWidget(self.results_table)

        # Set splitter sizes
        splitter.setSizes([400, 200])

        layout.addWidget(splitter)

    def update_raw_plot(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        period: Optional[float] = None,
        window_params: Optional[Dict[str, float]] = None
    ):
        """Update raw data plot."""
        self.raw_plot.plot_data(time, signal, period, window_params)

    def update_results(self, df: pd.DataFrame, threshold: float = 0.95):
        """Update all result visualizations."""
        self.tau_plot.plot_tau(df)
        self.r2_plot.plot_r_squared(df, threshold)
        self.results_table.set_results(df)

    def clear(self):
        """Clear all visualizations."""
        self.raw_plot.clear()
        self.tau_plot.clear()
        self.r2_plot.clear()
        self.results_table.clear()
