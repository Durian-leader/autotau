"""
Data input widget for AutoTau GUI.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog,
    QGroupBox, QSplitter
)
import numpy as np


class DataInputWidget(QWidget):
    """
    Widget for data input via clipboard or file.

    Signals:
        parse_requested: Emitted when Parse Clipboard button is clicked
        load_file_requested: Emitted when Load File button is clicked
        text_changed: Emitted when input text changes
    """
    parse_requested = pyqtSignal()
    load_file_requested = pyqtSignal(str)  # filepath
    text_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Instructions
        instructions = QLabel(
            "Paste two-column data from Excel (Time, Signal) below, "
            "or use 'Load File' to import from a file."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Create splitter for input and preview
        splitter = QSplitter(Qt.Vertical)

        # Input area
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)

        input_label = QLabel("Data Input:")
        input_label.setStyleSheet("font-weight: bold;")
        input_layout.addWidget(input_label)

        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText(
            "Time\tSignal\n"
            "0.000\t0.001234\n"
            "0.001\t0.002345\n"
            "...\t..."
        )
        self.text_input.textChanged.connect(self.text_changed)
        input_layout.addWidget(self.text_input)

        splitter.addWidget(input_widget)

        # Preview table
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("Data Preview: No data loaded")
        self.preview_label.setStyleSheet("font-weight: bold;")
        preview_layout.addWidget(self.preview_label)

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(2)
        self.preview_table.setHorizontalHeaderLabels(["Time (s)", "Signal"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        preview_layout.addWidget(self.preview_table)

        splitter.addWidget(preview_widget)

        # Set splitter sizes
        splitter.setSizes([200, 300])

        layout.addWidget(splitter, 1)

        # Buttons
        button_layout = QHBoxLayout()

        self.parse_button = QPushButton("Parse Clipboard")
        self.parse_button.setToolTip("Parse data from system clipboard")
        self.parse_button.clicked.connect(self._on_parse_clicked)
        button_layout.addWidget(self.parse_button)

        self.parse_text_button = QPushButton("Parse Text Above")
        self.parse_text_button.setToolTip("Parse data from the text input above")
        self.parse_text_button.clicked.connect(self._on_parse_text_clicked)
        button_layout.addWidget(self.parse_text_button)

        self.load_button = QPushButton("Load File...")
        self.load_button.setToolTip("Load data from a CSV or TXT file")
        self.load_button.clicked.connect(self._on_load_clicked)
        button_layout.addWidget(self.load_button)

        button_layout.addStretch()

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        button_layout.addWidget(self.clear_button)

        layout.addLayout(button_layout)

        # Statistics
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #666;")
        layout.addWidget(self.stats_label)

    def _on_parse_clicked(self):
        """Handle Parse Clipboard button click."""
        self.parse_requested.emit()

    def _on_parse_text_clicked(self):
        """Handle Parse Text button click."""
        text = self.text_input.toPlainText()
        if text.strip():
            # Store for later retrieval
            self._pending_text = text

    def _on_load_clicked(self):
        """Handle Load File button click."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data File",
            "",
            "All Supported (*.csv *.txt *.tsv);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if filepath:
            self.load_file_requested.emit(filepath)

    def get_input_text(self) -> str:
        """Get text from input area."""
        return self.text_input.toPlainText()

    def set_input_text(self, text: str):
        """Set text in input area."""
        self.text_input.setPlainText(text)

    def update_preview(self, time: np.ndarray, signal: np.ndarray, max_rows: int = 100):
        """
        Update the preview table with data.

        Args:
            time: Time array
            signal: Signal array
            max_rows: Maximum rows to display
        """
        n_points = len(time)
        display_count = min(n_points, max_rows)

        self.preview_table.setRowCount(display_count)

        for i in range(display_count):
            self.preview_table.setItem(i, 0, QTableWidgetItem(f"{time[i]:.6g}"))
            self.preview_table.setItem(i, 1, QTableWidgetItem(f"{signal[i]:.6g}"))

        # Update label
        if n_points > max_rows:
            self.preview_label.setText(
                f"Data Preview: Showing {max_rows} of {n_points} points"
            )
        else:
            self.preview_label.setText(f"Data Preview: {n_points} points")

        # Update statistics
        time_range = time[-1] - time[0]
        signal_min, signal_max = signal.min(), signal.max()
        self.stats_label.setText(
            f"Time range: {time[0]:.4g} - {time[-1]:.4g} s ({time_range:.4g} s) | "
            f"Signal range: {signal_min:.4g} - {signal_max:.4g}"
        )

    def clear(self):
        """Clear all data."""
        self.text_input.clear()
        self.preview_table.setRowCount(0)
        self.preview_label.setText("Data Preview: No data loaded")
        self.stats_label.setText("")

    def set_loading(self, loading: bool):
        """Set loading state."""
        self.parse_button.setEnabled(not loading)
        self.parse_text_button.setEnabled(not loading)
        self.load_button.setEnabled(not loading)
