"""
Sidebar widget with parameter inputs for AutoTau GUI.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QCheckBox, QPushButton, QGroupBox,
    QRadioButton, QButtonGroup, QFrame, QProgressBar,
    QSpacerItem, QSizePolicy
)

from ..utils.validators import ParameterValidator


class Sidebar(QWidget):
    """
    Sidebar widget for parameter configuration.

    Signals:
        fit_clicked: Emitted when Fit button is clicked
        export_clicked: Emitted when Export button is clicked
        clear_clicked: Emitted when Clear button is clicked
        mode_changed: Emitted when fitting mode changes (auto/manual)
    """
    fit_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()
    mode_changed = pyqtSignal(str)  # 'auto' or 'manual'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the sidebar UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Mode selection
        mode_group = self._create_mode_group()
        layout.addWidget(mode_group)

        # Common parameters
        common_group = self._create_common_params_group()
        layout.addWidget(common_group)

        # Manual mode parameters (initially hidden)
        self.manual_group = self._create_manual_params_group()
        self.manual_group.setVisible(False)
        layout.addWidget(self.manual_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        # Spacer
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Action buttons
        buttons_layout = self._create_action_buttons()
        layout.addLayout(buttons_layout)

    def _create_mode_group(self) -> QGroupBox:
        """Create mode selection group."""
        group = QGroupBox("Fitting Mode")
        layout = QHBoxLayout(group)

        self.mode_group = QButtonGroup(self)

        self.auto_radio = QRadioButton("Auto")
        self.auto_radio.setChecked(True)
        self.auto_radio.setToolTip("Automatically search for optimal fitting windows")
        self.mode_group.addButton(self.auto_radio)
        layout.addWidget(self.auto_radio)

        self.manual_radio = QRadioButton("Manual")
        self.manual_radio.setToolTip("Manually specify fitting window parameters")
        self.mode_group.addButton(self.manual_radio)
        layout.addWidget(self.manual_radio)

        # Connect mode change
        self.auto_radio.toggled.connect(self._on_mode_changed)

        return group

    def _create_common_params_group(self) -> QGroupBox:
        """Create common parameters group."""
        group = QGroupBox("Common Parameters")
        layout = QVBoxLayout(group)

        # Period
        period_layout = QHBoxLayout()
        period_layout.addWidget(QLabel("Period (s):"))
        self.period_input = QLineEdit()
        self.period_input.setPlaceholderText("e.g., 0.2")
        self.period_input.setToolTip("Signal period in seconds")
        period_layout.addWidget(self.period_input)
        layout.addLayout(period_layout)

        # Sample rate
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Sample Rate (Hz):"))
        self.sample_rate_input = QLineEdit()
        self.sample_rate_input.setPlaceholderText("e.g., 1000")
        self.sample_rate_input.setToolTip("Sampling rate in Hz")
        rate_layout.addWidget(self.sample_rate_input)
        layout.addLayout(rate_layout)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # Normalize checkbox
        self.normalize_check = QCheckBox("Normalize Signal")
        self.normalize_check.setToolTip("Normalize signal to [0, 1] range")
        layout.addWidget(self.normalize_check)

        # R-squared threshold
        r2_layout = QHBoxLayout()
        r2_layout.addWidget(QLabel("R2 Threshold:"))
        self.r2_threshold_input = QLineEdit("0.95")
        self.r2_threshold_input.setToolTip("Cycles with R2 below this will be refitted")
        r2_layout.addWidget(self.r2_threshold_input)
        layout.addLayout(r2_layout)

        return group

    def _create_manual_params_group(self) -> QGroupBox:
        """Create manual mode parameters group."""
        group = QGroupBox("Window Parameters")
        layout = QVBoxLayout(group)

        # On window
        on_label = QLabel("Turn-On Window:")
        on_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(on_label)

        on_offset_layout = QHBoxLayout()
        on_offset_layout.addWidget(QLabel("  Offset (s):"))
        self.on_offset_input = QLineEdit()
        self.on_offset_input.setPlaceholderText("e.g., 0.01")
        on_offset_layout.addWidget(self.on_offset_input)
        layout.addLayout(on_offset_layout)

        on_size_layout = QHBoxLayout()
        on_size_layout.addWidget(QLabel("  Size (s):"))
        self.on_size_input = QLineEdit()
        self.on_size_input.setPlaceholderText("e.g., 0.05")
        on_size_layout.addWidget(self.on_size_input)
        layout.addLayout(on_size_layout)

        # Off window
        off_label = QLabel("Turn-Off Window:")
        off_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(off_label)

        off_offset_layout = QHBoxLayout()
        off_offset_layout.addWidget(QLabel("  Offset (s):"))
        self.off_offset_input = QLineEdit()
        self.off_offset_input.setPlaceholderText("e.g., 0.11")
        off_offset_layout.addWidget(self.off_offset_input)
        layout.addLayout(off_offset_layout)

        off_size_layout = QHBoxLayout()
        off_size_layout.addWidget(QLabel("  Size (s):"))
        self.off_size_input = QLineEdit()
        self.off_size_input.setPlaceholderText("e.g., 0.05")
        off_size_layout.addWidget(self.off_size_input)
        layout.addLayout(off_size_layout)

        return group

    def _create_action_buttons(self) -> QVBoxLayout:
        """Create action buttons."""
        layout = QVBoxLayout()

        self.fit_button = QPushButton("Fit Data")
        self.fit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.fit_button.clicked.connect(self.fit_clicked)
        layout.addWidget(self.fit_button)

        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_clicked)
        layout.addWidget(self.export_button)

        self.clear_button = QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_clicked)
        layout.addWidget(self.clear_button)

        return layout

    def _on_mode_changed(self, checked):
        """Handle mode radio button change."""
        if self.auto_radio.isChecked():
            self.manual_group.setVisible(False)
            self.mode_changed.emit('auto')
        else:
            self.manual_group.setVisible(True)
            self.mode_changed.emit('manual')

    def get_mode(self) -> str:
        """Get current fitting mode."""
        return 'auto' if self.auto_radio.isChecked() else 'manual'

    def get_period(self) -> str:
        """Get period value."""
        return self.period_input.text().strip()

    def get_sample_rate(self) -> str:
        """Get sample rate value."""
        return self.sample_rate_input.text().strip()

    def get_normalize(self) -> bool:
        """Get normalize checkbox state."""
        return self.normalize_check.isChecked()

    def get_r2_threshold(self) -> str:
        """Get R2 threshold value."""
        return self.r2_threshold_input.text().strip()

    def get_on_offset(self) -> str:
        """Get on window offset."""
        return self.on_offset_input.text().strip()

    def get_on_size(self) -> str:
        """Get on window size."""
        return self.on_size_input.text().strip()

    def get_off_offset(self) -> str:
        """Get off window offset."""
        return self.off_offset_input.text().strip()

    def get_off_size(self) -> str:
        """Get off window size."""
        return self.off_size_input.text().strip()

    def set_period(self, value: float):
        """Set period value."""
        self.period_input.setText(f"{value:.6g}")

    def set_sample_rate(self, value: float):
        """Set sample rate value."""
        self.sample_rate_input.setText(f"{value:.1f}")

    def set_window_params(self, on_offset: float, on_size: float,
                          off_offset: float, off_size: float):
        """Set window parameters."""
        self.on_offset_input.setText(f"{on_offset:.6g}")
        self.on_size_input.setText(f"{on_size:.6g}")
        self.off_offset_input.setText(f"{off_offset:.6g}")
        self.off_size_input.setText(f"{off_size:.6g}")

    def set_status(self, message: str):
        """Set status label text."""
        self.status_label.setText(message)

    def set_progress(self, value: int, maximum: int = 100):
        """Set progress bar value."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

    def show_progress(self, show: bool = True):
        """Show or hide progress bar."""
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setValue(0)

    def set_fitting_enabled(self, enabled: bool):
        """Enable or disable fitting button."""
        self.fit_button.setEnabled(enabled)

    def set_export_enabled(self, enabled: bool):
        """Enable or disable export button."""
        self.export_button.setEnabled(enabled)

    def validate_common_params(self) -> tuple:
        """
        Validate common parameters.

        Returns:
            Tuple of (is_valid, error_message)
        """
        results = ParameterValidator.validate_all_common(
            self.get_period(),
            self.get_sample_rate(),
            self.get_r2_threshold()
        )

        errors = []
        for name, result in results.items():
            if not result.valid:
                errors.append(result.message)

        if errors:
            return False, "\n".join(errors)

        return True, ""

    def validate_manual_params(self, period: float) -> tuple:
        """
        Validate manual mode parameters.

        Args:
            period: Signal period

        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []

        on_offset, on_size = ParameterValidator.validate_window_params(
            self.get_on_offset(),
            self.get_on_size(),
            period,
            "On"
        )
        if not on_offset.valid:
            errors.append(on_offset.message)
        if not on_size.valid:
            errors.append(on_size.message)

        off_offset, off_size = ParameterValidator.validate_window_params(
            self.get_off_offset(),
            self.get_off_size(),
            period,
            "Off"
        )
        if not off_offset.valid:
            errors.append(off_offset.message)
        if not off_size.valid:
            errors.append(off_size.message)

        if errors:
            return False, "\n".join(errors)

        return True, ""
