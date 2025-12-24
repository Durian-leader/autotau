"""
Clipboard data parser for Excel two-column data.

Supports:
- Tab-separated (Excel default)
- Comma-separated
- Space-separated
"""

import numpy as np
from typing import Tuple, Optional


def detect_delimiter(text: str) -> Optional[str]:
    """
    Detect the delimiter used in the clipboard text.

    Args:
        text: Raw clipboard text

    Returns:
        Delimiter character or None for whitespace
    """
    first_line = text.strip().split('\n')[0]

    # Check common delimiters in order of priority
    if '\t' in first_line:
        return '\t'
    elif ',' in first_line:
        return ','
    elif ';' in first_line:
        return ';'
    else:
        return None  # Use whitespace splitting


def parse_clipboard_data(
    clipboard_text: str,
    skip_header: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse two-column data from Excel clipboard.

    Args:
        clipboard_text: Raw text from clipboard
        skip_header: Whether to skip non-numeric header rows

    Returns:
        Tuple of (time_array, signal_array)

    Raises:
        ValueError: If data format is invalid or insufficient data
    """
    if not clipboard_text or not clipboard_text.strip():
        raise ValueError("No data in clipboard")

    lines = clipboard_text.strip().split('\n')
    delimiter = detect_delimiter(clipboard_text)

    time_data = []
    signal_data = []
    skipped_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Split by delimiter
        if delimiter:
            parts = line.split(delimiter)
        else:
            parts = line.split()

        # Need at least 2 columns
        if len(parts) < 2:
            skipped_lines.append((i + 1, "Less than 2 columns"))
            continue

        # Try to parse as numbers
        try:
            time_val = float(parts[0].strip())
            signal_val = float(parts[1].strip())
            time_data.append(time_val)
            signal_data.append(signal_val)
        except ValueError:
            if skip_header:
                skipped_lines.append((i + 1, "Non-numeric value (header?)"))
                continue
            else:
                raise ValueError(f"Line {i + 1}: Cannot parse '{parts[0]}' or '{parts[1]}' as number")

    # Validate data
    if len(time_data) < 10:
        raise ValueError(
            f"Insufficient data points: got {len(time_data)}, need at least 10. "
            f"Skipped {len(skipped_lines)} lines."
        )

    time_array = np.array(time_data)
    signal_array = np.array(signal_data)

    # Check for monotonically increasing time
    if not np.all(np.diff(time_array) > 0):
        # Try to sort by time
        sort_idx = np.argsort(time_array)
        time_array = time_array[sort_idx]
        signal_array = signal_array[sort_idx]

        # Check again
        if not np.all(np.diff(time_array) > 0):
            raise ValueError("Time values must be monotonically increasing")

    return time_array, signal_array


def estimate_sample_rate(time: np.ndarray) -> float:
    """
    Estimate sampling rate from time array.

    Args:
        time: Time array

    Returns:
        Estimated sample rate in Hz
    """
    if len(time) < 2:
        raise ValueError("Need at least 2 data points to estimate sample rate")

    # Use median of time differences for robustness
    dt = np.median(np.diff(time))
    if dt <= 0:
        raise ValueError("Cannot estimate sample rate: invalid time differences")

    return 1.0 / dt


def estimate_period(time: np.ndarray, signal: np.ndarray) -> Optional[float]:
    """
    Attempt to estimate signal period using FFT.

    Args:
        time: Time array
        signal: Signal array

    Returns:
        Estimated period in seconds, or None if cannot determine
    """
    try:
        from scipy import signal as sp_signal

        # Estimate sample rate
        dt = np.median(np.diff(time))
        fs = 1.0 / dt

        # Find peaks in FFT
        n = len(signal)
        fft = np.fft.rfft(signal - np.mean(signal))
        freqs = np.fft.rfftfreq(n, dt)

        # Find dominant frequency (skip DC component)
        magnitudes = np.abs(fft[1:])
        if len(magnitudes) == 0:
            return None

        peak_idx = np.argmax(magnitudes) + 1
        dominant_freq = freqs[peak_idx]

        if dominant_freq > 0:
            return 1.0 / dominant_freq
        return None

    except Exception:
        return None
