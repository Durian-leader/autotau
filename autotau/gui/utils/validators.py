"""
Parameter validators for AutoTau GUI.
"""

from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    valid: bool
    value: Optional[float]
    message: str


class ParameterValidator:
    """Validator for fitting parameters."""

    @staticmethod
    def validate_positive_float(
        value: str,
        name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate a positive float value.

        Args:
            value: String value to validate
            name: Parameter name for error messages
            min_val: Optional minimum value (exclusive)
            max_val: Optional maximum value (exclusive)

        Returns:
            ValidationResult with validity, parsed value, and message
        """
        try:
            val = float(value)
        except ValueError:
            return ValidationResult(False, None, f"{name}: Invalid number format")

        if val <= 0:
            return ValidationResult(False, None, f"{name} must be positive")

        if min_val is not None and val <= min_val:
            return ValidationResult(False, None, f"{name} must be > {min_val}")

        if max_val is not None and val >= max_val:
            return ValidationResult(False, None, f"{name} must be < {max_val}")

        return ValidationResult(True, val, "")

    @staticmethod
    def validate_period(value: str) -> ValidationResult:
        """Validate period parameter."""
        return ParameterValidator.validate_positive_float(value, "Period")

    @staticmethod
    def validate_sample_rate(value: str) -> ValidationResult:
        """Validate sample rate parameter."""
        return ParameterValidator.validate_positive_float(value, "Sample rate")

    @staticmethod
    def validate_r_squared_threshold(value: str) -> ValidationResult:
        """Validate R-squared threshold (must be between 0 and 1)."""
        try:
            val = float(value)
        except ValueError:
            return ValidationResult(False, None, "R2 threshold: Invalid number format")

        if val <= 0 or val >= 1:
            return ValidationResult(False, None, "R2 threshold must be between 0 and 1")

        return ValidationResult(True, val, "")

    @staticmethod
    def validate_window_params(
        offset: str,
        size: str,
        period: float,
        name: str
    ) -> Tuple[ValidationResult, ValidationResult]:
        """
        Validate window offset and size parameters.

        Args:
            offset: Window offset string
            size: Window size string
            period: Signal period
            name: Window name ("On" or "Off")

        Returns:
            Tuple of (offset_result, size_result)
        """
        # Validate offset
        try:
            offset_val = float(offset)
        except ValueError:
            return (
                ValidationResult(False, None, f"{name} offset: Invalid number format"),
                ValidationResult(False, None, "")
            )

        if offset_val < 0:
            return (
                ValidationResult(False, None, f"{name} offset must be >= 0"),
                ValidationResult(False, None, "")
            )

        if offset_val >= period:
            return (
                ValidationResult(False, None, f"{name} offset must be < period ({period:.4f})"),
                ValidationResult(False, None, "")
            )

        # Validate size
        try:
            size_val = float(size)
        except ValueError:
            return (
                ValidationResult(True, offset_val, ""),
                ValidationResult(False, None, f"{name} size: Invalid number format")
            )

        if size_val <= 0:
            return (
                ValidationResult(True, offset_val, ""),
                ValidationResult(False, None, f"{name} size must be > 0")
            )

        if offset_val + size_val > period:
            return (
                ValidationResult(True, offset_val, ""),
                ValidationResult(False, None, f"{name} window exceeds period boundary")
            )

        return (
            ValidationResult(True, offset_val, ""),
            ValidationResult(True, size_val, "")
        )

    @staticmethod
    def validate_all_common(
        period: str,
        sample_rate: str,
        r_squared_threshold: str
    ) -> dict:
        """
        Validate all common parameters.

        Returns:
            Dict with validation results for each parameter
        """
        return {
            'period': ParameterValidator.validate_period(period),
            'sample_rate': ParameterValidator.validate_sample_rate(sample_rate),
            'r_squared_threshold': ParameterValidator.validate_r_squared_threshold(r_squared_threshold)
        }
