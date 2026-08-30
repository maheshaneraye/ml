"""
Unit Tests for Model Prediction Pipeline
========================================
"""

import pytest
import pandas as pd
import numpy as np
from src.predict import get_attendance_category, predict_attendance


def test_attendance_category_thresholds():
    # Low: < 50%
    assert get_attendance_category(45.0) == "Low Attendance"
    assert get_attendance_category(49.99) == "Low Attendance"

    # Medium: 50% - 75%
    assert get_attendance_category(50.0) == "Medium Attendance"
    assert get_attendance_category(65.5) == "Medium Attendance"
    assert get_attendance_category(75.0) == "Medium Attendance"

    # High: > 75%
    assert get_attendance_category(75.01) == "High Attendance"
    assert get_attendance_category(88.4) == "High Attendance"
    assert get_attendance_category(100.0) == "High Attendance"


def test_expected_students_calculation():
    # Formula: round(Predicted % / 100 * Total Enrolled)
    total_enrolled = 60
    pred_pct = 82.4
    expected_present = int(round((pred_pct / 100.0) * total_enrolled))
    assert expected_present == 49
    assert expected_present <= total_enrolled
