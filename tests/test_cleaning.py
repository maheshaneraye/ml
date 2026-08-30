"""
Unit Tests for Data Cleaning and Validation
===========================================
"""

import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import clean_attendance_data, validate_schema, REQUIRED_COLUMNS


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame({
        "Date": ["01-08-2026", "01-08-2026", "02-08-2026", "03-08-2026", "03-08-2026"],
        "Day of Week": ["Monday", "Monday", "Tuesday", "Wednesday", "Wednesday"],
        "Lecture Number": [1, 2, 1, 1, 1],  # Row 5 is duplicate of Row 4
        "Start Time": ["09:00", "10:00", "09:00", "09:00", "09:00"],
        "Subject": ["Python", "DBMS", "OS", "CN", "CN"],
        "Faculty ID": ["F_001", "F_002", "F_003", "F_004", "F_004"],
        "Semester": [5, 5, 5, 5, 5],
        "Branch": ["CSE", "CSE", "CSE", "CSE", "CSE"],
        "Section": ["A", "A", "A", "A", "A"],
        "Classroom": ["Room 401", "Room 401", "Room 401", "Room 401", "Room 401"],
        "Total Enrolled Students": [60, 60, 60, 60, 60],
        "Students Present": [50, 70, -5, 45, 45],  # Row 2: 70 > 60 (anomaly), Row 3: -5 (negative)
        "Attendance Percentage": [83.33, 90.0, 0.0, 75.0, 75.0],
        "Previous Lecture Attendance": [80.0, 83.33, 75.0, 70.0, 70.0],
        "Gap Since Previous Lecture": [24.0, 1.0, 24.0, 24.0, 24.0],
        "Practical/Theory": ["Theory", "Theory", "Theory", "Theory", "Theory"],
        "Internal Test Week": ["No", "No", "No", "No", "No"],
        "Assignment Due": ["No", "No", "No", "No", "No"],
        "Holiday Before/After": ["No", "No", "No", "No", "No"],
        "Weather": ["Sunny", "Sunny", "Sunny", "Sunny", "Sunny"],
        "Special Event": ["No", "No", "No", "No", "No"],
        "Faculty Experience": [8, 12, 5, 7, 7]
    })


def test_schema_validation_success(sample_raw_df):
    is_valid, missing = validate_schema(sample_raw_df)
    assert is_valid is True
    assert len(missing) == 0


def test_schema_validation_failure():
    incomplete_df = pd.DataFrame({"Date": ["01-08-2026"], "Subject": ["Python"]})
    is_valid, missing = validate_schema(incomplete_df)
    assert is_valid is False
    assert len(missing) > 0


def test_clean_attendance_removes_duplicates(sample_raw_df):
    cleaned_df, report = clean_attendance_data(sample_raw_df)
    assert report["duplicate_rows_removed"] == 1
    assert len(cleaned_df) == 4


def test_clean_attendance_caps_overflow_students(sample_raw_df):
    cleaned_df, report = clean_attendance_data(sample_raw_df)
    # Row with Students Present = 70 in a class of 60 should be capped to 60
    assert (cleaned_df["Students Present"] <= cleaned_df["Total Enrolled Students"]).all()
    assert report["invalid_attendance_anomalies_fixed"] >= 1


def test_clean_attendance_corrects_negative_students(sample_raw_df):
    cleaned_df, _ = clean_attendance_data(sample_raw_df)
    assert (cleaned_df["Students Present"] >= 0).all()


def test_clean_attendance_percentage_mathematical_consistency(sample_raw_df):
    cleaned_df, _ = clean_attendance_data(sample_raw_df)
    expected_pct = np.round((cleaned_df["Students Present"] / cleaned_df["Total Enrolled Students"]) * 100.0, 2)
    np.testing.assert_allclose(cleaned_df["Attendance Percentage"], expected_pct, atol=0.05)
