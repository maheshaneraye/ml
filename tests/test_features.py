"""
Unit Tests for Feature Engineering
==================================
"""

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    extract_time_features,
    extract_academic_calendar_features,
    extract_cohort_and_lag_features,
    build_engineered_features
)


@pytest.fixture
def sample_cleaned_df():
    return pd.DataFrame({
        "Date": ["2026-08-03", "2026-08-03", "2026-08-04", "2026-08-10"],
        "Day of Week": ["Monday", "Monday", "Tuesday", "Monday"],
        "Lecture Number": [1, 5, 2, 1],
        "Start Time": ["09:00", "13:45", "10:00", "09:00"],
        "Subject": ["Python", "DBMS", "Python", "Python"],
        "Faculty ID": ["F_001", "F_002", "F_001", "F_001"],
        "Semester": [5, 5, 5, 5],
        "Branch": ["CSE", "CSE", "CSE", "CSE"],
        "Section": ["A", "A", "A", "A"],
        "Classroom": ["Room 401", "Room 401", "Room 401", "Room 401"],
        "Total Enrolled Students": [60, 60, 60, 60],
        "Students Present": [50, 48, 52, 54],
        "Attendance Percentage": [83.33, 80.0, 86.67, 90.0],
        "Previous Lecture Attendance": [80.0, 83.33, 80.0, 86.67],
        "Gap Since Previous Lecture": [24.0, 4.0, 24.0, 144.0],
        "Practical/Theory": ["Theory", "Theory", "Theory", "Theory"],
        "Internal Test Week": ["No", "No", "No", "Yes"],
        "Assignment Due": ["No", "No", "No", "No"],
        "Holiday Before/After": ["No", "No", "No", "No"],
        "Weather": ["Sunny", "Sunny", "Sunny", "Sunny"],
        "Special Event": ["No", "No", "No", "No"],
        "Faculty Experience": [8, 12, 8, 8]
    })


def test_time_features_extraction(sample_cleaned_df):
    df_time = extract_time_features(sample_cleaned_df)
    assert "Start_Hour" in df_time.columns
    assert "Time_of_Day" in df_time.columns
    assert "Is_Morning" in df_time.columns
    assert "Lunch_Timing" in df_time.columns

    # 09:00 should be Morning and Before Lunch
    assert df_time.iloc[0]["Time_of_Day"] == "Morning"
    assert df_time.iloc[0]["Lunch_Timing"] == "Before Lunch"
    assert df_time.iloc[0]["Is_Morning"] == 1

    # 13:45 should be Afternoon and After Lunch
    assert df_time.iloc[1]["Time_of_Day"] == "Afternoon"
    assert df_time.iloc[1]["Lunch_Timing"] == "After Lunch"
    assert df_time.iloc[1]["Is_Morning"] == 0


def test_academic_calendar_features(sample_cleaned_df):
    df_cal = extract_academic_calendar_features(sample_cleaned_df)
    assert "Day_of_Semester" in df_cal.columns
    assert "Week_Number" in df_cal.columns
    assert "Days_Since_Holiday" in df_cal.columns
    assert "Week_Before_Exam_Flag" in df_cal.columns

    # Day of semester should start at 1
    assert df_cal.iloc[0]["Day_of_Semester"] == 1
    assert df_cal.iloc[0]["Week_Number"] == 1
    assert df_cal.iloc[3]["Day_of_Semester"] == 8
    assert df_cal.iloc[3]["Week_Number"] == 2


def test_lag_features_leakage_free(sample_cleaned_df):
    df_lag, stats = extract_cohort_and_lag_features(sample_cleaned_df, is_training=True)
    assert "Rolling_Prev_3_Avg_Attendance" in df_lag.columns
    assert "Macro_Subject_Mean_Attendance" in df_lag.columns

    # Verify no NaNs remain
    assert df_lag["Rolling_Prev_3_Avg_Attendance"].isna().sum() == 0
    assert df_lag["Macro_Subject_Mean_Attendance"].isna().sum() == 0


def test_build_engineered_features_pipeline(sample_cleaned_df):
    feat_df, stats = build_engineered_features(sample_cleaned_df, is_training=True)
    assert len(feat_df) == len(sample_cleaned_df)
    assert "Daily_Lecture_Sequence" in feat_df.columns
    assert "Rolling_Prev_3_Avg_Attendance" in feat_df.columns
