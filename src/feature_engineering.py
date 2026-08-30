"""
Feature Engineering Module
==========================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Extracts and constructs high-signal predictive features:
1. Day of semester & Week number
2. Days elapsed since last holiday
3. Daily consecutive lecture count for student cohort
4. Strictly leakage-free rolling average attendance of previous 3 lectures
5. Historical macro/monthly attendance trends
6. Time-of-day categorization (Morning, Afternoon, Evening)
7. Before Lunch vs After Lunch classifications
8. Week before examination indicator
"""

import logging
from typing import Optional, Union, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-of-day categories, hour representations, and lunch timing flags.
    """
    df = df.copy()

    # Parse Start Time into numerical hour/minute
    def parse_hour_float(time_str: Any) -> float:
        if pd.isna(time_str):
            return 9.0
        s = str(time_str).strip().lower().replace(" ", "")
        # Handle '09:00', '9:00', '13:30', '9am', '1pm'
        try:
            if ":" in s:
                parts = s.split(":")
                h = float(parts[0])
                m = float(parts[1][:2]) if len(parts) > 1 else 0.0
                if "pm" in s and h < 12:
                    h += 12
                elif "am" in s and h == 12:
                    h = 0
                return h + m / 60.0
            else:
                return float(s)
        except Exception:
            return 9.0

    df["Start_Hour"] = df["Start Time"].apply(parse_hour_float)

    # Time of Day Classification
    # Morning: < 12.00, Afternoon: 12.00 - 16.50, Evening: > 16.50
    conditions = [
        (df["Start_Hour"] < 12.0),
        (df["Start_Hour"] >= 12.0) & (df["Start_Hour"] < 16.5),
        (df["Start_Hour"] >= 16.5)
    ]
    choices = ["Morning", "Afternoon", "Evening"]
    df["Time_of_Day"] = np.select(conditions, choices, default="Morning")

    # Morning vs Afternoon binary flag
    df["Is_Morning"] = (df["Start_Hour"] < 12.0).astype(int)

    # Before Lunch vs After Lunch (College standard: lunch break at 12:30 - 13:30)
    # Lectures starting at or after 13:00 / lecture slot >= 4 are After Lunch
    if "Lecture Number" in df.columns:
        df["Lunch_Timing"] = np.where(
            (df["Start_Hour"] >= 13.0) | (df["Lecture Number"] >= 4),
            "After Lunch",
            "Before Lunch"
        )
    else:
        df["Lunch_Timing"] = np.where(df["Start_Hour"] >= 13.0, "After Lunch", "Before Lunch")
    
    df["Is_After_Lunch"] = (df["Lunch_Timing"] == "After Lunch").astype(int)

    return df


def extract_academic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Day of Semester, Week Number, Holiday Proximity, and Exam Proximity.
    """
    df = df.copy()

    # Ensure datetime parsing for Date
    df["Date_DT"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
    
    # Fill missing dates with min date if any
    if df["Date_DT"].isna().all():
        df["Date_DT"] = pd.Timestamp.now()
    elif df["Date_DT"].isna().any():
        df["Date_DT"] = df["Date_DT"].fillna(df["Date_DT"].min())

    min_date = df["Date_DT"].min()

    # Day of Semester (integer day index starting from 1)
    df["Day_of_Semester"] = (df["Date_DT"] - min_date).dt.days + 1

    # Academic Week Number (1 to 16)
    df["Week_Number"] = ((df["Day_of_Semester"] - 1) // 7) + 1
    df["Week_Number"] = df["Week_Number"].clip(lower=1, upper=20)

    # Month of lecture
    df["Month"] = df["Date_DT"].dt.month_name()

    # Days elapsed since last holiday
    # We identify all dates marked with Holiday Before/After == 'Yes'
    holiday_dates = sorted(df[df["Holiday Before/After"] == "Yes"]["Date_DT"].unique())
    
    def calc_days_since_holiday(current_date):
        past_holidays = [h for h in holiday_dates if h <= current_date]
        if not past_holidays:
            return 7  # Baseline assumption if early in semester
        return int((current_date - max(past_holidays)).days)

    df["Days_Since_Holiday"] = df["Date_DT"].apply(calc_days_since_holiday).clip(lower=0, upper=60)

    # Week before Examination Flag
    # Identify test dates
    test_dates = sorted(df[df["Internal Test Week"] == "Yes"]["Date_DT"].unique())

    def calc_week_before_exam(current_date):
        # 1 if currently in test week or within 7 days before an internal test
        future_tests = [t for t in test_dates if 0 <= (t - current_date).days <= 7]
        return 1 if len(future_tests) > 0 else 0

    df["Week_Before_Exam_Flag"] = df["Date_DT"].apply(calc_week_before_exam)

    df = df.drop(columns=["Date_DT"])
    return df


def extract_cohort_and_lag_features(
    df: pd.DataFrame,
    is_training: bool = True,
    historical_cohort_stats: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Constructs cohort-level daily load and strictly leakage-free historical rolling averages.
    
    IMPORTANT: Rolling averages are computed strictly using *prior* historical rows (shift=1)
    so no future or current row attendance is leaked into the predictor signals.
    """
    df = df.copy()

    # Consecutive lecture count for the cohort on that day
    # Group by Date, Branch, Semester, Section and order by Start Time / Lecture Number
    cohort_keys = ["Date", "Branch", "Semester", "Section"]
    if all(k in df.columns for k in cohort_keys):
        df["Daily_Lecture_Sequence"] = df.groupby(cohort_keys).cumcount() + 1
    else:
        df["Daily_Lecture_Sequence"] = df.get("Lecture Number", 1)

    # Historical Rolling and Macro Attendance Features
    # If training and 'Attendance Percentage' is available, compute strictly shifted historical averages
    if "Attendance Percentage" in df.columns:
        # Group by Subject + Branch + Semester
        # Use shift(1) to avoid target leakage!
        df["Rolling_Prev_3_Avg_Attendance"] = (
            df.groupby(["Subject", "Branch", "Semester"])["Attendance Percentage"]
            .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
        )
        
        # Macro Subject Historical Mean (prior rows expanding mean)
        df["Macro_Subject_Mean_Attendance"] = (
            df.groupby("Subject")["Attendance Percentage"]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

        # Macro Faculty Historical Mean (prior rows expanding mean)
        df["Macro_Faculty_Mean_Attendance"] = (
            df.groupby("Faculty ID")["Attendance Percentage"]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )
        
        # Monthly Average Trend
        if "Month" in df.columns:
            df["Monthly_Avg_Attendance"] = (
                df.groupby("Month")["Attendance Percentage"]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            )
        else:
            df["Monthly_Avg_Attendance"] = np.nan

        # Fill remaining initial NaNs with Previous Lecture Attendance or global baseline (e.g. 75.0)
        baseline = df["Attendance Percentage"].mean() if not df.empty else 75.0
        fallback_prev = df.get("Previous Lecture Attendance", baseline)

        df["Rolling_Prev_3_Avg_Attendance"] = df["Rolling_Prev_3_Avg_Attendance"].fillna(fallback_prev).fillna(baseline)
        df["Macro_Subject_Mean_Attendance"] = df["Macro_Subject_Mean_Attendance"].fillna(fallback_prev).fillna(baseline)
        df["Macro_Faculty_Mean_Attendance"] = df["Macro_Faculty_Mean_Attendance"].fillna(fallback_prev).fillna(baseline)
        df["Monthly_Avg_Attendance"] = df["Monthly_Avg_Attendance"].fillna(baseline)

        # Store historical stats dictionary for inference on unseen future rows
        stats = {
            "global_baseline": float(baseline),
            "subject_means": df.groupby("Subject")["Attendance Percentage"].mean().to_dict(),
            "faculty_means": df.groupby("Faculty ID")["Attendance Percentage"].mean().to_dict(),
            "branch_means": df.groupby("Branch")["Attendance Percentage"].mean().to_dict(),
        }
    else:
        # Inference mode (when Attendance Percentage is not present)
        stats = historical_cohort_stats or {}
        baseline = stats.get("global_baseline", 75.0)
        subject_means = stats.get("subject_means", {})
        faculty_means = stats.get("faculty_means", {})

        df["Rolling_Prev_3_Avg_Attendance"] = df.get("Previous Lecture Attendance", baseline)
        df["Macro_Subject_Mean_Attendance"] = df["Subject"].map(subject_means).fillna(df.get("Previous Lecture Attendance", baseline))
        df["Macro_Faculty_Mean_Attendance"] = df["Faculty ID"].map(faculty_means).fillna(df.get("Previous Lecture Attendance", baseline))
        df["Monthly_Avg_Attendance"] = baseline

    return df, stats


def build_engineered_features(
    df: pd.DataFrame,
    is_training: bool = True,
    historical_stats: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full end-to-end feature engineering pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned attendance dataset.
    is_training : bool
        True if training phase, False for production inference.
    historical_stats : Optional[Dict]
        Precomputed training statistics for inference imputation.

    Returns:
    --------
    engineered_df : pd.DataFrame
        Dataset augmented with all engineered temporal and behavioral features.
    stats : Dict
        Calculated historical statistics metadata.
    """
    logger.info(f"Building engineered features for {len(df)} records (is_training={is_training})...")
    
    # 1. Time and timing features
    df = extract_time_features(df)

    # 2. Academic calendar and holiday/exam features
    df = extract_academic_calendar_features(df)

    # 3. Cohort load and leakage-free historical lags
    df, stats = extract_cohort_and_lag_features(df, is_training=is_training, historical_cohort_stats=historical_stats)

    logger.info("Feature engineering pipeline completed successfully.")
    return df, stats


if __name__ == "__main__":
    import os
    clean_path = os.path.join("data", "processed", "attendance_cleaned.csv")
    if os.path.exists(clean_path):
        clean_df = pd.read_csv(clean_path)
        feat_df, _ = build_engineered_features(clean_df, is_training=True)
        print("Engineered Columns:", feat_df.columns.tolist())
        print(feat_df.head(3))
    else:
        print("Run data cleaning first to generate processed attendance dataset.")
