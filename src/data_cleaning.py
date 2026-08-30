"""
Data Cleaning and Validation Module
===================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Handles:
- Data ingestion and structural validation against required schema
- Missing value detection and smart imputation/flagging
- Constraint validation (Students Present <= Total Enrolled, non-negative values)
- Target variable recalculation/validation to guarantee mathematical consistency
- Duplicate detection and removal
- Generation of comprehensive data cleaning audit reports
"""

import os
import json
import logging
from typing import Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Required columns based on capstone specification
REQUIRED_COLUMNS = [
    "Date",
    "Day of Week",
    "Lecture Number",
    "Start Time",
    "Subject",
    "Faculty ID",
    "Semester",
    "Branch",
    "Section",
    "Classroom",
    "Total Enrolled Students",
    "Students Present",
    "Attendance Percentage",
    "Previous Lecture Attendance",
    "Gap Since Previous Lecture",
    "Practical/Theory",
    "Internal Test Week",
    "Assignment Due",
    "Holiday Before/After",
    "Weather",
    "Special Event",
    "Faculty Experience"
]

VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
VALID_BINARY_FLAGS = ["Yes", "No", "Y", "N", 1, 0, "1", "0", True, False]
VALID_FORMATS = ["Theory", "Practical", "Lab"]


def standardize_binary(val: Any) -> str:
    """Standardizes binary yes/no strings or booleans to 'Yes' or 'No'."""
    if pd.isna(val):
        return "No"
    s = str(val).strip().capitalize()
    if s in ["Yes", "Y", "1", "True"]:
        return "Yes"
    return "No"


def validate_schema(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validates whether the dataframe contains all required columns.
    Returns (is_valid, missing_columns_list).
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_cols) == 0, missing_cols


def clean_attendance_data(
    input_source: Any,
    output_path: Optional[str] = None,
    report_output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans, validates, and standardizes raw attendance data.

    Parameters:
    -----------
    input_source : str or pd.DataFrame
        Path to CSV file or existing pandas DataFrame.
    output_path : Optional[str]
        Filepath to save the cleaned CSV dataset.
    report_output_path : Optional[str]
        Filepath to save the JSON/Text cleaning audit report.

    Returns:
    --------
    cleaned_df : pd.DataFrame
        Cleaned, strictly validated attendance dataset.
    report : Dict[str, Any]
        Audit summary metrics of the cleaning pipeline.
    """
    if isinstance(input_source, str):
        if not os.path.exists(input_source):
            raise FileNotFoundError(f"Raw attendance file not found at: {input_source}")
        logger.info(f"Loading raw attendance data from: {input_source}")
        df = pd.read_csv(input_source)
    elif isinstance(input_source, pd.DataFrame):
        df = input_source.copy()
    else:
        raise ValueError("input_source must be a file path string or pandas DataFrame.")

    initial_rows = len(df)
    logger.info(f"Initial raw record count: {initial_rows}")

    # Validate Schema
    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Please ensure dataset adheres to the standardized data dictionary schema."
        )

    # Initialize Cleaning Audit Metrics
    report = {
        "initial_row_count": initial_rows,
        "final_row_count": 0,
        "duplicate_rows_removed": 0,
        "missing_values_imputed_or_handled": 0,
        "invalid_attendance_anomalies_fixed": 0,
        "negative_or_zero_enrollment_rows_removed": 0,
        "date_parsing_errors": 0,
        "cleaning_status": "SUCCESS"
    }

    # 1. Deduplication
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found and removing {dup_count} duplicate rows.")
        df = df.drop_duplicates()
    report["duplicate_rows_removed"] = int(dup_count)

    # 2. Date parsing and standard formatting (DD-MM-YYYY -> datetime)
    df["Date_Parsed"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
    date_nulls = df["Date_Parsed"].isna().sum()
    if date_nulls > 0:
        logger.warning(f"Found {date_nulls} rows with unparseable dates. Removing them.")
        df = df.dropna(subset=["Date_Parsed"])
    report["date_parsing_errors"] = int(date_nulls)

    # Standardize Date representation to YYYY-MM-DD
    df["Date"] = df["Date_Parsed"].dt.strftime("%Y-%m-%d")
    df["Date_Parsed"] = pd.to_datetime(df["Date"])

    # Synchronize Day of Week if missing or mismatched
    df["Day of Week"] = df["Date_Parsed"].dt.day_name()

    # 3. Numeric Types and Constraint Checks
    # Total Enrolled Students must be positive
    df["Total Enrolled Students"] = pd.to_numeric(df["Total Enrolled Students"], errors="coerce")
    invalid_enrollment = (df["Total Enrolled Students"].isna()) | (df["Total Enrolled Students"] <= 0)
    if invalid_enrollment.sum() > 0:
        logger.warning(f"Removing {invalid_enrollment.sum()} rows with invalid or non-positive Total Enrolled Students.")
        df = df[~invalid_enrollment]
        report["negative_or_zero_enrollment_rows_removed"] = int(invalid_enrollment.sum())

    df["Total Enrolled Students"] = df["Total Enrolled Students"].astype(int)

    # Students Present must be non-negative and <= Total Enrolled Students
    df["Students Present"] = pd.to_numeric(df["Students Present"], errors="coerce")
    
    # Handle missing Students Present if Attendance Percentage exists
    missing_present = df["Students Present"].isna()
    if missing_present.sum() > 0 and "Attendance Percentage" in df.columns:
        df.loc[missing_present, "Students Present"] = np.round(
            (pd.to_numeric(df.loc[missing_present, "Attendance Percentage"], errors="coerce") / 100.0)
            * df.loc[missing_present, "Total Enrolled Students"]
        )
        report["missing_values_imputed_or_handled"] += int(missing_present.sum())

    # Fix anomalies where Students Present > Total Enrolled
    exceeded_mask = df["Students Present"] > df["Total Enrolled Students"]
    if exceeded_mask.sum() > 0:
        logger.warning(f"Capping {exceeded_mask.sum()} records where Students Present exceeded Total Enrolled.")
        df.loc[exceeded_mask, "Students Present"] = df.loc[exceeded_mask, "Total Enrolled Students"]
        report["invalid_attendance_anomalies_fixed"] += int(exceeded_mask.sum())

    # Fix negative Students Present
    negative_mask = df["Students Present"] < 0
    if negative_mask.sum() > 0:
        logger.warning(f"Correcting {negative_mask.sum()} negative Students Present values to 0.")
        df.loc[negative_mask, "Students Present"] = 0
        report["invalid_attendance_anomalies_fixed"] += int(negative_mask.sum())

    df["Students Present"] = df["Students Present"].fillna(0).astype(int)

    # 4. Strict Attendance Percentage Recalculation (Target Integrity)
    # Formula: (Students Present / Total Enrolled Students) * 100
    calculated_pct = np.round((df["Students Present"] / df["Total Enrolled Students"]) * 100.0, 2)
    df["Attendance Percentage"] = calculated_pct

    # 5. Lecture Number & Semester Normalization
    df["Lecture Number"] = pd.to_numeric(df["Lecture Number"], errors="coerce").fillna(1).astype(int)
    df["Lecture Number"] = df["Lecture Number"].clip(lower=1, upper=12)

    df["Semester"] = pd.to_numeric(df["Semester"], errors="coerce").fillna(5).astype(int)
    df["Semester"] = df["Semester"].clip(lower=1, upper=8)

    # 6. Previous Lecture Attendance & Gap
    df["Previous Lecture Attendance"] = pd.to_numeric(df["Previous Lecture Attendance"], errors="coerce")
    # Impute missing Previous Lecture Attendance with current Attendance Percentage or cohort mean
    prev_null_count = df["Previous Lecture Attendance"].isna().sum()
    if prev_null_count > 0:
        cohort_mean = df["Attendance Percentage"].mean() if not df.empty else 75.0
        df["Previous Lecture Attendance"] = df["Previous Lecture Attendance"].fillna(cohort_mean)
        report["missing_values_imputed_or_handled"] += int(prev_null_count)
    df["Previous Lecture Attendance"] = df["Previous Lecture Attendance"].clip(lower=0.0, upper=100.0)

    df["Gap Since Previous Lecture"] = pd.to_numeric(df["Gap Since Previous Lecture"], errors="coerce").fillna(24.0)
    df["Gap Since Previous Lecture"] = df["Gap Since Previous Lecture"].clip(lower=0.0, upper=168.0)

    # 7. Standardize Categoricals & Binary Flags
    df["Subject"] = df["Subject"].astype(str).str.strip()
    df["Faculty ID"] = df["Faculty ID"].astype(str).str.strip()
    df["Branch"] = df["Branch"].astype(str).str.strip().str.upper()
    df["Section"] = df["Section"].astype(str).str.strip().str.upper()
    df["Classroom"] = df["Classroom"].astype(str).str.strip()
    df["Weather"] = df["Weather"].fillna("Sunny").astype(str).str.strip().str.capitalize()
    
    # Practical vs Theory
    df["Practical/Theory"] = df["Practical/Theory"].astype(str).str.strip().str.capitalize()
    df["Practical/Theory"] = df["Practical/Theory"].apply(lambda x: "Practical" if "prac" in x.lower() or "lab" in x.lower() else "Theory")

    # Binary flags
    df["Internal Test Week"] = df["Internal Test Week"].apply(standardize_binary)
    df["Assignment Due"] = df["Assignment Due"].apply(standardize_binary)
    df["Holiday Before/After"] = df["Holiday Before/After"].apply(standardize_binary)
    df["Special Event"] = df["Special Event"].apply(standardize_binary)

    # Faculty Experience (years)
    df["Faculty Experience"] = pd.to_numeric(df["Faculty Experience"], errors="coerce").fillna(5.0).clip(lower=0.0, upper=45.0)

    # 8. Chronological Sort
    # Sorting ensures temporal sequence is strictly maintained
    df = df.sort_values(by=["Date_Parsed", "Start Time", "Lecture Number"]).reset_index(drop=True)
    df = df.drop(columns=["Date_Parsed"])

    final_rows = len(df)
    report["final_row_count"] = final_rows
    logger.info(f"Data cleaning complete. Final record count: {final_rows} rows.")

    # Optional Save Cleaned Data
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned dataset saved successfully to: {output_path}")

    # Optional Save Audit Report
    if report_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_output_path)), exist_ok=True)
        with open(report_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logger.info(f"Cleaning audit report saved to: {report_output_path}")

    return df, report


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "attendance_raw.csv")
    clean_path = os.path.join("data", "processed", "attendance_cleaned.csv")
    report_path = os.path.join("reports", "cleaning_report.json")

    if os.path.exists(raw_path):
        clean_df, audit_rep = clean_attendance_data(raw_path, clean_path, report_path)
        print("=== DATA CLEANING AUDIT REPORT ===")
        for k, v in audit_rep.items():
            print(f" - {k}: {v}")
    else:
        print(f"Raw data file not found at {raw_path}. Run synthetic generator or place real data.")
