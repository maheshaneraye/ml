"""
Exploratory Data Analysis (EDA) Module
======================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Generates 16 comprehensive publication-grade visualizations and statistical findings:
1. Overall attendance distribution
2. Attendance by day of week
3. Attendance by lecture number
4. Attendance by time of day
5. Attendance by subject
6. Attendance by semester
7. Attendance by branch
8. Attendance by section
9. Attendance by faculty
10. Attendance during test weeks
11. Attendance around holidays
12. Practical vs Theory attendance
13. Weather vs attendance
14. Monthly attendance trend
15. Previous attendance vs current attendance
16. Correlation matrix for numerical variables
"""

import os
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300


def run_full_eda(
    df: pd.DataFrame,
    figures_dir: str = os.path.join("reports", "figures")
) -> List[str]:
    """
    Executes complete academic EDA suite and saves 16 high-resolution figures.
    """
    os.makedirs(figures_dir, exist_ok=True)
    generated_files = []
    logger.info(f"Generating 16 EDA visualizations from {len(df)} records...")

    # Helper for saving
    def save_fig(filename: str):
        path = os.path.join(figures_dir, filename)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        generated_files.append(path)
        logger.info(f"Saved figure: {filename}")

    # 1. Overall attendance distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Attendance Percentage"], kde=True, color="#2b5c8f", bins=25, edgecolor="black")
    plt.axvline(df["Attendance Percentage"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean ({df['Attendance Percentage'].mean():.1f}%)")
    plt.axvline(df["Attendance Percentage"].median(), color="green", linestyle=":", linewidth=1.5, label=f"Median ({df['Attendance Percentage'].median():.1f}%)")
    plt.title("Distribution of Classroom Attendance Percentage", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Attendance Percentage (%)", fontweight="bold")
    plt.ylabel("Frequency (Lectures)", fontweight="bold")
    plt.legend()
    save_fig("01_overall_attendance_distribution.png")

    # 2. Attendance by day of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Day of Week", y="Attendance Percentage", order=[d for d in day_order if d in df["Day of Week"].unique()], palette="Blues_r")
    plt.title("Attendance Percentage by Day of Week", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Day of Week", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("02_attendance_by_day_of_week.png")

    # 3. Attendance by lecture number
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="Lecture Number", y="Attendance Percentage", palette="viridis")
    plt.title("Attendance Percentage Across Timetable Slots (Lecture Numbers)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Timetable Period Slot (Lecture Number)", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("03_attendance_by_lecture_number.png")

    # 4. Attendance by time of day
    if "Time_of_Day" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Time_of_Day", y="Attendance Percentage", palette="magma", errorbar="sd", edgecolor="black")
        plt.title("Mean Attendance Percentage by Time of Day", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Time of Day Category", fontweight="bold")
        plt.ylabel("Attendance Percentage (%)", fontweight="bold")
        save_fig("04_attendance_by_time_of_day.png")
    else:
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Start Time", y="Attendance Percentage", palette="magma", errorbar="sd", edgecolor="black")
        plt.title("Mean Attendance Percentage by Start Time", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Start Time", fontweight="bold")
        plt.ylabel("Attendance Percentage (%)", fontweight="bold")
        plt.xticks(rotation=45)
        save_fig("04_attendance_by_time_of_day.png")

    # 5. Attendance by subject
    plt.figure(figsize=(12, 6))
    sub_order = df.groupby("Subject")["Attendance Percentage"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, y="Subject", x="Attendance Percentage", order=sub_order, palette="crest", edgecolor="black")
    plt.title("Mean Attendance Percentage by Subject", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Attendance Percentage (%)", fontweight="bold")
    plt.ylabel("Subject", fontweight="bold")
    save_fig("05_attendance_by_subject.png")

    # 6. Attendance by semester
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Semester", y="Attendance Percentage", palette="Set2")
    plt.title("Attendance Percentage Distribution Across Semesters", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Academic Semester", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("06_attendance_by_semester.png")

    # 7. Attendance by branch
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Branch", y="Attendance Percentage", palette="coolwarm", edgecolor="black")
    plt.title("Mean Attendance Percentage Across Engineering Branches", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Department / Branch", fontweight="bold")
    plt.ylabel("Mean Attendance (%)", fontweight="bold")
    save_fig("07_attendance_by_branch.png")

    # 8. Attendance by section
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Section", y="Attendance Percentage", palette="Spectral")
    plt.title("Attendance Distribution by Cohort Section", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Section", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("08_attendance_by_section.png")

    # 9. Attendance by faculty
    plt.figure(figsize=(10, 5))
    fac_order = df.groupby("Faculty ID")["Attendance Percentage"].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x="Faculty ID", y="Attendance Percentage", order=fac_order, palette="PuBuGn_r", edgecolor="black")
    plt.title("Mean Attendance by Anonymized Faculty ID", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Anonymized Faculty Identifier", fontweight="bold")
    plt.ylabel("Mean Attendance (%)", fontweight="bold")
    plt.xticks(rotation=45)
    save_fig("09_attendance_by_faculty.png")

    # 10. Attendance during test weeks
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Internal Test Week", y="Attendance Percentage", palette=["#e74c3c", "#2ecc71"])
    plt.title("Attendance Comparison: Regular Week vs. Internal Test Week", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Internal Test Week Active?", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("10_attendance_during_test_weeks.png")

    # 11. Attendance around holidays
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Holiday Before/After", y="Attendance Percentage", palette=["#3498db", "#f39c12"])
    plt.title("Attendance Impact: Adjacent to Public Holidays / Weekends", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Holiday Adjacent (Before/After)", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("11_attendance_around_holidays.png")

    # 12. Practical vs Theory attendance
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Practical/Theory", y="Attendance Percentage", palette=["#1abc9c", "#9b59b6"])
    plt.title("Attendance: Practical/Laboratory Sessions vs. Theory Lectures", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Session Format", fontweight="bold")
    plt.ylabel("Attendance Percentage (%)", fontweight="bold")
    save_fig("12_practical_vs_theory_attendance.png")

    # 13. Weather vs attendance
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Weather", y="Attendance Percentage", palette="YlOrBr_r", edgecolor="black")
    plt.title("Environmental Impact: Weather Conditions vs. Attendance", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Weather Condition", fontweight="bold")
    plt.ylabel("Mean Attendance Percentage (%)", fontweight="bold")
    save_fig("13_weather_vs_attendance.png")

    # 14. Monthly attendance trend
    if "Date" in df.columns:
        temp_df = df.copy()
        temp_df["Date_DT"] = pd.to_datetime(temp_df["Date"], format="mixed")
        temp_df = temp_df.sort_values("Date_DT")
        daily_trend = temp_df.groupby("Date_DT")["Attendance Percentage"].mean().reset_index()

        plt.figure(figsize=(12, 5))
        plt.plot(daily_trend["Date_DT"], daily_trend["Attendance Percentage"], marker="o", markersize=3, color="#2980b9", linewidth=2)
        plt.title("Temporal Macro Attendance Trend Across the Academic Calendar", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Date", fontweight="bold")
        plt.ylabel("Daily Mean Attendance (%)", fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        save_fig("14_monthly_attendance_trend.png")

    # 15. Previous attendance vs current attendance
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x="Previous Lecture Attendance", y="Attendance Percentage", scatter_kws={"alpha": 0.4, "color": "#2c3e50"}, line_kws={"color": "#e74c3c"})
    plt.title("Autoregressive Pattern: Previous vs. Current Lecture Attendance", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Previous Lecture Attendance (%)", fontweight="bold")
    plt.ylabel("Current Lecture Attendance (%)", fontweight="bold")
    save_fig("15_previous_vs_current_attendance.png")

    # 16. Correlation analysis for numerical variables
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
        plt.title("Correlation Matrix of Numerical Lecture Attributes", fontsize=13, fontweight="bold", pad=12)
        save_fig("16_correlation_heatmap.png")

    logger.info(f"All {len(generated_files)} EDA plots generated and saved successfully.")
    return generated_files


if __name__ == "__main__":
    clean_csv = os.path.join("data", "processed", "attendance_cleaned.csv")
    if not os.path.exists(clean_csv):
        raw_csv = os.path.join("data", "raw", "attendance_raw.csv")
        from src.data_cleaning import clean_attendance_data
        clean_attendance_data(raw_csv, clean_csv)
    
    df_clean = pd.read_csv(clean_csv)
    run_full_eda(df_clean)
