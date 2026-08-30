"""
End-to-End Capstone Execution Pipeline
======================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Automates the complete machine learning lifecycle:
1. Ingestion and Cleaning of Raw Attendance Data
2. Exploratory Data Analysis (Generates 16 figures)
3. Feature Engineering & Chronological Train/Val/Test Splitting
4. Multi-Model Training, Hyperparameter Tuning & Model Selection
5. Serialization of Model, Preprocessor, Metadata & Experiment Reports
6. Sample Prediction Test
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

from src.data_cleaning import clean_attendance_data
from src.eda import run_full_eda
from src.train import train_and_tune_models
from src.predict import predict_attendance


def main():
    logger.info("================================================================================")
    logger.info("   CLASSROOM ATTENDANCE PREDICTION: FULL MACHINE LEARNING LIFECYCLE PIPELINE   ")
    logger.info("================================================================================")

    raw_data_path = os.path.join("data", "raw", "attendance_raw.csv")
    cleaned_data_path = os.path.join("data", "processed", "attendance_cleaned.csv")
    cleaning_report_path = os.path.join("reports", "cleaning_report.json")

    # Step 0: Ensure raw data exists
    if not os.path.exists(raw_data_path):
        logger.info(f"Raw attendance data not found at {raw_data_path}. Generating synthetic demo dataset...")
        from src.generate_demo_data import generate_synthetic_attendance_dataset
        generate_synthetic_attendance_dataset(output_path=raw_data_path)

    # Step 1: Data Cleaning & Validation
    logger.info("\n--- STEP 1: DATA CLEANING & VALIDATION ---")
    cleaned_df, clean_report = clean_attendance_data(
        input_source=raw_data_path,
        output_path=cleaned_data_path,
        report_output_path=cleaning_report_path
    )
    logger.info(f"Cleaning Report Summary: {json.dumps(clean_report, indent=2)}")

    # Step 2: Exploratory Data Analysis (16 Visualizations)
    logger.info("\n--- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---")
    eda_figures = run_full_eda(cleaned_df, figures_dir=os.path.join("reports", "figures"))
    logger.info(f"Successfully generated and saved {len(eda_figures)} EDA figures in reports/figures/")

    # Step 3 & 4: Feature Engineering, Model Training, Hyperparameter Tuning & Evaluation
    logger.info("\n--- STEP 3 & 4: FEATURE ENGINEERING, MODEL TRAINING & SELECTION ---")
    train_results = train_and_tune_models(train_csv_path=cleaned_data_path)
    logger.info(f"Training Complete! Selected Best Model: {train_results['best_model_name']}")
    logger.info(f"Best Model Validation Metrics: {train_results['metadata']['validation_metrics']}")
    logger.info(f"Best Model Test Metrics: {train_results['metadata']['test_metrics']}")

    # Step 5: Test Single Prediction Inference
    logger.info("\n--- STEP 5: INFERENCE & PREDICTION TEST ---")
    sample_lecture = {
        "Date": "2026-09-15",
        "Day of Week": "Monday",
        "Lecture Number": 2,
        "Start Time": "10:00",
        "Subject": "Python Programming",
        "Faculty ID": "F_001",
        "Semester": 5,
        "Branch": "CSE",
        "Section": "A",
        "Classroom": "Room 401",
        "Total Enrolled Students": 60,
        "Previous Lecture Attendance": 82.0,
        "Gap Since Previous Lecture": 24.0,
        "Practical/Theory": "Theory",
        "Internal Test Week": "No",
        "Assignment Due": "No",
        "Holiday Before/After": "No",
        "Weather": "Sunny",
        "Special Event": "No",
        "Faculty Experience": 8.0
    }
    pred_res = predict_attendance(sample_lecture)
    logger.info(f"Test Prediction Result: {json.dumps(pred_res, indent=2)}")

    logger.info("\n================================================================================")
    logger.info("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY! ALL ARTIFACTS ARE GENERATED.      ")
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
