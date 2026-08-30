"""
Notebook Generator Script
=========================
Creates five comprehensive, academically documented Jupyter Notebooks (.ipynb)
for the Classroom Attendance Prediction Capstone project.
"""

import os
import json


def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.13.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")


def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def generate_all_notebooks():
    # -------------------------------------------------------------
    # 01_data_cleaning.ipynb
    # -------------------------------------------------------------
    nb1_cells = [
        md_cell("""# Classroom Attendance Prediction: Phase 1 — Data Cleaning & Validation
## Capstone Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

### Objectives:
1. Load raw lecture attendance records.
2. Validate the schema against all 22 required attributes (Privacy Compliant, Anonymized Faculty IDs).
3. Detect and handle missing values, duplicates, and out-of-bounds attendance entries.
4. Enforce strict mathematical consistency: $\\text{Attendance Percentage} = (\\text{Students Present} / \\text{Total Enrolled}) \\times 100$.
5. Generate an automated Data Cleaning Audit Report.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

# Append project root to import src modules
sys.path.append("..")
from src.data_cleaning import clean_attendance_data, validate_schema, REQUIRED_COLUMNS"""),
        md_cell("### 1. Ingest Raw Attendance Records"),
        code_cell("""raw_path = os.path.join("..", "data", "raw", "attendance_raw.csv")
raw_df = pd.read_csv(raw_path)
print(f"Raw Dataset Shape: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")
raw_df.head()"""),
        md_cell("### 2. Schema Validation"),
        code_cell("""is_valid, missing = validate_schema(raw_df)
print(f"Schema Valid: {is_valid}")
if not is_valid:
    print(f"Missing columns: {missing}")
else:
    print("All 22 required attributes are present.")"""),
        md_cell("### 3. Execute Cleaning and Audit Pipeline"),
        code_cell("""cleaned_path = os.path.join("..", "data", "processed", "attendance_cleaned.csv")
report_path = os.path.join("..", "reports", "cleaning_report.json")

clean_df, audit_report = clean_attendance_data(raw_df, output_path=cleaned_path, report_output_path=report_path)
print("=== CLEANING AUDIT REPORT ===")
for k, v in audit_report.items():
    print(f" - {k}: {v}")"""),
        md_cell("### 4. Verify Cleaned Dataset Integrity"),
        code_cell("""print(f"Cleaned dataset shape: {clean_df.shape}")
print("\\nMissing values count:")
print(clean_df.isna().sum())
print("\\nSummary statistics:")
clean_df[["Total Enrolled Students", "Students Present", "Attendance Percentage"]].describe()""")
    ]
    create_notebook(nb1_cells, os.path.join("notebooks", "01_data_cleaning.ipynb"))

    # -------------------------------------------------------------
    # 02_eda.ipynb
    # -------------------------------------------------------------
    nb2_cells = [
        md_cell("""# Classroom Attendance Prediction: Phase 2 — Exploratory Data Analysis (EDA)
## Capstone Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

### Objectives:
Explore distribution patterns, temporal seasonality, subject variances, exam proximity impacts, and correlation structures across classroom attendance logs.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("..")
from src.eda import run_full_eda

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 150"""),
        md_cell("### 1. Load Cleaned Attendance Dataset"),
        code_cell("""clean_path = os.path.join("..", "data", "processed", "attendance_cleaned.csv")
df = pd.read_csv(clean_path)
print(f"Loaded {len(df)} cleaned lecture records.")
df.head()"""),
        md_cell("### 2. Run Automated 16-Figure Academic EDA Suite"),
        code_cell("""figures_dir = os.path.join("..", "reports", "figures")
fig_paths = run_full_eda(df, figures_dir=figures_dir)
print(f"Generated {len(fig_paths)} high-resolution figures in reports/figures/")"""),
        md_cell("### 3. Key Attendance Findings & Statistical Highlights"),
        code_cell("""print(f"Overall Mean Attendance: {df['Attendance Percentage'].mean():.2f}%")
print(f"Overall Median Attendance: {df['Attendance Percentage'].median():.2f}%")
print(f"Attendance Standard Deviation: {df['Attendance Percentage'].std():.2f}%")

print("\\nMean Attendance by Day of Week:")
print(df.groupby('Day of Week')['Attendance Percentage'].mean().sort_values(ascending=False))

print("\\nMean Attendance by Session Format:")
print(df.groupby('Practical/Theory')['Attendance Percentage'].mean())

print("\\nInternal Test Week Impact:")
print(df.groupby('Internal Test Week')['Attendance Percentage'].mean())""")
    ]
    create_notebook(nb2_cells, os.path.join("notebooks", "02_eda.ipynb"))

    # -------------------------------------------------------------
    # 03_feature_engineering.ipynb
    # -------------------------------------------------------------
    nb3_cells = [
        md_cell("""# Classroom Attendance Prediction: Phase 3 — Feature Engineering
## Capstone Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

### Objectives:
1. Extract temporal and timetable signals (Day of Semester, Week Number, Lunch Timings).
2. Calculate academic holiday and examination proximity flags.
3. Compute strictly **leakage-free** rolling lag features (Previous 3-lecture rolling mean).
4. Perform chronological train/val/test splitting to eliminate lookahead bias.
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.append("..")
from src.feature_engineering import build_engineered_features
from src.preprocessing import chronological_split, ALL_FEATURE_COLUMNS, TARGET_COLUMN"""),
        md_cell("### 1. Build Engineered Features"),
        code_cell("""clean_path = os.path.join("..", "data", "processed", "attendance_cleaned.csv")
df = pd.read_csv(clean_path)

feat_df, hist_stats = build_engineered_features(df, is_training=True)
print(f"Feature dataframe columns count: {len(feat_df.columns)}")
print("Engineered features preview:")
feat_df[['Date', 'Subject', 'Day_of_Semester', 'Week_Number', 'Lunch_Timing', 'Rolling_Prev_3_Avg_Attendance']].head()"""),
        md_cell("### 2. Chronological Splitting (70% Train, 15% Val, 15% Test)"),
        code_cell("""train_df, val_df, test_df = chronological_split(feat_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
print(f"Train split: {len(train_df)} rows")
print(f"Val split:   {len(val_df)} rows")
print(f"Test split:  {len(test_df)} rows")""")
    ]
    create_notebook(nb3_cells, os.path.join("notebooks", "03_feature_engineering.ipynb"))

    # -------------------------------------------------------------
    # 04_model_training.ipynb
    # -------------------------------------------------------------
    nb4_cells = [
        md_cell("""# Classroom Attendance Prediction: Phase 4 — Model Training & Tuning
## Capstone Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

### Objectives:
1. Construct scikit-learn ColumnTransformer preprocessor pipeline fitted only on train split.
2. Train and tune multiple regression models:
   - Linear Regression / Ridge (Baseline)
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - XGBoost Regressor
3. Evaluate models on validation split and select the best model.
4. Save serialized artifacts (`best_model.pkl`, `preprocessor.pkl`, `model_metadata.json`).
"""),
        code_cell("""import os
import sys
import pandas as pd
import numpy as np

sys.path.append("..")
from src.train import train_and_tune_models"""),
        md_cell("### 1. Execute Training & Hyperparameter Tuning Pipeline"),
        code_cell("""results = train_and_tune_models(
    train_csv_path=os.path.join("..", "data", "processed", "attendance_cleaned.csv"),
    models_dir=os.path.join("..", "models"),
    reports_dir=os.path.join("..", "reports")
)

print(f"=== BEST MODEL SELECTED: {results['best_model_name']} ===")"""),
        md_cell("### 2. Inspect Experiment Results Table"),
        code_cell("""exp_df = pd.read_csv(os.path.join("..", "reports", "experiment_results.csv"))
exp_df""")
    ]
    create_notebook(nb4_cells, os.path.join("notebooks", "04_model_training.ipynb"))

    # -------------------------------------------------------------
    # 05_model_evaluation.ipynb
    # -------------------------------------------------------------
    nb5_cells = [
        md_cell("""# Classroom Attendance Prediction: Phase 5 — Model Evaluation & Diagnostics
## Capstone Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

### Objectives:
1. Compare candidate models across MAE, RMSE, MAPE, and $R^2$.
2. Analyze feature importance and permutation importance.
3. Perform residual error diagnostics and actual vs. predicted goodness of fit.
4. Test real-world inference and expected student calculation.
"""),
        code_cell("""import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")
from src.predict import predict_attendance"""),
        md_cell("### 1. Multi-Model Comparison"),
        code_cell("""comp_df = pd.read_csv(os.path.join("..", "reports", "model_comparison.csv"))
comp_df"""),
        md_cell("### 2. Model Metadata & Test Metrics"),
        code_cell("""with open(os.path.join("..", "models", "model_metadata.json"), "r") as f:
    meta = json.load(f)

print(f"Model Name: {meta['model_name']}")
print(f"Validation Metrics: {meta['validation_metrics']}")
print(f"Test Metrics: {meta['test_metrics']}")"""),
        md_cell("### 3. Test Inference and Expected Students Present Calculation"),
        code_cell("""sample_lecture = {
    "Date": "2026-09-20",
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
    "Previous Lecture Attendance": 80.0,
    "Gap Since Previous Lecture": 24.0,
    "Practical/Theory": "Theory",
    "Internal Test Week": "No",
    "Assignment Due": "No",
    "Holiday Before/After": "No",
    "Weather": "Sunny",
    "Special Event": "No",
    "Faculty Experience": 8.0
}

pred = predict_attendance(
    sample_lecture,
    model_path=os.path.join("..", "models", "best_model.pkl"),
    preprocessor_path=os.path.join("..", "models", "preprocessor.pkl"),
    metadata_path=os.path.join("..", "models", "model_metadata.json")
)

print("Prediction Output:")
for k, v in pred.items():
    print(f"  {k}: {v}")""")
    ]
    create_notebook(nb5_cells, os.path.join("notebooks", "05_model_evaluation.ipynb"))


if __name__ == "__main__":
    generate_all_notebooks()
