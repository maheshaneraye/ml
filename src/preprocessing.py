"""
Preprocessing and Chronological Data Splitting Module
=====================================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Defines:
- Chronological train/validation/test splitting (prevents temporal data leakage)
- Feature ColumnTransformer with imputation, scaling, and one-hot encoding
- Preprocessing pipeline serialization
"""

import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMN = "Attendance Percentage"

# Explicit definition of numerical and categorical predictive features
NUMERICAL_FEATURES = [
    "Lecture Number",
    "Start_Hour",
    "Semester",
    "Total Enrolled Students",
    "Previous Lecture Attendance",
    "Gap Since Previous Lecture",
    "Faculty Experience",
    "Day_of_Semester",
    "Week_Number",
    "Days_Since_Holiday",
    "Daily_Lecture_Sequence",
    "Rolling_Prev_3_Avg_Attendance",
    "Macro_Subject_Mean_Attendance",
    "Macro_Faculty_Mean_Attendance",
    "Monthly_Avg_Attendance",
    "Is_Morning",
    "Is_After_Lunch",
    "Week_Before_Exam_Flag"
]

CATEGORICAL_FEATURES = [
    "Day of Week",
    "Subject",
    "Faculty ID",
    "Branch",
    "Section",
    "Classroom",
    "Practical/Theory",
    "Internal Test Week",
    "Assignment Due",
    "Holiday Before/After",
    "Weather",
    "Special Event",
    "Time_of_Day",
    "Lunch_Timing"
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset chronologically into Train, Validation, and Test sets.
    
    Academic Rationale:
    Attendance patterns change over time (seasonality, syllabus progression, fatigue).
    Using a standard random shuffle split would cause future attendance dynamics to leak
    into the past training set (lookahead bias). Chronological splitting strictly tests the model
    on unseen future lectures, mirroring actual operational deployment.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-4, "Split ratios must sum to 1.0"
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Chronological Split: Total={n} | Train={len(train_df)} ({train_ratio*100:.0f}%) | "
        f"Val={len(val_df)} ({val_ratio*100:.0f}%) | Test={len(test_df)} ({test_ratio*100:.0f}%)"
    )

    return train_df, val_df, test_df


def build_preprocessor_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None
) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    num_cols = numerical_cols or NUMERICAL_FEATURES
    cat_cols = categorical_cols or CATEGORICAL_FEATURES

    # Numerical Transformer: Median Imputation + Standardization
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical Transformer: Frequent Imputation + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine into unified preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names_out(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names from the fitted ColumnTransformer.
    """
    feature_names = []
    
    # Numerical features
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(num_cols)

    # Categorical features from OneHotEncoder
    cat_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]
    cat_cols = preprocessor.transformers_[1][2]
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    feature_names.extend(list(encoded_cat_names))

    return feature_names
