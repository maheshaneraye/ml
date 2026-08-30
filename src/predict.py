"""
Prediction and Inference Pipeline Module
========================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Provides:
- Reusable inference function `predict_attendance`
- Mathematical calculation of Expected Students Present
- Categorical Attendance Band assignment (Low <50%, Medium 50-75%, High >75%)
- Standalone CLI execution for test predictions
"""

import os
import json
import logging
from typing import Union, Dict, List, Any
import pandas as pd
import numpy as np
import joblib

from src.feature_engineering import build_engineered_features
from src.preprocessing import ALL_FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join("models", "best_model.pkl")
DEFAULT_PREPROCESSOR_PATH = os.path.join("models", "preprocessor.pkl")
DEFAULT_METADATA_PATH = os.path.join("models", "model_metadata.json")


def get_attendance_category(attendance_pct: float) -> str:
    """
    Categorizes predicted attendance percentage into academic risk bands:
    - Low Attendance: < 50% (High risk of academic deficit)
    - Medium Attendance: 50% - 75% (Moderate attendance)
    - High Attendance: > 75% (Healthy attendance / university benchmark compliant)
    """
    if attendance_pct < 50.0:
        return "Low Attendance"
    elif attendance_pct <= 75.0:
        return "Medium Attendance"
    else:
        return "High Attendance"


def predict_attendance(
    input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH,
    preprocessor_path: str = DEFAULT_PREPROCESSOR_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH
) -> Union[Dict[str, Any], pd.DataFrame]:
    """
    Predicts classroom attendance percentage and expected students present for upcoming lectures.

    Parameters:
    -----------
    input_data : Dict, List of Dicts, or DataFrame
        Raw input feature(s) for scheduled lecture(s).
    model_path : str
        Path to serialized best_model.pkl.
    preprocessor_path : str
        Path to serialized preprocessor.pkl.
    metadata_path : str
        Path to model_metadata.json.

    Returns:
    --------
    results : Dict or DataFrame
        Predicted attendance percentage, expected students present, and attendance risk band.
    """
    # 1. Convert input to DataFrame
    is_single_dict = isinstance(input_data, dict)
    if is_single_dict:
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be a dict, list of dicts, or pandas DataFrame.")

    # 2. Check model artifacts existence with smart fallback
    model = None
    preprocessor = None
    historical_stats = {}

    # Check candidate model bundle paths
    bundle_candidates = [
        "attendance_model.pkl",
        os.path.join("04_Deployment", "attendance_model.pkl"),
        os.path.join("..", "04_Deployment", "attendance_model.pkl")
    ]
    for bp in bundle_candidates:
        if os.path.exists(bp):
            try:
                bundle = joblib.load(bp)
                if isinstance(bundle, dict) and "model" in bundle and "preprocessor" in bundle:
                    model = bundle["model"]
                    preprocessor = bundle["preprocessor"]
                    historical_stats = bundle.get("metadata", {}).get("historical_stats", {})
                    break
            except Exception:
                pass

    if model is None:
        model_candidates = [model_path, os.path.join("..", model_path)]
        prep_candidates = [preprocessor_path, os.path.join("..", preprocessor_path)]
        
        m_file = next((p for p in model_candidates if os.path.exists(p)), None)
        p_file = next((p for p in prep_candidates if os.path.exists(p)), None)
        
        if m_file and p_file:
            model = joblib.load(m_file)
            preprocessor = joblib.load(p_file)
            
            meta_candidates = [metadata_path, os.path.join("..", metadata_path)]
            meta_file = next((p for p in meta_candidates if os.path.exists(p)), None)
            if meta_file:
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        historical_stats = meta.get("historical_stats", {})
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
        else:
            raise FileNotFoundError(
                f"Trained model artifacts not found at {model_path} / {preprocessor_path}. "
                "Please train the model first using `python src/train.py`."
            )

    # 4. Fill required baseline defaults if missing in inference input
    defaults = {
        "Date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "Day of Week": "Monday",
        "Lecture Number": 1,
        "Start Time": "09:00",
        "Subject": "Python",
        "Faculty ID": "F_001",
        "Semester": 5,
        "Branch": "CSE",
        "Section": "A",
        "Classroom": "Room 401",
        "Total Enrolled Students": 60,
        "Previous Lecture Attendance": 75.0,
        "Gap Since Previous Lecture": 24.0,
        "Practical/Theory": "Theory",
        "Internal Test Week": "No",
        "Assignment Due": "No",
        "Holiday Before/After": "No",
        "Weather": "Sunny",
        "Special Event": "No",
        "Faculty Experience": 5.0
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # 5. Extract engineered features for inference (is_training=False)
    feat_df, _ = build_engineered_features(df, is_training=False, historical_stats=historical_stats)

    # 6. Apply Preprocessor
    X_inference_raw = feat_df[ALL_FEATURE_COLUMNS]
    X_inference_transformed = preprocessor.transform(X_inference_raw)

    # 7. Predict and Clip to [0, 100]
    raw_predictions = model.predict(X_inference_transformed)
    clipped_predictions = np.clip(raw_predictions, 0.0, 100.0)

    # 8. Calculate Expected Students Present and Categories
    enrolled = feat_df["Total Enrolled Students"].values
    expected_present = np.round((clipped_predictions / 100.0) * enrolled).astype(int)
    # Ensure expected present doesn't exceed enrolled
    expected_present = np.minimum(expected_present, enrolled)
    categories = [get_attendance_category(p) for p in clipped_predictions]

    feat_df["Predicted Attendance Percentage"] = np.round(clipped_predictions, 2)
    feat_df["Expected Students Present"] = expected_present
    feat_df["Attendance Category"] = categories

    if is_single_dict:
        return {
            "Predicted Attendance Percentage": float(feat_df["Predicted Attendance Percentage"].iloc[0]),
            "Expected Students Present": int(feat_df["Expected Students Present"].iloc[0]),
            "Total Enrolled Students": int(feat_df["Total Enrolled Students"].iloc[0]),
            "Attendance Category": str(feat_df["Attendance Category"].iloc[0]),
            "Status": "SUCCESS"
        }

    return feat_df[[
        "Date", "Day of Week", "Start Time", "Subject", "Faculty ID",
        "Total Enrolled Students", "Predicted Attendance Percentage",
        "Expected Students Present", "Attendance Category"
    ]]


if __name__ == "__main__":
    sample_lecture = {
        "Date": "2026-09-15",
        "Day of Week": "Monday",
        "Lecture Number": 1,
        "Start Time": "09:00",
        "Subject": "Python",
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

    print("Executing Sample Prediction...")
    try:
        res = predict_attendance(sample_lecture)
        print("Prediction Result:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Prediction could not execute (requires trained model first): {e}")
