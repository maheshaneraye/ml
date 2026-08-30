"""
Model Training and Hyperparameter Tuning Module
===============================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Trains multiple machine learning regression models:
1. Linear Regression (Baseline)
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. XGBoost Regressor (Advanced Ensembling)

Performs hyperparameter tuning, model comparison, best model selection,
and saves serialization artifacts and experiment logs.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# XGBoost conditional import
XGB_AVAILABLE = False
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.data_cleaning import clean_attendance_data
from src.feature_engineering import build_engineered_features
from src.preprocessing import (
    chronological_split,
    build_preprocessor_pipeline,
    get_feature_names_out,
    TARGET_COLUMN,
    ALL_FEATURE_COLUMNS
)
from src.evaluate import (
    calculate_metrics,
    save_model_comparison_table,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    extract_and_plot_feature_importance,
    compute_and_plot_permutation_importance
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_tune_models(
    train_csv_path: str = os.path.join("data", "processed", "attendance_cleaned.csv"),
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Full training pipeline:
    1. Loads cleaned dataset
    2. Builds engineered features
    3. Chronological train/val/test split
    4. Fits ColumnTransformer preprocessor
    5. Trains and tunes all regression models
    6. Selects best model based on validation score
    7. Evaluates on test set
    8. Saves model artifacts, metadata, and visual reports
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)

    if not os.path.exists(train_csv_path):
        raw_csv_path = os.path.join("data", "raw", "attendance_raw.csv")
        if os.path.exists(raw_csv_path):
            logger.info(f"Cleaned dataset not found. Generating from {raw_csv_path}...")
            clean_attendance_data(raw_csv_path, train_csv_path)
        else:
            raise FileNotFoundError(f"Neither {train_csv_path} nor {raw_csv_path} exists.")

    df = pd.read_csv(train_csv_path)
    logger.info(f"Loaded {len(df)} records from {train_csv_path}")

    # 1. Feature Engineering
    feat_df, hist_stats = build_engineered_features(df, is_training=True)

    # 2. Chronological Splitting (70% Train, 15% Validation, 15% Test)
    train_df, val_df, test_df = chronological_split(feat_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    X_train_raw = train_df[ALL_FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN].values

    X_val_raw = val_df[ALL_FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN].values

    X_test_raw = test_df[ALL_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN].values

    # 3. Fit Preprocessing Pipeline ONLY on Training Data
    logger.info("Fitting Preprocessing ColumnTransformer on training split...")
    preprocessor = build_preprocessor_pipeline()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names_out(preprocessor)
    logger.info(f"Transformed feature space dimensions: {X_train.shape[1]} features.")

    # 4. Define Candidate Model Specs and Hyperparameter Grids
    model_definitions = {
        "Linear Regression": {
            "estimator": Ridge(alpha=1.0),
            "param_grid": {"alpha": [0.1, 1.0, 10.0]}
        },
        "Decision Tree": {
            "estimator": DecisionTreeRegressor(random_state=42),
            "param_grid": {
                "max_depth": [4, 6, 8, 12],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [50, 100],
                "max_depth": [6, 10, 15],
                "min_samples_split": [2, 5]
            }
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingRegressor(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }
    }

    if XGB_AVAILABLE:
        model_definitions["XGBoost"] = {
            "estimator": xgb.XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror"),
            "param_grid": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        }
    else:
        logger.warning("XGBoost is not installed or enabled. Skipping XGBoost model.")

    # 5. Training, Hyperparameter Search, and Multi-Split Evaluation
    experiment_records = []
    val_performance = {}
    trained_models = {}

    for name, spec in model_definitions.items():
        logger.info(f"--> Training and tuning model: {name}")
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            cv=3,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        train_duration = round(time.time() - start_time, 3)

        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        trained_models[name] = best_estimator

        # Predictions on Train, Validation, and Test
        y_train_pred = best_estimator.predict(X_train)
        y_val_pred = best_estimator.predict(X_val)
        y_test_pred = best_estimator.predict(X_test)

        # Clip predictions to sensible percentage bounds [0, 100]
        y_train_pred = np.clip(y_train_pred, 0.0, 100.0)
        y_val_pred = np.clip(y_val_pred, 0.0, 100.0)
        y_test_pred = np.clip(y_test_pred, 0.0, 100.0)

        # Calculate metrics
        train_m = calculate_metrics(y_train, y_train_pred)
        val_m = calculate_metrics(y_val, y_val_pred)
        test_m = calculate_metrics(y_test, y_test_pred)

        val_performance[name] = val_m

        exp_row = {
            "Model": name,
            "Hyperparameters": json.dumps(best_params),
            "Training MAE": train_m["MAE"],
            "Validation MAE": val_m["MAE"],
            "Test MAE": test_m["MAE"],
            "Training RMSE": train_m["RMSE"],
            "Validation RMSE": val_m["RMSE"],
            "Test RMSE": test_m["RMSE"],
            "Training R2": train_m["R2"],
            "Validation R2": val_m["R2"],
            "Test R2": test_m["R2"],
            "Test MAPE (%)": test_m["MAPE (%)"],
            "Training Time (s)": train_duration
        }
        experiment_records.append(exp_row)
        logger.info(f"{name} Results -> Val MAE: {val_m['MAE']} | Val R2: {val_m['R2']} | Test MAE: {test_m['MAE']} | Test R2: {test_m['R2']}")

    # 6. Save Experiment Results Matrix
    exp_df = pd.DataFrame(experiment_records)
    exp_csv_path = os.path.join(reports_dir, "experiment_results.csv")
    exp_df.to_csv(exp_csv_path, index=False)
    logger.info(f"Experiment matrix saved to: {exp_csv_path}")

    # 7. Save Model Comparison Table and Comparison Plot
    comp_df = save_model_comparison_table(
        val_performance,
        output_path=os.path.join(reports_dir, "model_comparison.csv")
    )
    plot_model_comparison(comp_df, output_path=os.path.join(reports_dir, "figures", "model_comparison_metrics.png"))

    # 8. Select Best Model Based on Validation Performance (Lowest MAE / Highest R2)
    best_model_name = comp_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_val_metrics = val_performance[best_model_name]
    
    # Final Test evaluation on best model
    y_test_pred_best = np.clip(best_model.predict(X_test), 0.0, 100.0)
    best_test_metrics = calculate_metrics(y_test, y_test_pred_best)

    logger.info(f"=== BEST MODEL SELECTED: {best_model_name} ===")
    logger.info(f"Validation Metrics: {best_val_metrics}")
    logger.info(f"Final Test Metrics: {best_test_metrics}")

    # 9. Generate Diagnostic Visuals for Best Model
    plot_actual_vs_predicted(
        y_test, y_test_pred_best, best_model_name,
        output_path=os.path.join(reports_dir, "figures", "actual_vs_predicted.png")
    )
    plot_residual_distribution(
        y_test, y_test_pred_best, best_model_name,
        output_path=os.path.join(reports_dir, "figures", "residual_distribution.png")
    )
    feat_imp_df = extract_and_plot_feature_importance(
        best_model, feature_names, top_n=10,
        output_path=os.path.join(reports_dir, "figures", "feature_importance.png")
    )
    compute_and_plot_permutation_importance(
        best_model, X_test, y_test, feature_names, top_n=10,
        output_path=os.path.join(reports_dir, "figures", "permutation_importance.png")
    )

    # 10. Persist Model, Preprocessor, and Comprehensive Metadata
    model_save_path = os.path.join(models_dir, "best_model.pkl")
    preprocessor_save_path = os.path.join(models_dir, "preprocessor.pkl")
    metadata_save_path = os.path.join(models_dir, "model_metadata.json")

    joblib.dump(best_model, model_save_path)
    joblib.dump(preprocessor, preprocessor_save_path)

    metadata = {
        "model_name": best_model_name,
        "training_timestamp": datetime.now().isoformat(),
        "target_variable": TARGET_COLUMN,
        "features_used": ALL_FEATURE_COLUMNS,
        "transformed_feature_count": int(X_train.shape[1]),
        "validation_metrics": best_val_metrics,
        "test_metrics": best_test_metrics,
        "best_hyperparameters": getattr(best_model, "get_params", lambda: {})(),
        "historical_stats": hist_stats,
        "dataset_record_counts": {
            "total_records": len(feat_df),
            "train_records": len(train_df),
            "validation_records": len(val_df),
            "test_records": len(test_df)
        }
    }

    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, default=str)

    logger.info(f"Saved best model to: {model_save_path}")
    logger.info(f"Saved preprocessor to: {preprocessor_save_path}")
    logger.info(f"Saved metadata to: {metadata_save_path}")

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "comparison_df": comp_df,
        "experiment_df": exp_df
    }


if __name__ == "__main__":
    train_and_tune_models()
