"""
Model Evaluation and Diagnostic Reporting Module
================================================
Project: Classroom Attendance Prediction Using Academic Schedule and Historical Attendance Data

Calculates:
- Academic regression metrics: MAE, RMSE, MAPE, R² Score
- Generates `reports/model_comparison.csv`
- Generates `reports/experiment_results.csv`
- Evaluates Feature Importances (Tree MDI & Permutation Importance)
- Produces publication-grade evaluation figures in `reports/figures/`
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard aesthetic styling for figures
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 300


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard academic regression evaluation metrics:
    - MAE: Mean Absolute Error (lower is better)
    - RMSE: Root Mean Squared Error (lower is better)
    - MAPE: Mean Absolute Percentage Error (lower is better)
    - R2: Coefficient of Determination (higher is better, max 1.0)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # Avoid division by zero in MAPE
    nonzero_mask = y_true > 1e-4
    if np.any(nonzero_mask):
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
    else:
        mape = 0.0

    r2 = float(r2_score(y_true, y_pred))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 2),
        "R2": round(r2, 4)
    }


def save_model_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    output_path: str = os.path.join("reports", "model_comparison.csv")
) -> pd.DataFrame:
    """
    Formats and saves the multi-model comparison table.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    df_comp = df_comp.sort_values(by="R2", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_comp.to_csv(output_path, index=False)
    logger.info(f"Model comparison table saved to: {output_path}")

    return df_comp


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = os.path.join("reports", "figures", "model_comparison_metrics.png")
):
    """
    Plots multi-metric bar chart comparing all candidate algorithms.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    metrics = ["MAE", "RMSE", "MAPE (%)", "R2"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(comparison_df["Model"], comparison_df[metric], color=colors[idx], width=0.55, edgecolor="black", alpha=0.85)
        ax.set_title(f"Model Comparison - {metric}", fontweight="bold", fontsize=12)
        ax.set_ylabel(metric, fontweight="bold")
        ax.set_xticklabels(comparison_df["Model"], rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # Annotate bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Model comparison chart saved to: {output_path}")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str = os.path.join("reports", "figures", "actual_vs_predicted.png")
):
    """
    Creates an Actual vs. Predicted scatter plot with perfect agreement reference line.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.65, color="#1f77b4", edgecolors="w", s=50, label="Test Observations")
    
    # 45-degree reference line
    min_val = min(np.min(y_true), np.min(y_pred), 30)
    max_val = max(np.max(y_true), np.max(y_pred), 100)
    plt.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Ideal Prediction (y = x)")

    plt.title(f"Actual vs. Predicted Attendance Percentage ({model_name})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Attendance (%)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Attendance (%)", fontsize=11, fontweight="bold")
    plt.xlim(min_val - 2, max_val + 2)
    plt.ylim(min_val - 2, max_val + 2)
    plt.legend(frameon=True, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Actual vs Predicted chart saved to: {output_path}")


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: str = os.path.join("reports", "figures", "residual_distribution.png")
):
    """
    Plots residual errors distribution (Residual = Actual - Predicted).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#3498db", edgecolor="black", bins=20)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")

    plt.title(f"Residual Error Distribution ({model_name})", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Residual Error (Actual % - Predicted %)", fontsize=11, fontweight="bold")
    plt.ylabel("Frequency", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Residual distribution chart saved to: {output_path}")


def extract_and_plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
    output_path: str = os.path.join("reports", "figures", "feature_importance.png")
) -> pd.DataFrame:
    """
    Extracts Gini/MDI feature importance from tree-based regressor and generates bar plot.
    Adheres strictly to non-causal language.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        logger.warning("Model does not expose feature_importances_ or coef_.")
        return pd.DataFrame()

    feat_df = pd.DataFrame({
        "Feature": feature_names[:len(importances)],
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    top_feats = feat_df.head(top_n).copy()

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_feats["Feature"][::-1], top_feats["Importance"][::-1], color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.title(f"Top {top_n} Factors Associated with Classroom Attendance Predictions", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Relative Feature Importance Score", fontsize=11, fontweight="bold")
    plt.ylabel("Feature", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.annotate(f"{width:.3f}",
                     xy=(width, bar.get_y() + bar.get_height() / 2),
                     xytext=(5, 0),
                     textcoords="offset points",
                     ha="left", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance chart saved to: {output_path}")

    return feat_df


def compute_and_plot_permutation_importance(
    model: Any,
    X_val_or_test: np.ndarray,
    y_val_or_test: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
    output_path: str = os.path.join("reports", "figures", "permutation_importance.png")
) -> pd.DataFrame:
    """
    Computes Permutation Feature Importance on test data.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    logger.info("Computing Permutation Feature Importance...")

    perm_res = permutation_importance(
        model, X_val_or_test, y_val_or_test,
        n_repeats=10, random_state=42, scoring="neg_mean_absolute_error"
    )

    perm_df = pd.DataFrame({
        "Feature": feature_names[:len(perm_res.importances_mean)],
        "Importance_Mean": perm_res.importances_mean,
        "Importance_Std": perm_res.importances_std
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    top_perm = perm_df.head(top_n).copy()

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_perm["Feature"][::-1],
        top_perm["Importance_Mean"][::-1],
        xerr=top_perm["Importance_Std"][::-1],
        color="#9467bd", edgecolor="black", alpha=0.85, capsize=4
    )
    plt.title(f"Top {top_n} Permutation Feature Importances (Test Set MAE Impact)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Mean Drop in Test Performance (MAE Increase)", fontsize=11, fontweight="bold")
    plt.ylabel("Feature", fontsize=11, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    return perm_df


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Computes standard academic classification evaluation metrics:
    - Accuracy: Overall correct predictions
    - Precision: Positive predictive value
    - Recall: Sensitivity / True Positive Rate
    - F1-Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under the ROC curve
    - Specificity: True Negative Rate
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc = 0.5
    if y_prob is not None:
        try:
            roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc = 0.5
    else:
        try:
            roc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            roc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(roc, 4),
        "Specificity": round(spec, 4)
    }


def save_classification_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    output_path: str = os.path.join("reports", "classification_comparison.csv")
) -> pd.DataFrame:
    """
    Formats and saves the multi-classifier comparison table.
    """
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Classifier": model_name}
        row.update(metrics)
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    df_comp = df_comp.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df_comp.to_csv(output_path, index=False)
    logger.info(f"Classification comparison table saved to: {output_path}")

    return df_comp


def plot_classification_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = os.path.join("reports", "figures", "classification_comparison_metrics.png")
):
    """
    Plots multi-metric bar chart comparing all 7 classification algorithms.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sorted_df = comparison_df.sort_values(by=metric, ascending=False)
        bars = ax.bar(sorted_df["Classifier"], sorted_df[metric], color=palette[:len(sorted_df)], edgecolor="black", alpha=0.85)
        ax.set_title(f"{metric} Comparison", fontsize=11, fontweight="bold")
        ax.set_ylabel(metric, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(sorted_df["Classifier"], rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Classification Benchmark: Identifying At-Risk (<75%) vs. Compliant (>=75%) Attendance", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Classification comparison plot saved to: {output_path}")


def plot_confusion_matrices(
    fitted_models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_path: str = os.path.join("reports", "figures", "classification_confusion_matrices.png")
):
    """
    Generates a grid of confusion matrices for all fitted classification models.
    """
    from sklearn.metrics import confusion_matrix
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    n_models = len(fitted_models)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, (name, model) in enumerate(fitted_models.items()):
        preds = model.predict(X_val)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["High (>=75%)", "At-Risk (<75%)"],
                    yticklabels=["High (>=75%)", "At-Risk (<75%)"])
        axes[idx].set_title(f"{name}", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("True Label", fontsize=9)
        axes[idx].set_xlabel("Predicted Label", fontsize=9)

    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Confusion Matrices: Attendance Risk Classification on Validation Split", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrices plot saved to: {output_path}")


def plot_roc_curves(
    fitted_models: Dict[str, Any],
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_path: str = os.path.join("reports", "figures", "classification_roc_curves.png")
):
    """
    Generates ROC Curves for all fitted classifiers.
    """
    from sklearn.metrics import roc_curve, auc
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    plt.figure(figsize=(9, 7))
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)")

    for name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_val)
        else:
            probs = model.predict(X_val)

        fpr, tpr, _ = roc_curve(y_val, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Benchmark", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curves plot saved to: {output_path}")

