"""
Model Evaluation and Metrics Module.
Calculates MAE, MSE, RMSE, R², Naive baseline, cross-validation,
GroupKFold unseen brand testing, permutation feature importances,
range-band error stratification, and generates evaluation plots.
"""
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.inspection import permutation_importance

from config import STATIC_DIR
from ml.model_config import (
    PALETTE,
    MODEL_COLORS,
    TARGET_COLUMN
)

logger = logging.getLogger(__name__)
IMAGES_DIR = STATIC_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, MSE, RMSE, and R² for predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }


def compute_naive_baseline(y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """
    Computes performance of a dummy/naive baseline predicting the training mean.
    Serves as zero-skill benchmark for R² and error metrics.
    """
    mean_val = float(y_train.mean())
    y_pred_naive = np.full(shape=len(y_test), fill_value=mean_val)
    metrics = calculate_metrics(y_test.values, y_pred_naive)
    
    return {
        "model_name": "Naive Baseline (Training Mean)",
        "predicted_constant_km": round(mean_val, 2),
        "metrics": metrics,
        "description": "Predicts constant training target mean across all test instances"
    }


def compute_range_band_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Computes MAE, RMSE and count stratified into vehicle range bands:
    - Short Range (< 300 km)
    - Mid Range (300 km - 450 km)
    - Long Range (> 450 km)
    """
    bands = [
        ("Short Range (< 300 km)", lambda y: y < 300),
        ("Mid Range (300 - 450 km)", lambda y: (y >= 300) & (y <= 450)),
        ("Long Range (> 450 km)", lambda y: y > 450)
    ]
    
    band_results = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    for band_name, condition in bands:
        mask = condition(y_true)
        count = int(np.sum(mask))
        if count > 0:
            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            mae = float(mean_absolute_error(sub_true, sub_pred))
            rmse = float(np.sqrt(mean_squared_error(sub_true, sub_pred)))
            band_results.append({
                "band": band_name,
                "count": count,
                "mae": round(mae, 2),
                "rmse": round(rmse, 2)
            })
        else:
            band_results.append({
                "band": band_name,
                "count": 0,
                "mae": None,
                "rmse": None
            })
            
    return band_results


def evaluate_cross_validation(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Runs standard 5-fold cross-validation on the training set."""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2"
    }
    
    scores = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring)
    
    mae_scores = -scores["test_mae"]
    mse_scores = -scores["test_mse"]
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = scores["test_r2"]
    
    return {
        "mae_mean": round(float(np.mean(mae_scores)), 2),
        "mae_std": round(float(np.std(mae_scores)), 2),
        "rmse_mean": round(float(np.mean(rmse_scores)), 2),
        "rmse_std": round(float(np.std(rmse_scores)), 2),
        "r2_mean": round(float(np.mean(r2_scores)), 4),
        "r2_std": round(float(np.std(r2_scores)), 4),
        "folds": cv_folds
    }


def evaluate_unseen_brand_cv(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series
) -> Dict[str, Any]:
    """
    Runs GroupKFold grouped by brand on training set to evaluate unseen-brand generalization.
    Reports limitations if brand frequencies are highly imbalanced.
    """
    # Filter brands with at least 2 samples if possible or use available groups
    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    
    gkf = GroupKFold(n_splits=n_splits)
    maes = []
    rmses = []
    r2s = []
    
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Fit on fold training
        pipeline.fit(X_tr, y_tr)
        val_preds = pipeline.predict(X_val)
        
        maes.append(mean_absolute_error(y_val, val_preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, val_preds)))
        r2s.append(r2_score(y_val, val_preds))
        
    limitation_note = (
        "GroupKFold by brand simulates real-world generalization to completely new vehicle manufacturers. "
        "Because several boutique brands have only 1-2 models while major brands (e.g. BYD, Audi, Tesla) "
        "dominate, group variance is higher than random k-fold."
    )
    
    return {
        "n_groups": int(unique_groups),
        "n_splits": int(n_splits),
        "mae_mean": round(float(np.mean(maes)), 2),
        "mae_std": round(float(np.std(maes)), 2),
        "rmse_mean": round(float(np.mean(rmses)), 2),
        "rmse_std": round(float(np.std(rmses)), 2),
        "r2_mean": round(float(np.mean(r2s)), 4),
        "r2_std": round(float(np.std(r2s)), 4),
        "limitation_note": limitation_note
    }


def compute_feature_importances(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Computes permutation importance on test set (primary) and
    impurity/coefficient importance (secondary) for model interpretation.
    """
    # 1. Permutation importance on pipeline (evaluated on test set)
    perm_res = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )
    
    feature_names = list(X_test.columns)
    perm_means = perm_res.importances_mean
    perm_stds = perm_res.importances_std
    
    perm_list = []
    for f, mean_imp, std_imp in zip(feature_names, perm_means, perm_stds):
        perm_list.append({
            "feature": f,
            "feature_label": f.replace("_", " ").title(),
            "importance_mean": round(float(mean_imp), 4),
            "importance_std": round(float(std_imp), 4)
        })
    perm_list.sort(key=lambda x: x["importance_mean"], reverse=True)

    # 2. Extract model-native importances / coefficients
    model_step = pipeline.named_steps["regressor"]
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Get transformed feature names from ColumnTransformer
    try:
        encoded_feature_names = preprocessor.get_feature_names_out()
    except Exception:
        encoded_feature_names = [f"feat_{i}" for i in range(100)]
        
    secondary_importance = []
    importance_type = "None"
    
    if hasattr(model_step, "feature_importances_"):
        importance_type = "MDI (Mean Decrease in Impurity)"
        raw_importances = model_step.feature_importances_
        for name, val in zip(encoded_feature_names, raw_importances):
            # Clean transformer prefix for clarity (e.g. num_general__battery_capacity_kWh)
            clean_name = name.split("__")[-1]
            secondary_importance.append({
                "feature": clean_name,
                "importance": round(float(val), 4)
            })
        secondary_importance.sort(key=lambda x: x["importance"], reverse=True)
        secondary_importance = secondary_importance[:15]  # Top 15 features
        
    elif hasattr(model_step, "coef_"):
        importance_type = "Standardized Regression Coefficients"
        coefs = model_step.coef_
        for name, val in zip(encoded_feature_names, coefs):
            clean_name = name.split("__")[-1]
            secondary_importance.append({
                "feature": clean_name,
                "coefficient": round(float(val), 4),
                "abs_magnitude": round(float(abs(val)), 4)
            })
        secondary_importance.sort(key=lambda x: x["abs_magnitude"], reverse=True)
        secondary_importance = secondary_importance[:15]

    return {
        "model_name": model_name,
        "primary_permutation_test_set": perm_list,
        "secondary_type": importance_type,
        "secondary_importance": secondary_importance,
        "note": "Permutation importance measures drop in performance when a feature is shuffled on held-out test data, reflecting genuine generalizable reliance. Impurity-based importance measures splits in the training set and can be biased toward high-cardinality features."
    }


def generate_evaluation_charts(
    evaluation_results: Dict[str, Any],
    test_actual: np.ndarray,
    predictions_dict: Dict[str, np.ndarray]
):
    """Generates all required static model evaluation charts in static/images/."""
    # 1. Model comparison metric bars (MAE, RMSE, R²)
    models = list(predictions_dict.keys())
    maes = [evaluation_results[m]["test_metrics"]["mae"] for m in models]
    rmses = [evaluation_results[m]["test_metrics"]["rmse"] for m in models]
    r2s = [evaluation_results[m]["test_metrics"]["r2"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=150)
    
    # MAE
    axes[0].bar(models, maes, color=[MODEL_COLORS.get(m, PALETTE["burgundy_900"]) for m in models], edgecolor=PALETTE["burgundy_950"])
    axes[0].set_title("Mean Absolute Error (MAE)", weight="bold", fontsize=11)
    axes[0].set_ylabel("MAE (km) — Lower is better", fontsize=10)
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.5, f"{v:.2f}", ha="center", weight="bold", fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    
    # RMSE
    axes[1].bar(models, rmses, color=[MODEL_COLORS.get(m, PALETTE["burgundy_900"]) for m in models], edgecolor=PALETTE["burgundy_950"])
    axes[1].set_title("Root Mean Squared Error (RMSE)", weight="bold", fontsize=11)
    axes[1].set_ylabel("RMSE (km) — Lower is better", fontsize=10)
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.5, f"{v:.2f}", ha="center", weight="bold", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    # R2
    axes[2].bar(models, r2s, color=[MODEL_COLORS.get(m, PALETTE["burgundy_900"]) for m in models], edgecolor=PALETTE["burgundy_950"])
    axes[2].set_title("Variance Explained (R²)", weight="bold", fontsize=11)
    axes[2].set_ylabel("R² — Higher is better", fontsize=10)
    axes[2].set_ylim(0.9, 1.0)
    for i, v in enumerate(r2s):
        axes[2].text(i, v + 0.003, f"{v:.4f}", ha="center", weight="bold", fontsize=10)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eval_model_comparison.png")
    plt.close()

    # 2. Actual vs Predicted Scatter with y = x reference line
    plt.figure(figsize=(8, 7), dpi=150)
    markers = {"Linear Regression": "o", "Random Forest": "s", "Gradient Boosting": "^"}
    
    for m in models:
        plt.scatter(
            test_actual,
            predictions_dict[m],
            label=f"{m} (R²={evaluation_results[m]['test_metrics']['r2']:.4f})",
            color=MODEL_COLORS.get(m, PALETTE["burgundy_900"]),
            alpha=0.7,
            s=40,
            marker=markers.get(m, "o")
        )

    # y = x reference line
    min_val = min(np.min(test_actual), min(np.min(p) for p in predictions_dict.values())) - 20
    max_val = max(np.max(test_actual), max(np.max(p) for p in predictions_dict.values())) + 20
    plt.plot([min_val, max_val], [min_val, max_val], color=PALETTE["burgundy_950"], linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)")

    plt.title("Actual vs. Predicted EV Listed Driving Range (Held-Out Test Set)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Actual Listed Range (km)", fontsize=11, labelpad=8)
    plt.ylabel("Predicted Range (km)", fontsize=11, labelpad=8)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(frameon=True, facecolor=PALETTE["peach_50"], edgecolor=PALETTE["burgundy_500"], loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eval_actual_vs_predicted.png")
    plt.close()

    # 3. Residual Distribution & Residuals vs Predicted
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    
    for m in models:
        res = test_actual - predictions_dict[m]
        # Distribution
        sns.kdeplot(res, ax=axes[0], label=m, color=MODEL_COLORS.get(m, PALETTE["burgundy_900"]), linewidth=2)
        # Residuals vs Predicted
        axes[1].scatter(
            predictions_dict[m],
            res,
            label=m,
            color=MODEL_COLORS.get(m, PALETTE["burgundy_900"]),
            alpha=0.6,
            s=35,
            marker=markers.get(m, "o")
        )

    axes[0].axvline(0, color=PALETTE["burgundy_950"], linestyle="--", linewidth=1.5)
    axes[0].set_title("Prediction Residual Distribution (Actual - Pred)", weight="bold", fontsize=11)
    axes[0].set_xlabel("Residual Error (km)", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].legend(facecolor=PALETTE["peach_50"])
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0, color=PALETTE["burgundy_950"], linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs Predicted Range", weight="bold", fontsize=11)
    axes[1].set_xlabel("Predicted Range (km)", fontsize=10)
    axes[1].set_ylabel("Residual (km)", fontsize=10)
    axes[1].legend(facecolor=PALETTE["peach_50"])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eval_residual_analysis.png")
    plt.close()

    logger.info("Generated evaluation charts in static/images/")

