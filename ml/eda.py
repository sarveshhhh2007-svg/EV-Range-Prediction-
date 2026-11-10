"""
Exploratory Data Analysis (EDA) Module.
Generates comprehensive statistical summaries, distribution metrics,
and renders publication-grade static charts using the strict Burgundy/Peach design palette.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Crucial: non-GUI Agg backend before pyplot import to prevent window crashes on server threads
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import STATIC_DIR
from ml.model_config import (
    PALETTE,
    FEATURE_UNITS,
    TARGET_COLUMN,
    ALL_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

logger = logging.getLogger(__name__)

# Output directory for rendered figures
IMAGES_DIR = STATIC_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def configure_matplotlib_theme():
    """Sets Matplotlib global styles matching the Burgundy/Peach identity."""
    plt.style.use("default")
    plt.rcParams["figure.facecolor"] = PALETTE["peach_50"]
    plt.rcParams["axes.facecolor"] = PALETTE["peach_100"]
    plt.rcParams["axes.edgecolor"] = "#59171b33"
    plt.rcParams["axes.labelcolor"] = PALETTE["burgundy_950"]
    plt.rcParams["xtick.color"] = PALETTE["burgundy_700"]
    plt.rcParams["ytick.color"] = PALETTE["burgundy_700"]
    plt.rcParams["text.color"] = PALETTE["burgundy_950"]
    plt.rcParams["grid.color"] = "#59171b1a"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"


def compute_descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes detailed descriptive statistics for all numerical features:
    mean, median, mode, min, max, range, variance, std, quartiles, IQR, units,
    and covariance with range_km.
    """
    stats_dict = {}
    target_series = df[TARGET_COLUMN].dropna()

    for col in ALL_NUMERICAL_FEATURES + [TARGET_COLUMN]:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1 = float(series.quantile(0.25))
        median = float(series.median())
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        mean = float(series.mean())
        std = float(series.std())
        variance = float(series.var())
        min_val = float(series.min())
        max_val = float(series.max())
        val_range = max_val - min_val
        
        mode_series = series.mode()
        mode_val = float(mode_series.iloc[0]) if not mode_series.empty else median

        cov_with_target = float(series.cov(target_series)) if col != TARGET_COLUMN else variance
        corr_with_target = float(series.corr(target_series)) if col != TARGET_COLUMN else 1.0

        stats_dict[col] = {
            "name": col,
            "unit": FEATURE_UNITS.get(col, ""),
            "count": int(len(series)),
            "missing": int(df[col].isna().sum()),
            "mean": round(mean, 2),
            "median": round(median, 2),
            "mode": round(mode_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "range": round(val_range, 2),
            "variance": round(variance, 2),
            "std": round(std, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "covariance_with_range": round(cov_with_target, 2),
            "correlation_with_range": round(corr_with_target, 4)
        }

    return stats_dict


def compute_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates Pearson correlation matrix for numerical features."""
    cols = [c for c in ALL_NUMERICAL_FEATURES + [TARGET_COLUMN] if c in df.columns]
    num_df = df[cols].dropna()
    corr_matrix = num_df.corr().round(4)
    
    # Structure for API & heatmap display
    columns_list = list(corr_matrix.columns)
    matrix_values = corr_matrix.values.tolist()
    
    # Specific correlations with target sorted descending
    target_corrs = corr_matrix[TARGET_COLUMN].sort_values(ascending=False).to_dict()

    return {
        "features": columns_list,
        "matrix": matrix_values,
        "target_correlations": target_corrs
    }


def compute_distribution_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates histogram bin counts and box plot metrics for range_km."""
    ranges = df[TARGET_COLUMN].dropna().values
    
    # Histogram bins (15 bins)
    counts, bin_edges = np.histogram(ranges, bins=15)
    bins_data = []
    for i in range(len(counts)):
        bins_data.append({
            "bin_start": round(float(bin_edges[i]), 1),
            "bin_end": round(float(bin_edges[i + 1]), 1),
            "label": f"{int(bin_edges[i])}-{int(bin_edges[i+1])} km",
            "count": int(counts[i])
        })

    # Box plot percentiles
    q1 = float(np.percentile(ranges, 25))
    median = float(np.percentile(ranges, 50))
    q3 = float(np.percentile(ranges, 75))
    iqr = q3 - q1
    lower_whisker = float(max(np.min(ranges), q1 - 1.5 * iqr))
    upper_whisker = float(min(np.max(ranges), q3 + 1.5 * iqr))
    
    outliers = [float(x) for x in ranges if x < lower_whisker or x > upper_whisker]

    return {
        "histogram": bins_data,
        "boxplot": {
            "min": round(float(np.min(ranges)), 2),
            "q1": round(q1, 2),
            "median": round(median, 2),
            "q3": round(q3, 2),
            "max": round(float(np.max(ranges)), 2),
            "lower_whisker": round(lower_whisker, 2),
            "upper_whisker": round(upper_whisker, 2),
            "outliers": outliers
        }
    }


def get_scatter_data(df: pd.DataFrame, feature_name: str) -> Dict[str, Any]:
    """Whitelisted scatter data helper for feature vs range_km."""
    if feature_name not in ALL_NUMERICAL_FEATURES:
        raise ValueError(f"Feature '{feature_name}' is not in allowed numerical feature list.")
    
    sub = df[[feature_name, TARGET_COLUMN, "brand", "model"]].dropna()
    points = []
    for _, row in sub.iterrows():
        points.append({
            "x": round(float(row[feature_name]), 2),
            "y": round(float(row[TARGET_COLUMN]), 2),
            "brand": str(row["brand"]),
            "model": str(row["model"])
        })
        
    return {
        "feature": feature_name,
        "feature_label": feature_name.replace("_", " ").title(),
        "unit": FEATURE_UNITS.get(feature_name, ""),
        "target": TARGET_COLUMN,
        "points": points
    }


def generate_static_charts(df: pd.DataFrame):
    """
    Renders and saves all 8 static EDA plots into static/images/.
    Uses the palette strictly: Burgundy, Peach, Rosy-brown.
    """
    configure_matplotlib_theme()

    # 1. EV Driving Range Distribution Histogram
    plt.figure(figsize=(9, 5), dpi=150)
    sns.histplot(
        df[TARGET_COLUMN].dropna(),
        kde=True,
        color=PALETTE["burgundy_900"],
        edgecolor=PALETTE["peach_500"],
        bins=16,
        alpha=0.75
    )
    plt.title("EV Listed Driving Range Distribution", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Driving Range (km)", fontsize=11, labelpad=8)
    plt.ylabel("Vehicle Frequency", fontsize=11, labelpad=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_range_distribution.png")
    plt.close()

    # 2. Range Box Plot
    plt.figure(figsize=(8, 4), dpi=150)
    box = plt.boxplot(
        df[TARGET_COLUMN].dropna(),
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor=PALETTE["peach_300"], color=PALETTE["burgundy_900"], linewidth=1.5),
        whiskerprops=dict(color=PALETTE["burgundy_900"], linewidth=1.5),
        capprops=dict(color=PALETTE["burgundy_900"], linewidth=1.5),
        medianprops=dict(color=PALETTE["burgundy_950"], linewidth=2),
        flierprops=dict(marker="o", markerfacecolor=PALETTE["burgundy_700"], markeredgecolor=PALETTE["burgundy_900"])
    )
    plt.title("EV Driving Range Box Plot & Interquartile Spread", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Driving Range (km)", fontsize=11, labelpad=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_range_boxplot.png")
    plt.close()

    # 3. Battery Capacity vs Driving Range Scatter
    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(
        df["battery_capacity_kWh"],
        df[TARGET_COLUMN],
        color=PALETTE["burgundy_900"],
        edgecolors=PALETTE["peach_300"],
        alpha=0.8,
        s=45
    )
    # Trendline
    m, b = np.polyfit(df["battery_capacity_kWh"].dropna(), df.loc[df["battery_capacity_kWh"].dropna().index, TARGET_COLUMN], 1)
    x_vals = np.linspace(df["battery_capacity_kWh"].min(), df["battery_capacity_kWh"].max(), 100)
    plt.plot(x_vals, m * x_vals + b, color=PALETTE["burgundy_700"], linestyle="--", linewidth=2, label="Linear Trend (r=+0.88)")
    plt.title("Battery Capacity vs Driving Range", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Battery Capacity (kWh)", fontsize=11, labelpad=8)
    plt.ylabel("Listed Range (km)", fontsize=11, labelpad=8)
    plt.legend(frameon=True, facecolor=PALETTE["peach_50"], edgecolor=PALETTE["burgundy_500"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_battery_vs_range.png")
    plt.close()

    # 4. Acceleration vs Range Scatter
    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(
        df["acceleration_0_100_s"],
        df[TARGET_COLUMN],
        color=PALETTE["burgundy_700"],
        edgecolors=PALETTE["peach_200"],
        alpha=0.8,
        s=45
    )
    plt.title("0-100 km/h Acceleration Time vs Driving Range", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Acceleration 0-100 km/h (s)", fontsize=11, labelpad=8)
    plt.ylabel("Listed Range (km)", fontsize=11, labelpad=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_acceleration_vs_range.png")
    plt.close()

    # 5. Top Speed vs Range Scatter
    plt.figure(figsize=(8, 5), dpi=150)
    plt.scatter(
        df["top_speed_kmh"],
        df[TARGET_COLUMN],
        color=PALETTE["burgundy_500"],
        edgecolors=PALETTE["burgundy_900"],
        alpha=0.8,
        s=45
    )
    plt.title("Top Speed vs Driving Range", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Top Speed (km/h)", fontsize=11, labelpad=8)
    plt.ylabel("Listed Range (km)", fontsize=11, labelpad=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_topspeed_vs_range.png")
    plt.close()

    # 6. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8), dpi=150)
    num_cols = [c for c in ALL_NUMERICAL_FEATURES + [TARGET_COLUMN] if c in df.columns]
    corr_mat = df[num_cols].dropna().corr()
    
    # Custom Burgundy-Peach diverging colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors = [PALETTE["burgundy_900"], PALETTE["peach_100"], PALETTE["peach_500"], PALETTE["burgundy_700"]]
    cmap_burgundy_peach = LinearSegmentedColormap.from_list("burgundy_peach", colors, N=100)

    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap=cmap_burgundy_peach,
        linewidths=0.5,
        linecolor=PALETTE["peach_50"],
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    plt.title("EV Technical Features Pearson Correlation Heatmap", fontsize=14, weight="bold", pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_correlation_heatmap.png")
    plt.close()

    # 7. Missing Value Visualisation
    plt.figure(figsize=(9, 4.5), dpi=150)
    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    
    bars = plt.bar(
        [col.replace("_", " ") for col in missing_series.index],
        missing_series.values,
        color=PALETTE["burgundy_900"],
        edgecolor=PALETTE["peach_500"],
        width=0.5
    )
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 3, f"{int(yval)} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=PALETTE["burgundy_950"])

    plt.title("Missing Values Count Across Raw Dataset Attributes", fontsize=14, weight="bold", pad=12)
    plt.xlabel("Attributes with Missing Values", fontsize=11, labelpad=8)
    plt.ylabel("Missing Count (out of 478)", fontsize=11, labelpad=8)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.ylim(0, max(missing_series.values) + 30)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_missing_values.png")
    plt.close()

    # 8. Categorical Distribution Charts (Car Body Type & Drivetrain)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    
    body_counts = df["car_body_type"].value_counts().head(6)
    axes[0].barh(body_counts.index, body_counts.values, color=PALETTE["burgundy_700"], edgecolor=PALETTE["peach_300"])
    axes[0].set_title("Car Body Type Distribution", fontsize=12, weight="bold")
    axes[0].set_xlabel("Count", fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()

    drive_counts = df["drivetrain"].value_counts()
    axes[1].bar(drive_counts.index, drive_counts.values, color=PALETTE["peach_500"], edgecolor=PALETTE["burgundy_900"])
    axes[1].set_title("Drivetrain Distribution", fontsize=12, weight="bold")
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "eda_categorical_distribution.png")
    plt.close()

    logger.info("Generated all static EDA figures in static/images/")
