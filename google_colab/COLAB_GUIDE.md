# Google Colab Execution & Demonstration Guide
## Electric Vehicle Driving-Range Prediction & ML Analytics

This directory contains everything you need to showcase the Python code, models, training workflows, and outputs directly inside **Google Colab**.

---

## What is Inside this `google_colab/` Folder?

1. **`EV_Range_Prediction_Colab_Master.ipynb`**
   - The master Jupyter/Colab notebook with **all 25 sections**.
   - **Pre-executed with all outputs saved**: charts, tables, hyperparameter tuning results, cross-validation scores, and live prediction comparisons are already rendered. When you open it in Colab, faculty and reviewers will see the full execution history immediately.
   - Includes automatic Google Drive mount and Google Colab interactive file upload fallbacks.

2. **`colab_pipeline_runner.py`**
   - A standalone Python execution script.
   - When run in a Colab code cell (`!python colab_pipeline_runner.py`), it executes the entire end-to-end pipeline and prints clear, formatted banners for each stage:
     - Dataset inspection & SHA-256 verification
     - Preprocessing & cargo volume coercion
     - Target leakage audit
     - Train/test splitting (80/20)
     - Model 1: Linear Regression
     - Model 2: Random Forest tuning
     - Model 3: Gradient Boosting tuning
     - Model benchmark comparison table
     - Mandatory ablations (with/without efficiency, with/without cells)
     - Live test inference vs ground-truth actual range

3. **`models/` Directory**
   - Contains the pre-trained, serialized scikit-learn pipelines:
     - `linear_regression.joblib`
     - `random_forest.joblib`
     - `gradient_boosting.joblib`
     - `model_metadata.json`

4. **`data/` Directory**
   - Contains the benchmark CSV: `electric_vehicles_spec_2025.csv.csv`.

5. **`outputs/` Directory**
   - Contains all generated high-resolution PNG plots (Histograms, Boxplots, Scatter charts, Correlation heatmaps, Residual distributions, Model comparison bars).

---

## How to Run in Google Colab (2 Easy Methods)

### Method 1: Upload the Notebook Directly (Fastest — 30 Seconds)
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File** → **Upload notebook** → Select `EV_Range_Prediction_Colab_Master.ipynb`.
3. The notebook will open with **all 43 cells and charts already rendered**!
4. If you want to re-run all cells live:
   - In the left sidebar, click the **Folder icon** (Files).
   - Drag and drop `electric_vehicles_spec_2025.csv.csv` into the file area.
   - Click **Runtime** → **Run all**.

### Method 2: Run the Standalone Pipeline Script
In any Colab notebook cell, upload `colab_pipeline_runner.py` and `electric_vehicles_spec_2025.csv.csv`, then run:
```python
!python colab_pipeline_runner.py
```
This prints the entire training lifecycle, metrics table, and live predictions directly into the Colab cell output.

---

## Summary of Results Visible in Colab
- **Dataset Dimensions:** 478 rows, 22 columns (0 duplicates, SHA-256 verified).
- **Linear Regression:** Test RMSE: **17.82 km**, Test $R^2$: **0.9700** (5-Fold CV RMSE: **17.76 km** — Recommended Model).
- **Random Forest:** Test RMSE: **20.10 km**, Test $R^2$: **0.9618**.
- **Gradient Boosting:** Test MAE: **11.76 km**, Test RMSE: **15.82 km**, Test $R^2$: **0.9764**.
- **Efficiency Ablation:** Test $R^2 > 0.95$ maintained even when efficiency is excluded.

