# Comparative Analysis of Machine Learning Regression Models for Electric Vehicle Driving-Range Prediction

**Machine Learning Continuous Learning Assessment 2 (CLA-2), Task 3 — Case-Based Implementation**  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**Degree Programme:** B.Tech Computer Science and Engineering — Artificial Intelligence and Machine Learning (Semester V)  
**Lead Contributor:** Sarvesh HS (Register Number: RA2411026050213)  
**Application Domain:** Electric Vehicles / Intelligent Transportation Systems  
**UN Sustainable Development Goals:** Primary: **SDG 9** (Industry, Innovation & Infrastructure), Secondary: **SDG 7** (Affordable & Clean Energy)  

---

### Collaborative Team Contribution Block
| Contributor | Register Number | Role & Primary Contributions |
| :--- | :--- | :--- |
| **Sarvesh HS (Lead)** | RA2411026050213 | System architecture design, preprocessing pipelines, model tuning & evaluation, Flask REST API, vanilla CSS/JS dashboard, test suite, and report. |
| **[Team Member 2 — Name]** | [Register No.] | [Individual contribution: Literature review of EV range modeling, exploratory data analysis documentation, and assisting in residual error analysis.] |
| **[Team Member 3 — Name]** | [Register No.] | [Individual contribution: UN Sustainable Development Goals (SDG 9 & 7) alignment research, technical documentation drafting, and viva voce presentation preparation.] |

---

## 1. Project Overview & Architecture
This platform delivers a complete, production-grade supervised machine learning system to predict the **official listed driving range** (`range_km`) of modern electric vehicles based on their architectural, electrical, and dimensional specifications. Three distinct regression paradigms—**Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**—are trained, tuned, and compared under strictly identical train/test splits.

### Key Capabilities
- **End-to-End Scikit-Learn Pipelines:** Leakage-free preprocessing using `Pipeline` and `ColumnTransformer` with median/modal imputation, missingness indicators for cell counts, and one-hot encoding.
- **Physics-Based Target Leakage Audit:** Rigorous empirical evaluation comparing physical implied range ($\text{Battery Capacity} \times 1000 / \text{Efficiency}$) against listed range.
- **Mandatory Feature Ablation Studies:** Cross-validated assessments with and without `number_of_cells` and `efficiency_wh_per_km`.
- **Interactive Multi-Model Prediction Studio:** Dynamic form generation from saved model schemas, client and server-side validation, test-set vehicle auto-loading, ensemble tree-spread calculation, and historical MAE reporting.
- **Strict Visual Identity:** Elegant academic user interface built exclusively in vanilla HTML5, CSS3, and modern modular ES JavaScript using an exact Burgundy (`#59171B`), Warm Peach (`#FED7B8`), and Rosy-Brown (`#8B4947`) palette.
- **100% Offline Capability:** Vendored Chart.js library with zero external runtime API or CDN dependencies.

### Scope Clarification
The target variable is strictly `range_km`. The platform predicts the **official listed laboratory test range** (e.g. WLTP/EPA). It does **not** predict:
- Battery State of Health (SOH)
- Battery degradation over time or cycle aging
- Remaining battery pack lifetime
- Real-world driving range under arbitrary sub-zero winter temperatures or adverse road gradients.

---

## 2. Technology Stack
- **Backend:** Python 3.10+, Flask 3.1, Pandas 2.3, NumPy 2.2, Scikit-learn 1.7, Joblib 1.6, Matplotlib 3.10 (Agg non-GUI backend), Seaborn 0.13.
- **Frontend:** HTML5, CSS3, Vanilla Modern JavaScript (ES Modules), Vendored Chart.js (v4.4.7 UMD), Inline SVG Sprite Icons.
- **Testing & Quality Assurance:** Pytest 9.2 (20 automated unit and integration tests passing).
- **Environment:** Runs locally in VS Code or any standard terminal; Jupyter Notebook compatible locally and in Google Colab.

---

## 3. Dataset Source & Integrity
- **Dataset File:** `electric_vehicles_spec_2025.csv.csv` (the double extension is intentional and defined as a constant in `ml/model_config.py`).
- **Source:** [Kaggle Electric Car Performance and Battery Dataset (2025)](https://www.kaggle.com/datasets/afnansaifafnan/electric-car-performance-and-battery-dataset).
- **Verified SHA-256 Hash:** `92312ec41bebcd359db1931b83ed450534f56a5899e6f232fc3262a6a4bcc4ca`.
- **Dimensions:** 478 vehicle rows, 22 columns, 0 exact duplicate records.

---

## 4. Directory Structure
```
EV-Range-Prediction/
│
├── app.py                      # Production Flask REST API and web application server
├── config.py                   # Global configuration, directory paths, and server settings
├── requirements.txt            # Pinned dependency ranges
├── README.md                   # Complete system documentation and execution guide
├── .gitignore                  # Git exclusions for venv, cache, and OS files
├── generate_notebook.py        # Automated notebook builder
│
├── data/
│   ├── electric_vehicles_spec_2025.csv.csv  # Raw benchmark dataset
│   └── cleaned_electric_vehicles.csv        # Preprocessed downloadable dataset
│
├── ml/
│   ├── __init__.py             # Package marker
│   ├── model_config.py         # Feature sets, constants, schema, and palette tokens
│   ├── data_loader.py          # Encoding fallback, validation, and SHA-256 hashing
│   ├── preprocessing.py        # Cleaning log, leakage audit, IQR outliers, sklearn pipelines
│   ├── eda.py                  # Descriptive stats, correlations, distributions, static figures
│   ├── train.py                # Split, GridSearchCV tuning, ablations, artifact serialization
│   ├── evaluate.py             # MAE/MSE/RMSE/R2, CV, GroupKFold, permutation importance
│   └── predict.py              # Schema generator, test-set example loader, multi-model inference
│
├── models/
│   ├── linear_regression.joblib    # Serialized Linear Regression pipeline artifact
│   ├── random_forest.joblib        # Serialized Random Forest pipeline artifact
│   ├── gradient_boosting.joblib    # Serialized Gradient Boosting pipeline artifact
│   └── model_metadata.json         # Complete hyperparameters, training metrics, and feature ranges
│
├── reports/
│   ├── model_comparison.json       # Benchmark metrics, CV results, and ablations
│   ├── dataset_summary.json        # Raw audit and descriptive statistics summary
│   ├── preprocessing_summary.json  # Data cleaning log and outlier observations
│   └── technical_report.md         # Comprehensive academic technical research report
│
├── notebooks/
│   └── ev_range_analysis.ipynb     # Executed 25-section Jupyter notebook with saved outputs
│
├── docs/
│   └── VIVA_NOTES.md               # 16 detailed faculty viva voce questions and model answers
│
├── templates/
│   └── index.html                  # Semantic single-page dashboard template
│
├── static/
│   ├── css/
│   │   └── styles.css              # Custom stylesheet with Burgundy/Peach design tokens
│   ├── js/
│   │   ├── app.js                  # Main dashboard orchestrator and table pagination
│   │   ├── api.js                  # REST API client with in-memory caching
│   │   ├── charts.js               # Interactive Chart.js controllers and table alternatives
│   │   ├── motion.js               # Scroll-spy, count-ups, and reduced-motion handling
│   │   └── prediction.js           # Dynamic form validation and inference presentation
│   ├── vendor/
│   │   ├── chart.umd.min.js        # Vendored offline Chart.js
│   │   └── chart.js                # Local alias
│   └── images/                     # Generated high-resolution publication figures
│
└── tests/
    ├── conftest.py                 # Pytest fixtures and Flask test client
    ├── test_preprocessing.py       # Data loading, cleaning, and leakage assertion tests
    ├── test_models.py              # Pipeline training, metrics, and reproducibility tests
    └── test_api.py                 # REST endpoint, HTTP status codes, and JSON safety tests
```

---

## 5. Quick-Start Execution Instructions

### Step 1: Environment Setup
Open a terminal (PowerShell, Command Prompt, or Bash) in the project directory:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Model Training Pipeline (Mandatory First Step)
> [!IMPORTANT]
> Always execute `python -m ml.train` first before launching the server. This performs dataset loading, preprocessing, static chart generation, cross-validation tuning, ablations, and writes all `.joblib` and `.json` artifacts into `models/` and `reports/`.

```bash
python -m ml.train
```

### Step 4: Launch Web Application
```bash
python app.py
```

### Step 5: Open Dashboard
Navigate to `http://127.0.0.1:5000` in any modern web browser.

---

## 6. Verification Checklist
To confirm system health and reproducibility:
1. **Training Verification:** `python -m ml.train` runs to completion and writes `models/*.joblib` and `models/model_metadata.json`.
2. **API Verification:** `curl http://127.0.0.1:5000/api/health` returns HTTP 200 with `status: "healthy"` and `models_available: true`.
3. **Overview Verification:** The dashboard hero displays 478 records, 18 features, 3 models, and status `Trained Models Active (Best: Linear Regression)`.
4. **Prediction Studio Verification:** Click **"Load Test Set Example EV"** in the Prediction Studio (loads CUPRA Tavascan Endurance, actual range 445.0 km), click **"Predict EV Driving Range"**, and verify all three models predict near the benchmark (LR: 415.3 km, RF: 447.6 km, GBR: 455.4 km).
5. **Automated Tests:** Run `pytest -q` to verify that all 20 tests pass.

```bash
pytest -q
```
*Expected Output:*
```
.................... [100%]
20 passed in 0.81s
```

---

## 7. Model Evaluation Summary (Held-Out Test Set, $N = 96$)
| Model Architecture | Test MAE (km) | Test MSE ($\text{km}^2$) | Test RMSE (km) | Test $R^2$ (Variance) | 5-Fold CV RMSE (km) | 5-Fold CV $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Train Mean)** | 88.06 | 10,591.24 | 102.91 | -0.0003 | — | — |
| **Linear Regression Baseline** | 14.37 | 317.42 | **17.82** | **0.9700** | **17.76 ± 1.84** | **0.9699** |
| **Random Forest Regressor** | 16.13 | 527.14 | 22.96 | 0.9502 | 23.15 ± 1.95 | 0.9491 |
| **Gradient Boosting Regressor** | **13.94** | 366.07 | 19.13 | 0.9654 | 18.85 ± 1.16 | 0.9661 |

*Final Recommendation:* **Linear Regression** was selected based on 5-fold cross-validation RMSE on the training set (17.76 km), physical interpretability, and deterministic stability.

---

## 8. REST API Endpoints Reference
| Endpoint | Method | Purpose | HTTP Status Codes |
| :--- | :---: | :--- | :--- |
| `/api/health` | GET | System health, dataset verification, model status, hash checks | 200 |
| `/api/summary` | GET | Summary record counts, features, target variable, model status | 200, 500 |
| `/api/dataset/preview` | GET | Server-side paginated preview (`page`, `page_size`, `search`, `sort`, `view`) | 200, 500 |
| `/api/dataset/schema` | GET | Full metadata for all 22 columns with missing counts and types | 200, 500 |
| `/api/dataset/download` | GET | Download preprocessed dataset as CSV file | 200 |
| `/api/preprocessing` | GET | Data cleaning log, IQR outlier table, and target leakage audit | 200, 500 |
| `/api/eda` | GET | Descriptive statistics, quartiles, and range covariances | 200, 500 |
| `/api/eda/correlation` | GET | Pearson correlation matrix and target correlation rankings | 200, 500 |
| `/api/eda/distribution` | GET | Target histogram bin frequencies and boxplot percentiles | 200, 500 |
| `/api/eda/scatter` | GET | Whitelisted bivariate scatter points for selected feature vs range | 200, 400, 500 |
| `/api/models/status` | GET | Training metadata, timestamps, and model availability | 200 |
| `/api/models/metrics` | GET | Test metrics, cross-validation, ablations, and recommendations | 200, 503 |
| `/api/models/importance`| GET | Permutation test-set importance and MDI / coefficients | 200, 503 |
| `/api/models/train` | POST | Trigger background model retraining with concurrency locking | 202, 409, 429 |
| `/api/models/train/status`| GET | Poll current background training progress message | 200 |
| `/api/predict/schema` | GET | Dynamic form schema with units, boundaries, and categories | 200, 503 |
| `/api/predict/example` | GET | Load genuine vehicle from held-out test set with actual range | 200, 503 |
| `/api/predict` | POST | Execute multi-model range prediction with domain validation | 200, 415, 422, 503 |

---

## 9. Assumptions & Engineering Defaults
1. **Double Extension Handling:** The dataset file is strictly named `electric_vehicles_spec_2025.csv.csv`. This constant is defined uniquely in `ml/model_config.py` and referenced project-wide.
2. **Missing Cell Imputation:** `number_of_cells` exhibits 42.26% missingness. Instead of dropping records, median imputation plus an explicit `MissingIndicator` was used in `ColumnTransformer`.
3. **Cargo Volume Coercion:** Banana-box string entries (`10 Banana Boxes`, etc.) were coerced to `NaN` using `pd.to_numeric` and imputed with the training median without inventing synthetic liter equivalents.
4. **Feature Exclusions:** `range_km` is strictly target; `model` and `source_url` were excluded to prevent memorization; `battery_type` was excluded due to zero variance (`Lithium-ion` everywhere).
5. **Prediction Input Handling:** Unspecified optional fields in the Prediction Studio are automatically assigned the training median (for numeric) or mode (for categoricals) by the trained imputer pipeline.
6. **Concurrency Controls:** Retraining requests use a threading lock and return HTTP 409 if already active, followed by a 10-second cooldown period to prevent rapid accidental retries.

---

## 10. Troubleshooting Guide
| Issue Encountered | Root Cause | Solution |
| :--- | :--- | :--- |
| **`FileNotFoundError: Dataset file not found`** | Working directory is not project root, or CSV was moved. | Ensure terminal prompt is at `.../TASK 2/` and `data/electric_vehicles_spec_2025.csv.csv` exists. |
| **`HTTP 503: Models not yet trained`** | Web app was started before running the training pipeline. | Execute `python -m ml.train` once, then restart `python app.py`. |
| **Port 5000 in use (especially macOS)** | macOS AirPlay Receiver occupies port 5000 by default. | Run `FLASK_PORT=5001 python app.py` or disable AirPlay Receiver in System Settings. |
| **Scikit-learn version mismatch warning** | Models trained on different sklearn version than runtime. | Re-run `python -m ml.train` within the active virtual environment to re-serialize artifacts. |
| **Windows execution policy error (`PSSecurityException`)** | PowerShell script execution disabled on Windows. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating venv. |
| **Chart.js not loading offline** | Browser blocking local scripts or missing vendor file. | Ensure `static/vendor/chart.umd.min.js` exists; the app loads locally vendored Chart.js. |

---

## 11. Viva Voce Examination Walkthrough
To demonstrate this project during technical viva evaluation:
1. **Step 1 — Integrity Check:** Open the **Overview** section; point out the exact dataset SHA-256 hash, 478 records, and zero duplicate instances.
2. **Step 2 — Target Leakage Audit:** Navigate to **Data Preprocessing**; explain the physical formula $\text{Battery Capacity} \times 1000 / \text{Efficiency}$ ($r = 0.9321$) and show the ablation table proving high model performance without efficiency.
3. **Step 3 — Benchmarks:** Open **Model Comparison**; explain why Linear Regression was selected over Random Forest based on cross-validation RMSE rather than test-set bias.
4. **Step 4 — Live Prediction:** Open **EV Range Prediction Studio**; click *"Load Test Set Example EV"*, observe the actual listed range (445.0 km), click *"Predict EV Driving Range"*, and review the 3-model comparison bars and Random Forest tree spread.

