# Comparative Analysis of Machine Learning Regression Models for Electric Vehicle Driving-Range Prediction

**Academic Assessment:** Continuous Learning Assessment 2 (CLA-2), Task 3 — Case-Based Implementation  
**Course:** Machine Learning (Course Code: 21CSC305P)  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**Degree Programme:** B.Tech Computer Science and Engineering — Artificial Intelligence and Machine Learning  
**Semester:** V  
**Lead Contributor:** Sarvesh HS (Register Number: RA2411026050213)  
**Application Domain:** Electric Vehicles / Intelligent Transportation Systems  
**Primary UN SDG Alignment:** SDG 9 — Industry, Innovation and Infrastructure  
**Secondary UN SDG Alignment:** SDG 7 — Affordable and Clean Energy  

---

### Collaborative Team Contribution Block
| Role | Contributor Name | Register Number | Individual Academic Contribution |
| :--- | :--- | :--- | :--- |
| **Lead Developer & ML Engineer** | Sarvesh HS | RA2411026050213 | System architecture design, data loading & verification, preprocessing pipeline construction, model implementation & cross-validation tuning, Flask REST API development, vanilla frontend engineering, and end-to-end testing. |
| **Team Member 2** | [Team Member 2 — Name] | [Register No.] | [Individual contribution: Literature survey of EV range prediction paradigms, exploratory data analysis documentation, and assisting in residual error analysis.] |
| **Team Member 3** | [Team Member 3 — Name] | [Register No.] | [Individual contribution: UN Sustainable Development Goals (SDG 9 & 7) alignment research, technical documentation drafting, and viva voce presentation preparation.] |

---

## 1. Abstract
The global transition toward electrified mobility is intrinsically tied to consumer confidence in advertised vehicle range and transparent energy metrics. This study investigates the predictive modelling of electric vehicle (EV) listed driving range (`range_km`) based on vehicle architectural, electrical, and dimensional specifications. Using a benchmark dataset comprising 478 contemporary 2025 production electric vehicles across 22 attributes, three supervised regression paradigms—**Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**—were developed, optimized, and rigorously evaluated. To guarantee scientific validity and prevent data leakage, all imputation, scaling, and categorical encoding steps were encapsulated within scikit-learn pipelines, maintaining a strictly untouched 20% held-out test set ($N = 96$). 

A physics-based target leakage audit revealed that physical implied range ($\text{Battery Capacity} \times 1000 / \text{Efficiency}$) shares a 0.9321 Pearson correlation with listed range; mandatory ablation studies demonstrated that models maintain high predictive power ($R^2 > 0.95$, MAE $< 16$ km) even when efficiency is excluded. On the held-out test set, Linear Regression achieved an $R^2$ of 0.9700 and RMSE of 17.82 km (5-fold CV RMSE: $17.76 \pm 1.84$ km), outperforming Random Forest ($R^2 = 0.9502$, RMSE = 22.96 km) and matching Gradient Boosting ($R^2 = 0.9654$, RMSE = 19.13 km, MAE = 13.94 km). Linear Regression was selected as the primary recommended architecture due to its superior cross-validation generalization, deterministic execution, and transparent physical interpretability. An interactive full-stack web platform was deployed to enable real-time inference and model inspection.

---

## 2. Introduction & Problem Statement
Range anxiety—the apprehension that an electric vehicle possesses insufficient energy storage to reach its destination—remains a principal impediment to widespread electric vehicle adoption. Standardized homologation cycles (such as the Worldwide Harmonised Light Vehicles Test Procedure [WLTP] and EPA five-cycle tests) assign official driving-range ratings to vehicles. However, prospective buyers, automotive engineers, and fleet strategists frequently require rapid, data-driven estimates of how specific engineering trade-offs (e.g., expanding battery capacity, increasing motor torque, optimizing curb aerodynamics, or altering chassis dimensions) influence rated driving range.

The fundamental objective of this project is to construct a robust, leakage-free machine learning platform that:
1. Analyzes the empirical distribution and inter-feature correlations of modern electric vehicles.
2. Formulates, tunes, and compares three distinct supervised machine learning regression algorithms under identical train/test conditions.
3. Conducts a critical audit of target leakage and feature ablations.
4. Provides an accessible, production-ready web platform for comparative range inference.

### Scope Clarification & Boundary Constraints
It is essential to clarify the analytical boundary of this investigation:
- The target variable is `range_km`, representing the **official listed laboratory test range**.
- The models **do not** predict Battery State of Health (SOH), chemical battery degradation across cycling aging, remaining battery calendar life, or real-world driving range under arbitrary sub-zero winter ambient temperatures or high-speed headwind conditions.

---

## 3. Dataset Description & Statistical Audit
The dataset utilized in this assessment is sourced from the curated Kaggle repository *Electric Car Performance and Battery Dataset (2025)*:
- **Filename:** `electric_vehicles_spec_2025.csv.csv` (SHA-256: `92312ec41bebcd359db1931b83ed450534f56a5899e6f232fc3262a6a4bcc4ca`)
- **Dimensions:** 478 records across 22 attributes
- **Exact Duplicate Rows:** 0 (verified programmatically)

### Complete Dataset Schema (22 Attributes)
1. `brand` (Categorical, 47 manufacturers)
2. `model` (Text/Categorical designation, 477 unique values — excluded from training to prevent memorization)
3. `top_speed_kmh` (Continuous numerical, km/h)
4. `battery_capacity_kWh` (Continuous numerical, gross pack kWh)
5. `battery_type` (Categorical, constant `Lithium-ion` across all 478 records — excluded from training)
6. `number_of_cells` (Discrete numerical, 202 missing values [42.26%])
7. `torque_nm` (Continuous numerical, Nm, 7 missing values)
8. `efficiency_wh_per_km` (Continuous numerical, Wh/km)
9. `range_km` (**Target Variable**, continuous numerical in km)
10. `acceleration_0_100_s` (Continuous numerical, seconds)
11. `fast_charging_power_kw_dc` (Continuous numerical, peak kW DC, 1 missing value)
12. `fast_charge_port` (Categorical, e.g. CCS, Tesla, CHAdeMO, 1 missing value)
13. `towing_capacity_kg` (Continuous numerical, braked kg, 26 missing values)
14. `cargo_volume_l` (Text with mixed numbers and unparseable banana-box strings, 1 missing value)
15. `seats` (Discrete numerical, passenger seating count)
16. `drivetrain` (Categorical: AWD, RWD, FWD)
17. `segment` (Categorical market segment: B, C, D, E, F, N)
18. `length_mm` (Continuous numerical, mm)
19. `width_mm` (Continuous numerical, mm)
20. `height_mm` (Continuous numerical, mm)
21. `car_body_type` (Categorical: SUV, Sedan, Hatchback, Estate, Liftback, Coupe)
22. `source_url` (Identifier URL — excluded from training)

### Target Variable Summary (`range_km`)
- **Count:** 478 valid observations
- **Mean:** 393.18 km
- **Median:** 397.50 km
- **Minimum:** 135.00 km (e.g., ultra-compact urban runabouts)
- **Maximum:** 685.00 km (e.g., luxury executive grand tourers)
- **Standard Deviation:** 103.29 km
- **Interquartile Range (IQR):** 150.00 km ($Q_1 = 320.00$ km, $Q_3 = 470.00$ km)

---

## 4. Methodology & Data Preprocessing
To adhere strictly to professional ML engineering practices, preprocessing is divided into dataset hygiene (clean string formatting, numeric coercion) and pipeline-encapsulated transformations.

### 4.1 Data Cleaning & Hygiene
1. **Whitespace Trimming:** Leading and trailing whitespace was stripped from all 22 column names and string observations.
2. **Cargo Volume Coercion:** The attribute `cargo_volume_l` contained qualitative volume estimates in text form (`10 Banana Boxes`, `31 Banana Boxes`, `13 Banana Boxes`). Rather than synthesizing arbitrary liter conversions, `pd.to_numeric(..., errors="coerce")` converted these three entries to `NaN`. They are subsequently handled alongside genuine missing values by the median imputer.
3. **Numeric Type Standardization:** All 13 candidate numerical columns were verified and cast to float64/int64.
4. **Zero-Variance Feature Pruning:** `battery_type` holds `Lithium-ion` across 100% of rows. Having zero variance, it was excluded from feature matrices.
5. **High-Cardinality Identifier Removal:** `model` contains 477 unique values among 478 records. Including unique names causes memorization and severe overfitting; it was excluded from training while retained in the Dataset Explorer for human inspection. `source_url` was likewise excluded.

### 4.2 Leakage-Free Pipeline Architecture
Data splitting occurred prior to any scaling, imputation, or categorical parameter estimation:
- **Split Ratio:** 80% Train ($N = 382$) / 20% Test ($N = 96$) with fixed `random_state = 42`.
- **General Numerical Branch:** Handled via `SimpleImputer(strategy="median")` + `StandardScaler()` (for Linear Regression).
- **Dedicated Cell Feature Branch:** Because `number_of_cells` exhibits 42.26% missingness, it was routed to `SimpleImputer(strategy="median", add_indicator=True)`. This creates an explicit binary missingness indicator column, allowing models to discern whether cell count was explicitly recorded.
- **Categorical Branch:** Handled via `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.

```
                    ┌─────────────────────────┐
                    │   Input Features (X)    │
                    └────────────┬────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ General Numeric  │   │ number_of_cells  │   │ Categoricals     │
│ (12 columns)     │   │ (1 column)       │   │ (5 columns)      │
├──────────────────┤   ├──────────────────┤   ├──────────────────┤
│ Median Imputer   │   │ Imputer +        │   │ Modal Imputer    │
│ + StandardScaler │   │ MissingIndicator │   │ + OneHotEncoder  │
└────────┬─────────┘   └─────────┬────────┘   └─────────┬────────┘
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Transformed Matrix    │
                    │      (101 Columns)      │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Regressor Pipeline    │
                    │   (LR / RF / GBR)       │
                    └─────────────────────────┘
```

---

## 5. Target-Leakage Audit & Feature Ablations

### 5.1 Physics-Based Target Leakage Audit
From first principles of electrical traction physics:
$$\text{implied\_range} = \frac{\text{battery\_capacity\_kWh} \times 1000}{\text{efficiency\_wh\_per\_km}}$$

In automotive homologation databases, vehicle energy consumption (Wh/km) is frequently computed by test facilities after recording test distance and battery energy draw. Therefore, `efficiency_wh_per_km` carries latent target proxy information.

**Audit Findings:**
- **Pearson Correlation (Implied vs Actual Range):** $r = 0.9321$
- **Median Absolute Percentage Difference:** $18.94\%$
- **Share of Records within 5% Error:** $11.30\%$
- **Share of Records within 10% Error:** $20.71\%$

*Conclusion:* While a strong physical dependency exists, real-world drag, motor losses, and auxiliary draws prevent a trivial identity function. Crucially, high model $R^2$ scores partly reflect this physical definition. To assess whether models can predict range without this proxy, a mandatory ablation was conducted.

### 5.2 Mandatory Ablation 1: Omission of `efficiency_wh_per_km`
| Model Architecture | Full Feature Set Test $R^2$ | Full Feature Set Test RMSE | Ablated Feature Set Test $R^2$ (Without Efficiency) | Ablated Feature Set Test RMSE (Without Efficiency) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.9700 | 17.82 km | **0.9640** | 19.52 km |
| **Random Forest** | 0.9502 | 22.96 km | **0.9596** | 20.68 km |
| **Gradient Boosting** | 0.9654 | 19.13 km | **0.9789** | 14.96 km |

*Takeaway:* When efficiency is omitted, all models retain an $R^2 > 0.95$. Gradient Boosting and Random Forest adapt by leveraging battery capacity, curb dimensions, motor torque, and acceleration to reconstruct energy demand, demonstrating robust predictive capability.

### 5.3 Mandatory Ablation 2: Value of `number_of_cells` and Missingness Indicator
| Model Architecture | 5-Fold CV RMSE With Cells & Indicator | 5-Fold CV RMSE Without Cells | Difference in RMSE ($\Delta$) | Impact on Generalization |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 17.76 km | 17.92 km | $+0.16$ km | Omitting cells degrades CV performance |
| **Random Forest** | 23.15 km | 23.39 km | $+0.24$ km | Omitting cells degrades CV performance |
| **Gradient Boosting** | 18.85 km | 19.21 km | $+0.36$ km | Omitting cells degrades CV performance |

*Takeaway:* Retaining `number_of_cells` with median imputation and a missingness indicator consistently reduces validation error across all three algorithms.

---

## 6. Model Evaluation, Tuning & Comparison
Hyperparameters were tuned via 5-fold `GridSearchCV` strictly on the training partitions using negative RMSE scoring:
- **Random Forest Best Params:** `n_estimators=300`, `max_depth=10`, `min_samples_split=2`, `min_samples_leaf=1` (Train CV RMSE: 21.67 km).
- **Gradient Boosting Best Params:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=4`, `subsample=0.8` (Train CV RMSE: 17.39 km).

### 6.1 Final Benchmark Results on Untouched Test Set ($N = 96$)
All values are computed from the real test set; alongside, 5-fold cross-validation and unseen-brand GroupKFold results are reported:

| Model Architecture | Test MAE (km) | Test MSE ($\text{km}^2$) | Test RMSE (km) | Test $R^2$ | 5-Fold CV RMSE (km) | 5-Fold CV $R^2$ | Unseen-Brand CV RMSE (km) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Train Mean)** | 88.06 | 10,591.24 | 102.91 | -0.0003 | — | — | — |
| **Linear Regression** | 14.37 | 317.42 | **17.82** | **0.9700** | **17.76 ± 1.84** | **0.9699** | 26.24 ± 4.12 |
| **Random Forest Regressor** | 16.13 | 527.14 | 22.96 | 0.9502 | 23.15 ± 1.95 | 0.9491 | 29.81 ± 5.30 |
| **Gradient Boosting Regressor** | **13.94** | 366.07 | 19.13 | 0.9654 | 18.85 ± 1.16 | 0.9661 | 25.10 ± 3.88 |

### 6.2 Error Stratification by Vehicle Range Bands
To diagnose where models struggle, test set errors were stratified into three operational range categories:
| Range Band | Sample Count | Linear Regression MAE | Random Forest MAE | Gradient Boosting MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Short Range ($< 300$ km)** | 18 | 15.22 km | 17.80 km | 14.95 km |
| **Mid Range ($300 - 450$ km)** | 48 | 12.84 km | 14.12 km | 12.60 km |
| **Long Range ($> 450$ km)** | 30 | 16.28 km | 18.35 km | 15.50 km |

*Observation:* Mid-range vehicles exhibit the lowest error across all architectures ($12.60 - 14.12$ km MAE) due to higher training data density. Long-range flagship models exhibit slightly higher dispersion due to specialized aerodynamic wheel packages and varying motor efficiencies.

### 6.3 Permutation Feature Importance (Held-Out Test Set)
Permutation importance on held-out test data revealed the true generalizable reliance of the ensemble:
1. `battery_capacity_kWh`: Test RMSE drop of $+36.42 \pm 3.12$ km (Predominant predictive driver)
2. `efficiency_wh_per_km`: Test RMSE drop of $+18.25 \pm 2.05$ km
3. `top_speed_kmh`: Test RMSE drop of $+4.82 \pm 0.88$ km
4. `acceleration_0_100_s`: Test RMSE drop of $+3.15 \pm 0.62$ km
5. `torque_nm`: Test RMSE drop of $+2.40 \pm 0.45$ km

---

## 7. Model Selection & Academic Justification
Per assessment criteria, the final recommended model is selected based on **cross-validation stability on the training set**, rather than isolated test-set performance:
- **Recommended Model:** **Linear Regression**
- **Justification:** Linear Regression achieved the lowest 5-fold cross-validation RMSE of **17.76 km** with a standard deviation of $\pm 1.84$ km and an $R^2$ of 0.9699. While Gradient Boosting achieved a marginally lower test MAE (13.94 km vs 14.37 km), Linear Regression demonstrates remarkable training efficiency ($<0.05$ s vs $>40$ s tuning), zero stochastic seed drift, and complete parameter transparency through standardized coefficients. Furthermore, EV driving range is governed predominantly by physical energy conservation, making a multilinear formulation mathematically grounded and resistant to overfitting.

---

## 8. UN Sustainable Development Goal (SDG) Integration
This study directly engages with **SDG 9: Industry, Innovation and Infrastructure**, specifically Target 9.4 (upgrading infrastructure and retrofitting industries to make them sustainable, with increased resource-use efficiency and greater adoption of clean and environmentally sound technologies).

By providing an open, reproducible machine learning framework that evaluates EV specifications against listed range:
- Automotive engineers can explore multi-objective trade-offs between battery pack mass and range returns.
- Fleet managers can evaluate vehicle suitability for commercial logistics without requiring proprietary telematics software.
- The project demonstrates how data-driven machine learning methodologies can analyze transport technologies to foster resilient, intelligent mobility infrastructure.
- As a secondary contextual connection, the platform aligns with **SDG 7 (Affordable and Clean Energy)** by fostering technical understanding of energy consumption rates in electrified drivetrains.

*Clarification:* This academic analytical software does not directly reduce greenhouse gas emissions or construct physical charging corridors; its value lies in educational exploration and engineering analytics.

---

## 9. Limitations & Future Research
1. **Homologation Standard vs Real-World Ambient Losses:** The dataset compiles laboratory homologation ratings (WLTP/EPA). In real-world operation, sub-zero ambient temperatures (-10°C to 0°C) increase battery internal resistance and demand up to 5 kW of electrical power for cabin heating, reducing effective range by 20% to 35%.
2. **Homogeneous Battery Chemistry:** All records denote `Lithium-ion`. Granular distinctions (NMC811 vs LFP vs NCA) are unrepresented in the public Kaggle schema. Future work should incorporate cathode chemistry sub-classifications.
3. **Vehicle Age & Thermal Management:** As open datasets expand, integrating battery pack cooling architectures (liquid immersion vs air cooling) will yield richer engineering insights.

---

## 10. References
1. Afnan Saif Afnan (2025). *Electric Car Performance and Battery Dataset*. Kaggle Dataset. Available at: `https://www.kaggle.com/datasets/afnansaifafnan/electric-car-performance-and-battery-dataset`
2. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*, 29(5), 1189–1232.
5. United Nations Department of Economic and Social Affairs (2015). *Sustainable Development Goal 9: Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation*. United Nations.

