# Faculty Viva Voce Examination Guide & Model Answers
## Electric Vehicle Driving-Range Prediction & ML Analytics Platform
**Course:** Machine Learning (21CSC305P) — Continuous Learning Assessment 2 (CLA-2), Task 3  
**Candidate:** Sarvesh HS (Register No: RA2411026050213)  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  

---

### Q1: Why is this problem formulated as regression rather than classification?
**Answer:** The target variable `range_km` is an inherently continuous physical measurement of distance in kilometres (ranging empirically from 135 km to 685 km). Formulating range prediction as classification would require discretizing the continuous spectrum into arbitrary bins (e.g., Short, Medium, Long), thereby discarding ordinal magnitude, penalizing boundary errors artificially, and destroying precise quantitative estimation. Supervised regression estimates numerical target values directly.

---

### Q2: Why did you choose these three specific regression models?
**Answer:** The three models represent a principled hierarchy of supervised learning paradigms:
1. **Linear Regression:** An interpretable parametric baseline that captures first-order physical energy conservation relationships and provides transparent standardized coefficients.
2. **Random Forest Regressor:** A non-parametric bootstrap-aggregated (bagging) ensemble that reduces variance, handles feature interactions automatically, and demonstrates resistance to outliers without requiring feature scaling.
3. **Gradient Boosting Regressor:** A sequential boosting ensemble that minimizes pseudo-residuals iteratively, prioritizing difficult edge cases across non-linear aerodynamic and power regimes.

---

### Q3: What does $R^2$ signify, and why is it scientifically incorrect to term it "accuracy"?
**Answer:** $R^2$ is the **coefficient of determination**, defining the proportion of variance in the dependent variable explained by the regression model relative to a naive mean-predicting benchmark:
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
"Accuracy" refers strictly to the ratio of correct discrete classifications in classification tasks. In continuous regression, exact matching is statistically near-zero; calling $R^2$ "accuracy" is an academic misnomer because a high $R^2$ can still coincide with substantial absolute error if the underlying variance is massive.

---

### Q4: How was data leakage strictly prevented during preprocessing?
**Answer:** Data leakage was prevented through three structural safeguards:
1. **Pipeline Encapsulation:** All scalers, imputers, and encoders were placed inside scikit-learn `Pipeline` and `ColumnTransformer` objects and fitted solely on the training split ($N = 382$).
2. **Untouched Test Split:** An 80/20 train/test split was established with fixed `random_state=42`. Hyperparameter tuning (GridSearchCV) was restricted entirely to internal training folds.
3. **Programmatic Assertion:** A strict runtime assertion checked that `range_km` and identifier columns were absent from feature matrices before training.

---

### Q5: Why were `model` and `source_url` excluded from the feature matrix?
**Answer:** 
- `model` contains 477 distinct non-missing values across 478 records (99.8% uniqueness). One-hot encoding `model` would create 477 sparse columns, allowing models to simply memorize vehicle names rather than learning physical dynamics.
- `source_url` is a technical web identifier containing zero physical information about vehicle performance.

---

### Q6: Why was `battery_type` excluded from model inputs?
**Answer:** Programmatic audit confirmed that `battery_type` holds the identical value `Lithium-ion` across 100% of the 478 rows. A feature with zero variance carries zero entropy and zero predictive information, contributing only dimensionality noise.

---

### Q7: How did you handle `number_of_cells` given that 42.3% of its values were missing?
**Answer:** Rather than dropping rows or omitting the feature entirely, `number_of_cells` was isolated in a dedicated pipeline branch using `SimpleImputer(strategy="median", add_indicator=True)`. This performs median imputation while simultaneously synthesizing a binary missingness indicator column. The model can thus learn both the median cell configuration and whether missingness itself correlates with vehicle segments. The ablation study proved that retaining this indicator reduced 5-fold CV RMSE from 17.92 km to 17.76 km.

---

### Q8: How were qualitative entries like "10 Banana Boxes" handled in `cargo_volume_l`?
**Answer:** The dataset contained text entries such as "10 Banana Boxes", "31 Banana Boxes", and "13 Banana Boxes". Inventing arbitrary liter conversion factors would introduce synthetic bias. Instead, `pd.to_numeric(..., errors="coerce")` was used to convert these unparseable entries into `NaN`. The numerical median imputer inside the pipeline then imputed principled values based on verified training trunk sizes.

---

### Q9: What did the mandatory target-leakage audit find regarding `efficiency_wh_per_km`?
**Answer:** Physically, range is defined by battery capacity divided by specific energy consumption:
$$\text{implied\_range} = \frac{\text{battery\_capacity\_kWh} \times 1000}{\text{efficiency\_wh\_per\_km}}$$
Because official efficiency figures are often derived directly during laboratory range homologation, including efficiency introduces partial target leakage. The audit showed a **0.9321 Pearson correlation** between implied range and actual listed range, with 20.71% of records within 10% error. Therefore, high model $R^2$ scores partly reflect this underlying physical formula.

---

### Q10: What did the ablation study reveal when `efficiency_wh_per_km` was removed?
**Answer:** When efficiency was completely excluded from training:
- Linear Regression test $R^2$ changed from 0.9700 to **0.9640** (RMSE: 19.52 km).
- Random Forest test $R^2$ changed from 0.9502 to **0.9596** (RMSE: 20.68 km).
- Gradient Boosting test $R^2$ changed from 0.9654 to **0.9789** (RMSE: 14.96 km).
This proves that the models do not merely rely on efficiency as a shortcut; they successfully learn range dynamics from battery capacity, vehicle dimensions, torque, and acceleration.

---

### Q11: Why was Linear Regression recommended over complex ensemble models?
**Answer:** The final recommendation was derived from **5-fold cross-validation on the training set**, not the test set:
- Linear Regression achieved the lowest validation RMSE of **17.76 km** ($\pm 1.84$ km) and highest CV $R^2$ of **0.9699**.
- While Gradient Boosting achieved a marginally lower test MAE (13.94 km vs 14.37 km), Linear Regression provides instant deterministic training ($<0.05$ s vs $>40$ s), zero risk of ensemble overfitting, and completely interpretable coefficients directly grounded in energy conservation laws.

---

### Q12: What is the difference between Permutation Feature Importance and Impurity (MDI) Importance?
**Answer:**
- **MDI (Mean Decrease in Impurity):** Measures how much decision tree splits reduce Gini/variance during training. It is computationally fast but heavily biased toward high-cardinality features and reflects training fit rather than generalization.
- **Permutation Importance:** Shuffles one feature's values on the **held-out test set** and measures the resulting degradation in RMSE. It directly quantifies how much the model depends on that feature for unseen real-world predictions.

---

### Q13: How was unseen-brand generalization evaluated?
**Answer:** Unseen-brand generalization was evaluated using `GroupKFold` grouped by `brand` on the training set. This simulates deployment where the model predicts range for a new manufacturer unseen during training. Validation RMSE increased from 17.76 km to 26.24 km, reflecting the fact that boutique manufacturers have unique motor geometries and proprietary battery enclosures not fully captured by general vehicle dimensions.

---

### Q14: What does the Random Forest "tree-spread" represent in the prediction interface?
**Answer:** In the UI, the tree-spread indicates the minimum and maximum predictions across all 300 individual decision trees in the ensemble for that specific vehicle. It is explicitly labelled as **"spread across trees"** and **not a confidence interval**, because individual decision trees in bagging are not calibrated posterior probability distributions.

---

### Q15: What are the primary engineering limitations of this platform?
**Answer:**
1. **Homologation Standard vs Real World:** Predictions estimate listed WLTP/EPA laboratory ratings measured at 20–23°C. Real-world winter temperatures (-10°C) with resistive cabin heating reduce range by 20–35%.
2. **Lithium-Ion Chemistry Granularity:** All records share a generic Lithium-ion designation without distinguishing LFP from high-nickel NMC811.
3. **Driver Behavior:** Aggressive highway driving ($>130$ km/h) dramatically increases aerodynamic drag ($F_d \propto v^2$), which cannot be modeled from static vehicle spec sheets alone.

---

### Q16: How does this project integrate with UN SDG 9?
**Answer:** The project directly aligns with **SDG 9: Industry, Innovation and Infrastructure**, specifically Target 9.4 (upgrading technological capabilities of industrial sectors and promoting sustainable infrastructure). By providing an open, reproducible analytics platform, it allows engineering students, fleet managers, and researchers to assess EV battery sizing and efficiency trade-offs to facilitate the transition to clean transportation infrastructure.

