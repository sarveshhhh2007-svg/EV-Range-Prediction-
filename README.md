# Comparative Analysis of Machine Learning Regression Models for Electric Vehicle Driving-Range Prediction

A clean 5-section web application comparing three machine learning regression models (**Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**) for predicting electric vehicle listed driving range from technical specifications.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models (if not already trained):**
   ```bash
   python -m ml.train
   ```

3. **Start the web application:**
   ```bash
   python app.py
   ```

4. **Open in your browser:**
   ```
   http://127.0.0.1:5000
   ```

---

## Web Application Sections

1. **Home:** Overview, listed-range scope clarification, key volume metrics (478 cars, 18 specs, 3 models), workflow steps, and best-model finding.
2. **Data:** Interactive 6-column vehicle specifications table with real-time search, pagination, and a 6-point data preparation summary.
3. **Insights:** Four visual charts illustrating range distribution, battery capacity relation, acceleration time relation, and top feature correlations.
4. **Models:** Performance comparison table (MAE, MSE, RMSE, R²), MAE error benchmark chart, and actual vs predicted scatter plot.
5. **Predict:** Interactive prediction form with the 6 primary specifications, optional collapsed secondary fields, example loader, and side-by-side comparison bars across all three models.

---

## Documentation References

- [Website Guide](docs/WEBSITE_GUIDE.md): Complete plain-language walkthrough of every visible item on the website.
- [Technical Report](reports/technical_report.md): In-depth technical analysis, preprocessing audit, ablation studies, and evaluation methodology.
- [Colab Master Notebook](google_colab/EV_Range_Prediction_Colab_Master.ipynb): Self-contained runnable Google Colab notebook.
