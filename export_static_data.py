import os
import json
import joblib
import numpy as np
import pandas as pd
import app

client = app.app.test_client()

os.makedirs("static/data", exist_ok=True)

# 1. Summary
res = client.get("/api/summary")
with open("static/data/summary.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported summary.json")

# 2. Preview (All 478 cars for client-side search & pagination)
res = client.get("/api/dataset/preview?page=1&page_size=1000")
with open("static/data/preview.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported preview.json")

# 3. Cleaning
res = client.get("/api/cleaning")
with open("static/data/cleaning.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported cleaning.json")

# 4. Insights
res = client.get("/api/insights")
with open("static/data/insights.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported insights.json")

# 5. Models Metrics
res = client.get("/api/models/metrics")
with open("static/data/models_metrics.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported models_metrics.json")

# 6. Schema
res = client.get("/api/predict/schema")
with open("static/data/schema.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported schema.json")

# 7. Example
res = client.get("/api/predict/example")
with open("static/data/example.json", "w", encoding="utf-8") as f:
    f.write(res.get_data(as_text=True))
print("Exported example.json")

# 8. Models Bundle for client-side JavaScript prediction
lr = joblib.load('models/linear_regression.joblib')
rf = joblib.load('models/random_forest.joblib')
gb = joblib.load('models/gradient_boosting.joblib')

prep = lr.named_steps['preprocessor']
ng = prep.named_transformers_['num_general']
cells = prep.named_transformers_['cells']
cat = prep.named_transformers_['cat']
encoder = cat.named_steps['encoder']
cat_cols = ['brand', 'fast_charge_port', 'drivetrain', 'segment', 'car_body_type']
categories = {col: [str(c) for c in cats] for col, cats in zip(cat_cols, encoder.categories_)}

gb_reg = gb.named_steps['regressor']
gb_trees = []
for stage in gb_reg.estimators_:
    tree = stage[0].tree_
    gb_trees.append({
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "feat": tree.feature.tolist(),
        "thresh": tree.threshold.tolist(),
        "val": tree.value.squeeze().tolist()
    })

rf_reg = rf.named_steps['regressor']
rf_trees = []
for est in rf_reg.estimators_[:60]:
    tree = est.tree_
    rf_trees.append({
        "left": tree.children_left.tolist(),
        "right": tree.children_right.tolist(),
        "feat": tree.feature.tolist(),
        "thresh": tree.threshold.tolist(),
        "val": tree.value.squeeze().tolist()
    })

bundle = {
    "num_general": {
        "cols": ['top_speed_kmh', 'battery_capacity_kWh', 'torque_nm', 'efficiency_wh_per_km', 'acceleration_0_100_s', 'fast_charging_power_kw_dc', 'towing_capacity_kg', 'cargo_volume_l', 'seats', 'length_mm', 'width_mm', 'height_mm'],
        "imputer": ng.named_steps['imputer'].statistics_.tolist(),
        "mean": ng.named_steps['scaler'].mean_.tolist(),
        "scale": ng.named_steps['scaler'].scale_.tolist()
    },
    "cells": {
        "imputer": cells.named_steps['imputer'].statistics_.tolist(),
        "mean": cells.named_steps['scaler'].mean_.tolist(),
        "scale": cells.named_steps['scaler'].scale_.tolist()
    },
    "cat": {
        "cols": cat_cols,
        "imputer": cat.named_steps['imputer'].statistics_.tolist(),
        "categories": categories
    },
    "lr": {
        "intercept": float(lr.named_steps['regressor'].intercept_),
        "coef": lr.named_steps['regressor'].coef_.tolist()
    },
    "gb": {
        "learning_rate": float(gb_reg.learning_rate),
        "init_value": float(gb_reg.init_.constant_[0][0]),
        "trees": gb_trees
    },
    "rf": {
        "trees": rf_trees
    }
}

with open("static/data/models_bundle.json", "w", encoding="utf-8") as f:
    json.dump(bundle, f)

print(f"Exported models_bundle.json ({os.path.getsize('static/data/models_bundle.json') / 1024:.1f} KB)")
print("ALL STATIC EXPORTS COMPLETE!")
