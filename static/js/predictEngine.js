/**
 * predictEngine.js - Client-Side ML Inference Engine
 * Runs Linear Regression, Gradient Boosting, and Random Forest models
 * entirely in the browser using pre-extracted pipeline weights.
 * Enables 100% offline and GitHub Pages static hosting.
 */

let modelsBundle = null;

export async function loadModelsBundle(basePath = "./static/data") {
  if (modelsBundle) return modelsBundle;
  try {
    const res = await fetch(`${basePath}/models_bundle.json`);
    if (!res.ok) throw new Error("Could not load models bundle");
    modelsBundle = await res.json();
    return modelsBundle;
  } catch (err) {
    console.error("Failed to load models bundle:", err);
    return null;
  }
}

export function predictClient(data, bundle = modelsBundle) {
  if (!bundle) throw new Error("Models bundle not loaded");

  const ng = bundle.num_general;
  const cells = bundle.cells;
  const cat = bundle.cat;

  const X_lr = [];
  const X_tree = [];

  // 1. General numeric features (12 features)
  for (let i = 0; i < ng.cols.length; i++) {
    const col = ng.cols[i];
    const raw = data[col];
    const val = (raw !== undefined && raw !== null && String(raw).trim() !== "")
      ? parseFloat(raw)
      : ng.imputer[i];
    const scaled = (val - ng.mean[i]) / ng.scale[i];
    X_lr.push(scaled);
    X_tree.push(val);
  }

  // 2. Cells (2 features: value + missing indicator)
  const rawCells = data["number_of_cells"];
  const hasCells = (rawCells !== undefined && rawCells !== null && String(rawCells).trim() !== "");
  const cellsVal = hasCells ? parseFloat(rawCells) : cells.imputer[0];
  const cellsScaled = (cellsVal - cells.mean[0]) / cells.scale[0];
  const cellsIndicatorScaled = ((hasCells ? 0 : 1) - cells.mean[1]) / cells.scale[1];
  X_lr.push(cellsScaled);
  X_lr.push(cellsIndicatorScaled);

  X_tree.push(cellsVal);
  X_tree.push(hasCells ? 0 : 1);

  // 3. Categorical one-hot features (85 features)
  for (let i = 0; i < cat.cols.length; i++) {
    const col = cat.cols[i];
    const raw = data[col];
    const val = (raw !== undefined && raw !== null && String(raw).trim() !== "")
      ? String(raw).trim()
      : cat.imputer[i];
    const options = cat.categories[col] || [];
    for (let j = 0; j < options.length; j++) {
      const isMatch = (val === options[j]) ? 1.0 : 0.0;
      X_lr.push(isMatch);
      X_tree.push(isMatch);
    }
  }

  // Linear Regression (uses X_lr)
  let y_lr = bundle.lr.intercept;
  for (let i = 0; i < bundle.lr.coef.length; i++) {
    y_lr += bundle.lr.coef[i] * X_lr[i];
  }

  // Gradient Boosting (uses X_tree)
  let y_gb = bundle.gb.init_value;
  const gbTrees = bundle.gb.trees;
  for (let t = 0; t < gbTrees.length; t++) {
    const tree = gbTrees[t];
    let node = 0;
    while (tree.left[node] !== -1) {
      const feat = tree.feat[node];
      const thresh = tree.thresh[node];
      node = (X_tree[feat] <= thresh) ? tree.left[node] : tree.right[node];
    }
    y_gb += bundle.gb.learning_rate * tree.val[node];
  }

  // Random Forest (uses X_tree)
  let rfSum = 0;
  const rfTrees = bundle.rf.trees;
  for (let t = 0; t < rfTrees.length; t++) {
    const tree = rfTrees[t];
    let node = 0;
    while (tree.left[node] !== -1) {
      const feat = tree.feat[node];
      const thresh = tree.thresh[node];
      node = (X_tree[feat] <= thresh) ? tree.left[node] : tree.right[node];
    }
    rfSum += tree.val[node];
  }
  const y_rf = rfSum / rfTrees.length;

  const predictions = {
    "Linear Regression": Math.max(10, Math.round(y_lr)),
    "Random Forest": Math.max(10, Math.round(y_rf)),
    "Gradient Boosting": Math.max(10, Math.round(y_gb))
  };

  const bestModel = "Linear Regression";
  const bestPrediction = predictions[bestModel];

  const maxVal = Math.max(...Object.values(predictions), 1);
  const bars = {
    "Linear Regression": Math.max(10, Math.min(100, Math.round((predictions["Linear Regression"] / maxVal) * 100))),
    "Random Forest": Math.max(10, Math.min(100, Math.round((predictions["Random Forest"] / maxVal) * 100))),
    "Gradient Boosting": Math.max(10, Math.min(100, Math.round((predictions["Gradient Boosting"] / maxVal) * 100)))
  };

  const batNum = parseFloat(data["battery_capacity_kWh"]);
  let unusualNote = null;
  if (batNum < 10 || batNum > 225) {
    unusualNote = "This value is unusual compared with the cars used for training.";
  }

  return {
    predictions,
    best_model: bestModel,
    best_prediction: bestPrediction,
    bars,
    unusual_note: unusualNote,
    disclaimer: "These are estimates of listed range, not guaranteed real-world range."
  };
}
