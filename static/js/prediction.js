/**
 * Interactive Prediction Studio Module.
 * Dynamically builds form from schema, validates client-side,
 * loads genuine test-set example vehicles, submits inference payloads,
 * and renders multi-model result comparisons.
 */

import { API } from './api.js';
import { showToast, animateCountUp } from './motion.js';

let formSchema = null;
let currentExample = null;
const predictionHistory = [];

export async function initPredictionStudio() {
  const formContainer = document.getElementById('prediction-form-container');
  if (!formContainer) return;

  try {
    formSchema = await API.getPredictSchema();
    renderDynamicForm(formContainer, formSchema);
    setupPredictionListeners();
  } catch (err) {
    formContainer.innerHTML = `
      <div class="card" style="border-left: 4px solid var(--burgundy-900);">
        <h4>Model Training Required</h4>
        <p style="color: var(--ink-muted);">Predictions require trained model artifacts. Please visit the Model Laboratory to train or reload models.</p>
        <a href="#laboratory" class="btn btn-primary" style="margin-top: 10px;">Go to Model Laboratory</a>
      </div>
    `;
  }
}

function renderDynamicForm(container, schema) {
  let html = '<form id="ev-prediction-form" novalidate>';

  schema.groups.forEach(group => {
    html += `
      <div class="form-group-card">
        <h3 class="form-group-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
          </svg>
          ${group.title}
        </h3>
        <p class="form-group-desc">${group.description}</p>
        <div class="fields-grid">
    `;

    group.fields.forEach(field => {
      const isRequired = field.required ? '<span style="color: var(--burgundy-900); font-weight: bold;">*</span>' : '';
      const placeholder = field.required ? 'Required' : `Optional (Median: ${field.median || 'auto'})`;

      if (field.type === 'select') {
        html += `
          <div class="form-field">
            <label class="field-label" for="field-${field.name}">
              ${field.label} ${isRequired}
            </label>
            <select class="field-select" id="field-${field.name}" name="${field.name}">
              <option value="">— Select ${field.label} —</option>
              ${field.options.map(opt => `<option value="${opt}" ${opt === field.default ? 'selected' : ''}>${opt}</option>`).join('')}
            </select>
            <span class="field-help">${field.help || ''}</span>
            <span class="field-error" id="err-${field.name}"></span>
          </div>
        `;
      } else {
        html += `
          <div class="form-field">
            <label class="field-label" for="field-${field.name}">
              <span>${field.label} ${isRequired}</span>
              <span class="field-unit">${field.unit || ''}</span>
            </label>
            <input 
              type="number" 
              class="field-input" 
              id="field-${field.name}" 
              name="${field.name}" 
              placeholder="${placeholder}"
              step="any"
              min="${field.min || 0}"
              ${field.required ? 'required' : ''}
              aria-describedby="help-${field.name} err-${field.name}"
            >
            <span class="field-help" id="help-${field.name}">${field.help || ''}</span>
            <span class="field-error" id="err-${field.name}"></span>
          </div>
        `;
      }
    });

    html += `
        </div>
      </div>
    `;
  });

  html += `
    <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-top: 24px;">
      <button type="submit" class="btn btn-primary" id="btn-submit-predict" style="padding: 14px 28px; font-size: 1.05rem;">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
        </svg>
        Predict EV Driving Range
      </button>
      <button type="button" class="btn btn-secondary" id="btn-load-example">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
        Load Test Set Example EV
      </button>
      <button type="button" class="btn btn-secondary" id="btn-reset-form">
        Reset
      </button>
    </div>
  </form>
  `;

  container.innerHTML = html;
}

function setupPredictionListeners() {
  const form = document.getElementById('ev-prediction-form');
  const btnExample = document.getElementById('btn-load-example');
  const btnReset = document.getElementById('btn-reset-form');

  if (btnExample) {
    btnExample.addEventListener('click', handleLoadExample);
  }

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      form.reset();
      currentExample = null;
      clearFormErrors();
      const exampleBadge = document.getElementById('example-actual-banner');
      if (exampleBadge) exampleBadge.remove();
      showToast('Prediction form cleared', 'info');
    });
  }

  if (form) {
    form.addEventListener('submit', handleFormSubmit);
  }
}

async function handleLoadExample() {
  try {
    const example = await API.getPredictExample();
    currentExample = example;
    const form = document.getElementById('ev-prediction-form');
    clearFormErrors();

    // Populate inputs
    Object.entries(example.inputs).forEach(([key, val]) => {
      const input = form.querySelector(`[name="${key}"]`);
      if (input && val !== null && val !== undefined) {
        input.value = val;
      }
    });

    // Show actual range benchmark banner
    let banner = document.getElementById('example-actual-banner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'example-actual-banner';
      banner.className = 'scope-banner';
      form.prepend(banner);
    }
    banner.innerHTML = `
      <div class="scope-banner-content">
        <h4>Loaded Test Vehicle: ${example.brand} ${example.model}</h4>
        <p><strong>Actual Official Listed Range:</strong> <span style="font-size: 1.15rem; color: var(--burgundy-900); font-weight: 800;">${example.actual_range_km} km</span></p>
        <p style="font-size: 0.8rem; color: var(--ink-muted); margin-top: 4px;">Click "Predict EV Driving Range" below to compare all three models against this real-world test benchmark.</p>
      </div>
    `;

    showToast(`Loaded test example: ${example.brand} ${example.model}`, 'info');
  } catch (err) {
    showToast('Failed to load example: ' + err.message, 'error');
  }
}

function clearFormErrors() {
  document.querySelectorAll('.field-error').forEach(el => {
    el.textContent = '';
    el.classList.remove('show');
  });
  document.querySelectorAll('[aria-invalid="true"]').forEach(el => {
    el.removeAttribute('aria-invalid');
  });
}

async function handleFormSubmit(e) {
  e.preventDefault();
  clearFormErrors();

  const form = e.target;
  const formData = new FormData(form);
  const payload = {};
  let hasErrors = false;
  let firstInvalidEl = null;

  for (const [key, val] of formData.entries()) {
    const trimmed = typeof val === 'string' ? val.trim() : val;
    if (trimmed !== '') {
      payload[key] = trimmed;
    }
  }

  // Client-side validation: Battery capacity is required
  if (!payload.battery_capacity_kWh) {
    const el = form.querySelector('[name="battery_capacity_kWh"]');
    const err = document.getElementById('err-battery_capacity_kWh');
    if (el && err) {
      el.setAttribute('aria-invalid', 'true');
      err.textContent = 'Battery Capacity is required.';
      err.classList.add('show');
      if (!firstInvalidEl) firstInvalidEl = el;
      hasErrors = true;
    }
  }

  // Validate non-negative numbers
  ['battery_capacity_kWh', 'top_speed_kmh', 'torque_nm', 'efficiency_wh_per_km', 'acceleration_0_100_s', 'seats'].forEach(field => {
    if (payload[field] !== undefined) {
      const num = parseFloat(payload[field]);
      if (isNaN(num) || num < 0) {
        const el = form.querySelector(`[name="${field}"]`);
        const err = document.getElementById(`err-${field}`);
        if (el && err) {
          el.setAttribute('aria-invalid', 'true');
          err.textContent = 'Value must be a positive number.';
          err.classList.add('show');
          if (!firstInvalidEl) firstInvalidEl = el;
          hasErrors = true;
        }
      }
    }
  });

  if (hasErrors) {
    if (firstInvalidEl) firstInvalidEl.focus();
    showToast('Please correct form validation errors before predicting.', 'error');
    return;
  }

  // Submit to backend
  const btn = document.getElementById('btn-submit-predict');
  const originalBtnText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = 'Computing predictions...';

  try {
    const result = await API.predictRange(payload);
    renderPredictionResults(result);
    // Record in session history
    predictionHistory.unshift({
      timestamp: new Date().toLocaleTimeString(),
      inputs: payload,
      result: result
    });
    if (predictionHistory.length > 5) predictionHistory.pop();
    showToast('Prediction successfully generated!', 'info');
  } catch (err) {
    showToast('Prediction error: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = originalBtnText;
  }
}

function renderPredictionResults(res) {
  const container = document.getElementById('prediction-results-container');
  if (!container) return;

  const recModel = res.recommended_model;
  const recPred = res.recommended_prediction_km;
  const models = Object.keys(res.predictions);

  let softWarningsHtml = '';
  if (res.soft_warnings && res.soft_warnings.length > 0) {
    softWarningsHtml = `
      <div style="background: var(--peach-200); border: 1px solid var(--peach-500); border-radius: var(--radius-sm); padding: 12px; margin-bottom: 16px; font-size: 0.82rem;">
        <strong style="color: var(--burgundy-900);">Distribution Warning:</strong>
        <ul style="margin: 4px 0 0 16px; color: var(--ink);">
          ${res.soft_warnings.map(w => `<li>${w}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  let actualBenchmarkHtml = '';
  if (currentExample && currentExample.actual_range_km) {
    const errorFromActual = Math.abs(recPred - currentExample.actual_range_km).toFixed(1);
    actualBenchmarkHtml = `
      <div style="background: var(--peach-50); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 10px 14px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
          <span style="font-size: 0.78rem; text-transform: uppercase; font-weight: 700; color: var(--ink-muted);">Actual Listed Benchmark</span>
          <div style="font-family: 'Manrope', sans-serif; font-size: 1.25rem; font-weight: 800; color: var(--burgundy-900);">${currentExample.actual_range_km} km</div>
        </div>
        <div style="text-align: right;">
          <span style="font-size: 0.78rem; color: var(--ink-muted);">Absolute Error</span>
          <div style="font-weight: 700; color: var(--burgundy-700);">${errorFromActual} km</div>
        </div>
      </div>
    `;
  }

  let html = `
    <div class="result-hero-box">
      <span class="result-hero-badge">Recommended Model: ${recModel}</span>
      <div style="display: flex; align-items: baseline; justify-content: center;">
        <span class="result-hero-number" id="result-hero-val">${recPred}</span>
        <span class="result-hero-unit">km</span>
      </div>
      <p style="font-size: 0.85rem; color: var(--peach-200); margin: 6px 0 0;">Estimated Listed Driving Range</p>
    </div>

    ${actualBenchmarkHtml}
    ${softWarningsHtml}

    <div class="result-models-comparison">
      <h4 style="font-size: 0.92rem; color: var(--burgundy-950); margin-bottom: 8px;">Three Regression Models Comparison:</h4>
  `;

  models.forEach(m => {
    const val = res.predictions[m];
    const isRec = m === recModel;
    html += `
      <div class="model-result-row ${isRec ? 'recommended' : ''}">
        <div class="model-row-name">
          <span>${m}</span>
          ${isRec ? '<span style="font-size: 0.7rem; font-weight: 700; background: var(--burgundy-900); color: var(--peach-50); padding: 2px 6px; border-radius: 4px;">BEST CV</span>' : ''}
        </div>
        <div class="model-row-val">${val} km</div>
      </div>
    `;
  });

  html += `
    </div>

    <div style="background: var(--peach-50); border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 14px; margin-bottom: 18px; font-size: 0.82rem; color: var(--ink-muted);">
      <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
        <span>Spread between models (Max - Min):</span>
        <strong style="color: var(--burgundy-950);">${res.spread_between_models_km} km</strong>
      </div>
      <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
        <span>Random Forest tree-spread:</span>
        <strong style="color: var(--burgundy-950);">[${res.random_forest_tree_spread.min_tree_km} - ${res.random_forest_tree_spread.max_tree_km}] km</strong>
      </div>
      <div style="font-size: 0.75rem; color: var(--ink-muted); margin-top: 6px; border-top: 1px dashed var(--line); padding-top: 6px;">
        ${res.mae_description}
      </div>
    </div>

    <div style="display: flex; gap: 10px;">
      <button class="btn btn-secondary" id="btn-copy-prediction" style="flex: 1; padding: 10px; font-size: 0.88rem;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        Copy Summary
      </button>
    </div>

    <p style="font-size: 0.75rem; color: var(--ink-muted); margin-top: 14px; font-style: italic;">
      ${res.scope_disclaimer}
    </p>
  `;

  container.innerHTML = html;

  // Add copy listener
  const copyBtn = document.getElementById('btn-copy-prediction');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const summaryText = `EV Driving Range Prediction:\n` +
        `Recommended Model: ${recModel} (${recPred} km)\n` +
        models.map(m => `- ${m}: ${res.predictions[m]} km`).join('\n') +
        `\nSpread: ${res.spread_between_models_km} km\n` +
        `Historical Test-Set MAE: ±${res.selected_model_test_mae} km`;

      navigator.clipboard.writeText(summaryText).then(() => {
        showToast('Prediction summary copied to clipboard!', 'info');
      });
    });
  }

  // Animate hero count up
  const heroValEl = document.getElementById('result-hero-val');
  if (heroValEl) {
    animateCountUp(heroValEl, recPred, 600);
  }
}

