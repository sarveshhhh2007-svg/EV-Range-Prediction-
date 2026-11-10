/**
 * Main Application Orchestrator.
 * Initializes all dashboard sections, coordinates state,
 * handles dataset explorer pagination/search, and manages background training.
 */

import { API, clearApiCache } from './api.js';
import { initScrollSpy, initMetricCountUps, initBackToTop, handleDeepLinking, showToast } from './motion.js';
import { initDistributionChart, updateScatterChart, initComparisonChart } from './charts.js';
import { initPredictionStudio } from './prediction.js';

// Dataset Explorer State
let currentTablePage = 1;
let currentTablePageSize = 10;
let currentTableView = 'processed';
let currentTableSearch = '';
let currentTableSort = '';
let searchDebounceTimer = null;

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
  initNavigation();
  initBackToTop();
  handleDeepLinking();

  try {
    await loadHealthAndSummary();
    await initDatasetExplorer();
    await loadPreprocessingData();
    await loadEDASection();
    await loadModelLaboratory();
    await loadModelComparison();
    await initPredictionStudio();

    initScrollSpy();
    initMetricCountUps();
  } catch (err) {
    console.error('Initialization error:', err);
    showToast('Dashboard initialized with partial data: ' + err.message, 'warning');
  }
});

// -------------------------------------------------------------
// Navigation & Layout
// -------------------------------------------------------------
function initNavigation() {
  const menuBtn = document.querySelector('.menu-toggle-btn');
  const sidebar = document.querySelector('.sidebar');
  if (menuBtn && sidebar) {
    menuBtn.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    document.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 860) {
          sidebar.classList.remove('open');
        }
      });
    });
  }
}

// -------------------------------------------------------------
// Section 1: Overview & Hero
// -------------------------------------------------------------
async function loadHealthAndSummary() {
  const summary = await API.getSummary();
  const health = await API.getHealth();

  // Populate hero triptych metrics
  const triptychRecords = document.getElementById('hero-metric-records');
  const triptychFeatures = document.getElementById('hero-metric-features');
  const triptychModels = document.getElementById('hero-metric-models');

  if (triptychRecords) triptychRecords.textContent = summary.total_records;
  if (triptychFeatures) triptychFeatures.textContent = summary.numerical_features_count + summary.categorical_features_count;
  if (triptychModels) triptychModels.textContent = 3;

  // Status badge
  const statusBadge = document.getElementById('hero-status-badge');
  if (statusBadge) {
    if (summary.models_trained) {
      statusBadge.textContent = `Trained Models Active (Best: ${summary.recommended_model})`;
    } else {
      statusBadge.textContent = 'Models Not Trained — Action Required';
    }
  }

  // Warning banner if dataset changed
  if (health.metadata_warning) {
    showToast(health.metadata_warning, 'warning', 6000);
  }
}

// -------------------------------------------------------------
// Section 2: Dataset Explorer
// -------------------------------------------------------------
async function initDatasetExplorer() {
  const searchInput = document.getElementById('dataset-search');
  const pageSizeSelect = document.getElementById('dataset-page-size');
  const viewToggleBtn = document.getElementById('dataset-view-toggle');
  const prevBtn = document.getElementById('dataset-prev-page');
  const nextBtn = document.getElementById('dataset-next-page');

  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      clearTimeout(searchDebounceTimer);
      searchDebounceTimer = setTimeout(() => {
        currentTableSearch = e.target.value;
        currentTablePage = 1;
        fetchAndRenderTable();
      }, 300);
    });
  }

  if (pageSizeSelect) {
    pageSizeSelect.addEventListener('change', (e) => {
      currentTablePageSize = parseInt(e.target.value, 10);
      currentTablePage = 1;
      fetchAndRenderTable();
    });
  }

  if (viewToggleBtn) {
    viewToggleBtn.addEventListener('click', () => {
      currentTableView = currentTableView === 'processed' ? 'original' : 'processed';
      viewToggleBtn.textContent = `View: ${currentTableView.toUpperCase()}`;
      const badge = document.getElementById('table-view-badge');
      if (badge) {
        badge.textContent = currentTableView === 'processed' ? 'Preprocessed View (Numbers Coerced)' : 'Raw Unprocessed View';
      }
      currentTablePage = 1;
      fetchAndRenderTable();
    });
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      if (currentTablePage > 1) {
        currentTablePage--;
        fetchAndRenderTable();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      currentTablePage++;
      fetchAndRenderTable();
    });
  }

  await fetchAndRenderTable();
}

async function fetchAndRenderTable() {
  const container = document.getElementById('dataset-table-container');
  if (!container) return;

  try {
    const res = await API.getDatasetPreview({
      view: currentTableView,
      page: currentTablePage,
      page_size: currentTablePageSize,
      search: currentTableSearch,
      sort: currentTableSort
    });

    renderDatasetTable(container, res);

    // Update pagination controls
    const pageIndicator = document.getElementById('dataset-page-indicator');
    const prevBtn = document.getElementById('dataset-prev-page');
    const nextBtn = document.getElementById('dataset-next-page');

    if (pageIndicator) {
      pageIndicator.textContent = `Page ${res.page} of ${res.total_pages} (${res.total_records} records)`;
    }
    if (prevBtn) prevBtn.disabled = res.page <= 1;
    if (nextBtn) nextBtn.disabled = res.page >= res.total_pages;
  } catch (err) {
    container.innerHTML = `<div class="card" style="color: var(--burgundy-900);">Error loading dataset preview: ${err.message}</div>`;
  }
}

function renderDatasetTable(container, data) {
  if (!data.data || !data.data.length) {
    container.innerHTML = '<div style="padding: 30px; text-align: center; color: var(--ink-muted);">No records found matching search query.</div>';
    return;
  }

  const columns = Object.keys(data.data[0]);
  let html = '<table class="data-table"><thead><tr>';

  columns.forEach(col => {
    const isSorted = currentTableSort === col ? '▲' : currentTableSort === `-${col}` ? '▼' : '';
    html += `<th class="sortable" data-col="${col}">${col.replace('_', ' ')} ${isSorted}</th>`;
  });
  html += '</tr></thead><tbody>';

  data.data.forEach(row => {
    html += '<tr>';
    columns.forEach(col => {
      const val = row[col];
      html += `<td>${val !== null && val !== undefined ? val : '<span style="color: var(--burgundy-500); font-style: italic;">null</span>'}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  container.innerHTML = html;

  // Add sorting listeners
  container.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.getAttribute('data-col');
      if (currentTableSort === col) {
        currentTableSort = `-${col}`;
      } else if (currentTableSort === `-${col}`) {
        currentTableSort = '';
      } else {
        currentTableSort = col;
      }
      fetchAndRenderTable();
    });
  });
}

// -------------------------------------------------------------
// Section 3: Data Preprocessing
// -------------------------------------------------------------
async function loadPreprocessingData() {
  try {
    const data = await API.getPreprocessing();

    // 1. Cleaning Log
    const logContainer = document.getElementById('cleaning-log-container');
    if (logContainer && data.cleaning_log) {
      let html = '<table class="data-table"><thead><tr>';
      html += '<th>Step</th><th>What Changed</th><th>Affected Records</th><th>Engineering Rationale</th></tr></thead><tbody>';
      data.cleaning_log.forEach(item => {
        html += `
          <tr>
            <td><strong>${item.step}</strong></td>
            <td>${item.what_changed}</td>
            <td style="font-weight: 700; color: var(--burgundy-900);">${item.count_affected}</td>
            <td style="color: var(--ink-muted); font-size: 0.85rem;">${item.reason}</td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      logContainer.innerHTML = html;
    }

    // 2. Leakage Audit Panel
    const leakageContainer = document.getElementById('leakage-audit-container');
    if (leakageContainer && data.leakage_audit) {
      const la = data.leakage_audit;
      leakageContainer.innerHTML = `
        <div class="card" style="border-left: 4px solid var(--burgundy-900); background: var(--peach-100);">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 16px;">
            <div>
              <h4 style="color: var(--burgundy-900); margin-bottom: 6px;">Target-Leakage Audit Findings</h4>
              <p style="font-family: monospace; font-size: 0.9rem; color: var(--burgundy-950); margin-bottom: 8px;">
                ${la.formula}
              </p>
            </div>
            <div style="display: flex; gap: 20px;">
              <div style="text-align: right;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: var(--ink-muted); font-weight: 700;">Pearson Corr</span>
                <div style="font-family: 'Manrope', sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--burgundy-900);">${la.pearson_correlation}</div>
              </div>
              <div style="text-align: right;">
                <span style="font-size: 0.75rem; text-transform: uppercase; color: var(--ink-muted); font-weight: 700;">Within 10%</span>
                <div style="font-family: 'Manrope', sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--burgundy-700);">${la.share_within_10_percent}%</div>
              </div>
            </div>
          </div>
          <p style="color: var(--ink); font-size: 0.95rem; margin-top: 10px;">${la.plain_language_conclusion}</p>
          <p style="font-size: 0.82rem; color: var(--ink-muted); margin-top: 6px; font-style: italic;">${la.leakage_note}</p>
        </div>
      `;
    }

    // 3. Outlier Table
    const outlierContainer = document.getElementById('outliers-table-container');
    if (outlierContainer && data.outliers_detected) {
      const sampleOutliers = data.outliers_detected.slice(0, 8);
      let html = '<table class="data-table"><thead><tr>';
      html += '<th>Brand & Model</th><th>Feature</th><th>Observed Value</th><th>IQR Rule</th><th>Action</th><th>Engineering Rationale</th></tr></thead><tbody>';
      sampleOutliers.forEach(o => {
        html += `
          <tr>
            <td><strong>${o.brand} ${o.model}</strong></td>
            <td>${o.feature_label}</td>
            <td style="font-weight: 700; color: var(--burgundy-900);">${o.value} ${o.unit}</td>
            <td style="font-size: 0.82rem; color: var(--ink-muted);">${o.rule}</td>
            <td><span style="background: var(--peach-300); color: var(--burgundy-950); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">${o.action}</span></td>
            <td style="font-size: 0.82rem; color: var(--ink-muted);">${o.reason}</td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      outlierContainer.innerHTML = html;
    }
  } catch (err) {
    console.error('Failed to load preprocessing data:', err);
  }
}

// -------------------------------------------------------------
// Section 4: Exploratory Data Analysis
// -------------------------------------------------------------
async function loadEDASection() {
  try {
    // 1. Descriptive stats table
    const statsData = await API.getEDAStats();
    const statsContainer = document.getElementById('eda-stats-table-container');
    if (statsContainer && statsData.descriptive_statistics) {
      const stats = statsData.descriptive_statistics;
      let html = '<table class="data-table"><thead><tr>';
      html += '<th>Feature</th><th>Unit</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th><th>Std Dev</th><th>Covariance with Range</th></tr></thead><tbody>';
      Object.keys(stats).forEach(col => {
        const item = stats[col];
        html += `
          <tr>
            <td><strong>${item.name.replace(/_/g, ' ')}</strong></td>
            <td>${item.unit}</td>
            <td>${item.mean}</td>
            <td>${item.median}</td>
            <td>${item.min}</td>
            <td>${item.max}</td>
            <td>${item.std}</td>
            <td style="font-weight: 700; color: var(--burgundy-900);">${item.covariance_with_range}</td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      statsContainer.innerHTML = html;
    }

    // 2. Interactive Charts
    await initDistributionChart();
    await updateScatterChart('battery_capacity_kWh');

    // Scatter dropdown listener
    const scatterSelect = document.getElementById('scatter-feature-select');
    if (scatterSelect) {
      scatterSelect.addEventListener('change', (e) => {
        updateScatterChart(e.target.value);
      });
    }
  } catch (err) {
    console.error('Failed to load EDA section:', err);
  }
}

// -------------------------------------------------------------
// Section 5: Model Laboratory & Background Retraining
// -------------------------------------------------------------
async function loadModelLaboratory() {
  const container = document.getElementById('models-cards-container');
  const retrainBtn = document.getElementById('btn-open-retrain-modal');
  const modal = document.getElementById('retrain-modal');
  const confirmBtn = document.getElementById('btn-confirm-retrain');
  const cancelBtn = document.getElementById('btn-cancel-retrain');

  if (retrainBtn && modal) {
    retrainBtn.addEventListener('click', () => modal.classList.add('show'));
    cancelBtn.addEventListener('click', () => modal.classList.remove('show'));
    confirmBtn.addEventListener('click', executeRetraining);
  }

  try {
    const meta = await API.getModelMetrics();
    renderModelCards(container, meta.models, meta.recommended_model);
  } catch (err) {
    if (container) {
      container.innerHTML = `
        <div class="card" style="grid-column: 1 / -1; border-left: 4px solid var(--burgundy-900);">
          <h4>Trained Pipelines Unavailable</h4>
          <p style="color: var(--ink-muted);">Click "Train Models" to trigger training in the background.</p>
        </div>
      `;
    }
  }
}

function renderModelCards(container, models, recommendedModel) {
  if (!container || !models) return;
  let html = '';

  Object.entries(models).forEach(([name, m]) => {
    const isRec = name === recommendedModel;
    html += `
      <div class="card" style="${isRec ? 'border: 2px solid var(--burgundy-900); background: var(--peach-100);' : ''}">
        <span class="model-card-badge">${isRec ? 'RECOMMENDED BY CV' : 'BASELINE REGRESSOR'}</span>
        <h3 style="color: var(--burgundy-900); margin-bottom: 6px;">${name}</h3>
        <p style="font-size: 0.85rem; color: var(--ink-muted); margin-bottom: 16px;">${m.description}</p>
        
        <div class="metric-pill">
          <span class="metric-pill-label">Test MAE:</span>
          <span class="metric-pill-val">${m.test_metrics.mae} km</span>
        </div>
        <div class="metric-pill">
          <span class="metric-pill-label">Test RMSE:</span>
          <span class="metric-pill-val">${m.test_metrics.rmse} km</span>
        </div>
        <div class="metric-pill">
          <span class="metric-pill-label">Variance Explained (R²):</span>
          <span class="metric-pill-val">${m.test_metrics.r2}</span>
        </div>
        <div class="metric-pill">
          <span class="metric-pill-label">5-Fold CV RMSE:</span>
          <span class="metric-pill-val">${m.cv_metrics.rmse_mean} ± ${m.cv_metrics.rmse_std} km</span>
        </div>

        <div style="margin-top: 14px; font-size: 0.78rem; color: var(--ink-muted); border-top: 1px solid var(--line); padding-top: 10px;">
          <strong>Hyperparameters:</strong><br>
          ${Object.entries(m.hyperparameters).map(([k, v]) => `${k}: <code>${v}</code>`).join(', ')}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

async function executeRetraining() {
  const modal = document.getElementById('retrain-modal');
  const progressBox = document.getElementById('retrain-progress-box');
  const progressText = document.getElementById('retrain-progress-text');

  modal.classList.remove('show');
  if (progressBox) progressBox.style.display = 'block';

  try {
    await API.startTraining();
    showToast('Retraining task initiated in background...', 'info');

    // Poll status
    const pollInterval = setInterval(async () => {
      try {
        const statusRes = await API.getTrainingStatus();
        if (progressText) progressText.textContent = statusRes.state.progress_message;

        if (statusRes.state.status === 'completed') {
          clearInterval(pollInterval);
          clearApiCache();
          if (progressBox) progressBox.style.display = 'none';
          showToast('Retraining complete! Reloading dashboard metrics...', 'info');
          setTimeout(() => window.location.reload(), 1500);
        } else if (statusRes.state.status === 'failed') {
          clearInterval(pollInterval);
          if (progressBox) progressBox.style.display = 'none';
          showToast('Training failed: ' + statusRes.state.error, 'error');
        }
      } catch (e) {
        clearInterval(pollInterval);
      }
    }, 2500);

  } catch (err) {
    if (progressBox) progressBox.style.display = 'none';
    showToast('Training request rejected: ' + err.message, 'error');
  }
}

// -------------------------------------------------------------
// Section 6: Model Comparison & Ablations
// -------------------------------------------------------------
async function loadModelComparison() {
  try {
    const meta = await API.getModelMetrics();
    const impData = await API.getModelImportance();

    // 1. Comparison Bar Chart
    await initComparisonChart();

    // 2. Metrics comparison table
    const tableContainer = document.getElementById('comparison-metrics-table-container');
    if (tableContainer && meta.models) {
      let html = '<table class="data-table"><thead><tr>';
      html += '<th>Model Architecture</th><th>Test MAE (km)</th><th>Test MSE (km²)</th><th>Test RMSE (km)</th><th>Test R² (Variance)</th><th>5-Fold CV RMSE</th><th>Unseen Brand CV RMSE</th></tr></thead><tbody>';

      // Naive Baseline
      if (meta.naive_baseline) {
        const nb = meta.naive_baseline.metrics;
        html += `
          <tr style="background: var(--peach-200); font-style: italic;">
            <td><strong>Naive Baseline (Train Mean)</strong></td>
            <td>${nb.mae}</td><td>${nb.mse}</td><td>${nb.rmse}</td><td>${nb.r2}</td><td>—</td><td>—</td>
          </tr>
        `;
      }

      Object.entries(meta.models).forEach(([name, m]) => {
        const isRec = name === meta.recommended_model;
        html += `
          <tr ${isRec ? 'style="font-weight: 700; background: var(--peach-200);"' : ''}>
            <td><strong>${name} ${isRec ? '★' : ''}</strong></td>
            <td>${m.test_metrics.mae}</td>
            <td>${m.test_metrics.mse}</td>
            <td>${m.test_metrics.rmse}</td>
            <td>${m.test_metrics.r2}</td>
            <td>${m.cv_metrics.rmse_mean} ± ${m.cv_metrics.rmse_std}</td>
            <td>${m.unseen_brand_cv.rmse_mean} ± ${m.unseen_brand_cv.rmse_std}</td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      tableContainer.innerHTML = html;
    }

    // 3. Recommendation Justification Box
    const recBox = document.getElementById('model-recommendation-box');
    if (recBox) {
      recBox.innerHTML = `
        <div class="card" style="border-left: 4px solid var(--burgundy-900); background: var(--peach-100);">
          <span class="badge badge-burgundy" style="margin-bottom: 8px;">Final Model Recommendation</span>
          <h3 style="color: var(--burgundy-900);">${meta.recommended_model}</h3>
          <p style="color: var(--ink); margin-bottom: 0;">${meta.recommendation_justification}</p>
        </div>
      `;
    }

    // 4. Ablations Table
    const ablContainer = document.getElementById('ablations-table-container');
    if (ablContainer && meta.ablations) {
      const cellsAbl = meta.ablations.cells_ablation.results;
      const effAbl = meta.ablations.efficiency_ablation.results;

      let html = '<table class="data-table"><thead><tr>';
      html += '<th>Model</th><th>With Cells (CV RMSE)</th><th>Without Cells (CV RMSE)</th><th>With Efficiency (Test R²)</th><th>Without Efficiency (Test R²)</th></tr></thead><tbody>';

      Object.keys(cellsAbl).forEach(m => {
        html += `
          <tr>
            <td><strong>${m}</strong></td>
            <td>${cellsAbl[m].with_cells_cv_rmse} km</td>
            <td>${cellsAbl[m].without_cells_cv_rmse} km</td>
            <td>${effAbl[m].with_efficiency_test_r2}</td>
            <td>${effAbl[m].without_efficiency_test_r2}</td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      ablContainer.innerHTML = html;
    }

    // 5. Permutation Feature Importance Table
    const impContainer = document.getElementById('feature-importance-container');
    if (impContainer && impData) {
      const rfImp = impData['Random Forest']?.primary_permutation_test_set?.slice(0, 8) || [];
      let html = '<table class="data-table"><thead><tr>';
      html += '<th>Rank</th><th>Feature</th><th>Mean Test Drop (RMSE)</th><th>Std Dev</th></tr></thead><tbody>';
      rfImp.forEach((f, idx) => {
        html += `
          <tr>
            <td><strong>#${idx + 1}</strong></td>
            <td>${f.feature_label}</td>
            <td style="font-weight: 700; color: var(--burgundy-900);">${f.importance_mean} km</td>
            <td style="color: var(--ink-muted);">${f.importance_std}</td>
          </tr>
        `;
      });
      html += '</tbody></table>';
      impContainer.innerHTML = html;
    }
  } catch (err) {
    console.error('Failed to load model comparison:', err);
  }
}

