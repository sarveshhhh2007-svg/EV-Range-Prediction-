/**
 * Interactive Charts Module.
 * Integrates Chart.js with the Burgundy & Peach design identity,
 * enforces accessible tabular alternatives, and handles dynamic scatter switching.
 */

import { API } from './api.js';

// Global palette definition for Chart.js
const PALETTE = {
  burgundy900: '#59171B',
  burgundy700: '#8B4947',
  burgundy500: '#A86B63',
  peach500: '#E8B38D',
  peach300: '#FED7B8',
  peach100: '#FFF1E6',
  peach50: '#FFF9F4',
  ink: '#3A0E11',
  inkMuted: '#6B4546',
  line: 'rgba(89, 23, 27, 0.14)'
};

let activeScatterChart = null;
let activeDistributionChart = null;
let activeComparisonChart = null;

export async function initDistributionChart() {
  const ctx = document.getElementById('chart-distribution');
  if (!ctx || !window.Chart) return;

  try {
    const data = await API.getEDADistribution();
    const bins = data.histogram;
    const labels = bins.map(b => b.label);
    const counts = bins.map(b => b.count);

    if (activeDistributionChart) activeDistributionChart.destroy();

    activeDistributionChart = new window.Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Vehicle Count',
          data: counts,
          backgroundColor: PALETTE.burgundy900,
          borderColor: PALETTE.peach500,
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: PALETTE.ink,
            titleColor: PALETTE.peach50,
            bodyColor: PALETTE.peach50,
            padding: 10,
            cornerRadius: 8
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: PALETTE.inkMuted, font: { size: 11 } }
          },
          y: {
            grid: { color: PALETTE.line },
            ticks: { color: PALETTE.inkMuted, font: { size: 11 } }
          }
        }
      }
    });

    // Populate accessible table alternative
    populateAccessibleTable('chart-distribution-table', [
      { header: 'Range Band', key: 'label' },
      { header: 'Vehicle Frequency', key: 'count' }
    ], bins);
  } catch (err) {
    console.error('Failed to init distribution chart:', err);
  }
}

export async function updateScatterChart(featureName) {
  const ctx = document.getElementById('chart-scatter');
  if (!ctx || !window.Chart) return;

  try {
    const data = await API.getEDAScatter(featureName);
    const scatterPoints = data.points.map(pt => ({
      x: pt.x,
      y: pt.y,
      brand: pt.brand,
      model: pt.model
    }));

    if (activeScatterChart) activeScatterChart.destroy();

    activeScatterChart = new window.Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: `${data.feature_label} vs Range`,
          data: scatterPoints,
          backgroundColor: PALETTE.burgundy700,
          borderColor: PALETTE.peach300,
          borderWidth: 1,
          pointRadius: 5,
          pointHoverRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: PALETTE.ink,
            titleColor: PALETTE.peach50,
            bodyColor: PALETTE.peach50,
            callbacks: {
              title: (items) => {
                const pt = items[0].raw;
                return `${pt.brand} ${pt.model}`;
              },
              label: (item) => {
                const pt = item.raw;
                return `${data.feature_label}: ${pt.x} ${data.unit} | Range: ${pt.y} km`;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: `${data.feature_label} (${data.unit})`,
              color: PALETTE.ink,
              font: { weight: 'bold' }
            },
            grid: { color: PALETTE.line },
            ticks: { color: PALETTE.inkMuted }
          },
          y: {
            title: {
              display: true,
              text: 'Listed Driving Range (km)',
              color: PALETTE.ink,
              font: { weight: 'bold' }
            },
            grid: { color: PALETTE.line },
            ticks: { color: PALETTE.inkMuted }
          }
        }
      }
    });

    // Populate accessible table alternative
    populateAccessibleTable('chart-scatter-table', [
      { header: 'Vehicle', formatter: r => `${r.brand} ${r.model}` },
      { header: `${data.feature_label} (${data.unit})`, key: 'x' },
      { header: 'Range (km)', key: 'y' }
    ], scatterPoints.slice(0, 15)); // Sample top 15 for table
  } catch (err) {
    console.error('Failed to update scatter chart:', err);
  }
}

export async function initComparisonChart() {
  const ctx = document.getElementById('chart-comparison');
  if (!ctx || !window.Chart) return;

  try {
    const data = await API.getModelMetrics();
    const models = Object.keys(data.models);
    const maes = models.map(m => data.models[m].test_metrics.mae);
    const rmses = models.map(m => data.models[m].test_metrics.rmse);

    if (activeComparisonChart) activeComparisonChart.destroy();

    activeComparisonChart = new window.Chart(ctx, {
      type: 'bar',
      data: {
        labels: models,
        datasets: [
          {
            label: 'Test MAE (km)',
            data: maes,
            backgroundColor: PALETTE.burgundy900,
            borderRadius: 6
          },
          {
            label: 'Test RMSE (km)',
            data: rmses,
            backgroundColor: PALETTE.peach500,
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: PALETTE.ink, font: { weight: 'bold' } }
          },
          tooltip: {
            backgroundColor: PALETTE.ink,
            titleColor: PALETTE.peach50,
            bodyColor: PALETTE.peach50
          }
        },
        scales: {
          x: { grid: { display: false }, ticks: { color: PALETTE.ink } },
          y: {
            title: { display: true, text: 'Error in km (Lower is better)', color: PALETTE.inkMuted },
            grid: { color: PALETTE.line },
            ticks: { color: PALETTE.inkMuted }
          }
        }
      }
    });
  } catch (err) {
    console.error('Failed to init comparison chart:', err);
  }
}

function populateAccessibleTable(containerId, columns, dataRows) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let html = '<table class="data-table"><thead><tr>';
  columns.forEach(col => {
    html += `<th>${col.header}</th>`;
  });
  html += '</tr></thead><tbody>';

  dataRows.forEach(row => {
    html += '<tr>';
    columns.forEach(col => {
      const val = col.formatter ? col.formatter(row) : row[col.key];
      html += `<td>${val !== undefined && val !== null ? val : '—'}</td>`;
    });
    html += '</tr>';
  });
  html += '</tbody></table>';

  container.innerHTML = html;
}

