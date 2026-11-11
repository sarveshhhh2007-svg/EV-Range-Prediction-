/**
 * Charts Controller.
 * Handles the 4 Insights charts and 2 Models comparison charts using Chart.js.
 * Strictly uses the Burgundy & Peach palette.
 */

const PALETTE = {
  burgundy: '#59171B',
  rose: '#8B4947',
  peach: '#E8B38D',
  ink: '#3A0E11',
  muted: '#6B4546',
  line: 'rgba(89, 23, 27, 0.14)'
};

export async function initInsightsCharts() {
  if (!window.Chart) return;

  try {
    const res = await fetch('/api/insights');
    if (!res.ok) return;
    const data = await res.json();

    // 1. Driving Range Distribution (Histogram)
    const ctxDist = document.getElementById('chart-distribution');
    if (ctxDist && data.distribution) {
      new window.Chart(ctxDist, {
        type: 'bar',
        data: {
          labels: data.distribution.bins.map(b => b.label),
          datasets: [{
            data: data.distribution.bins.map(b => b.count),
            backgroundColor: PALETTE.burgundy,
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, ticks: { color: PALETTE.muted, font: { size: 11 } } },
            y: { grid: { color: PALETTE.line }, ticks: { color: PALETTE.muted, font: { size: 11 } } }
          }
        }
      });
      const capDist = document.getElementById('caption-distribution');
      if (capDist) capDist.textContent = data.distribution.caption;
    }

    // 2. Battery Capacity vs Range (Scatter)
    const ctxBat = document.getElementById('chart-battery');
    if (ctxBat && data.battery_scatter) {
      new window.Chart(ctxBat, {
        type: 'scatter',
        data: {
          datasets: [{
            data: data.battery_scatter.points,
            backgroundColor: PALETTE.rose,
            pointRadius: 4,
            pointHoverRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => items[0].raw.label || '',
                label: (item) => `Battery: ${item.raw.x} kWh | Range: ${item.raw.y} km`
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: 'Battery capacity (kWh)', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            },
            y: {
              title: { display: true, text: 'Listed range (km)', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            }
          }
        }
      });
      const capBat = document.getElementById('caption-battery');
      if (capBat) capBat.textContent = data.battery_scatter.caption;
    }

    // 3. Acceleration Time vs Range (Scatter)
    const ctxAcc = document.getElementById('chart-acceleration');
    if (ctxAcc && data.acceleration_scatter) {
      new window.Chart(ctxAcc, {
        type: 'scatter',
        data: {
          datasets: [{
            data: data.acceleration_scatter.points,
            backgroundColor: PALETTE.peach,
            borderColor: PALETTE.burgundy,
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => items[0].raw.label || '',
                label: (item) => `0–100 km/h: ${item.raw.x} s | Range: ${item.raw.y} km`
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: '0–100 km/h time (s)', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            },
            y: {
              title: { display: true, text: 'Listed range (km)', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            }
          }
        }
      });
      const capAcc = document.getElementById('caption-acceleration');
      if (capAcc) capAcc.textContent = data.acceleration_scatter.caption;
    }

    // 4. Top 8 Correlations Horizontal Bar
    const ctxCorr = document.getElementById('chart-correlations');
    if (ctxCorr && data.correlations) {
      new window.Chart(ctxCorr, {
        type: 'bar',
        data: {
          labels: data.correlations.items.map(i => i.feature),
          datasets: [{
            data: data.correlations.items.map(i => i.value),
            backgroundColor: data.correlations.items.map(i => i.value >= 0 ? PALETTE.burgundy : PALETTE.rose),
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: {
              title: { display: true, text: 'Correlation with listed range', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            },
            y: {
              grid: { display: false },
              ticks: { color: PALETTE.ink, font: { weight: '600' } }
            }
          }
        }
      });
      const capCorr = document.getElementById('caption-correlations');
      if (capCorr) capCorr.textContent = data.correlations.caption;
    }
  } catch (err) {
    console.error('Error loading insights charts:', err);
  }
}

export async function initModelsCharts() {
  if (!window.Chart) return;

  try {
    const res = await fetch('/api/models/metrics');
    if (!res.ok) return;
    const data = await res.json();

    // 1. MAE Comparison Bar Chart
    const ctxMae = document.getElementById('chart-mae-compare');
    if (ctxMae && data.mae_bars) {
      new window.Chart(ctxMae, {
        type: 'bar',
        data: {
          labels: data.mae_bars.map(m => m.model),
          datasets: [{
            data: data.mae_bars.map(m => m.mae),
            backgroundColor: [PALETTE.burgundy, PALETTE.rose, PALETTE.peach],
            borderColor: PALETTE.burgundy,
            borderWidth: 1,
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false }, ticks: { color: PALETTE.ink, font: { weight: '600' } } },
            y: {
              title: { display: true, text: 'Average error in km (lower is better)', color: PALETTE.muted },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            }
          }
        }
      });
    }

    // 2. Actual vs Predicted Scatter (Best Model)
    const ctxAct = document.getElementById('chart-actual-vs-predicted');
    if (ctxAct && data.actual_vs_predicted && data.actual_vs_predicted.points) {
      const pts = data.actual_vs_predicted.points.map(p => ({ x: p.actual, y: p.predicted, label: p.label }));
      new window.Chart(ctxAct, {
        type: 'scatter',
        data: {
          datasets: [
            {
              label: 'Predictions',
              data: pts,
              backgroundColor: PALETTE.burgundy,
              pointRadius: 4
            },
            {
              label: 'Ideal line',
              data: [{ x: 100, y: 100 }, { x: 700, y: 700 }],
              type: 'line',
              borderColor: PALETTE.muted,
              borderDash: [5, 5],
              fill: false,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => items[0].raw.label || '',
                label: (item) => `Actual: ${item.raw.x} km | Predicted: ${item.raw.y} km`
              }
            }
          },
          scales: {
            x: {
              title: { display: true, text: 'Actual listed range (km)', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            },
            y: {
              title: { display: true, text: 'Predicted range (km)', color: PALETTE.ink },
              grid: { color: PALETTE.line },
              ticks: { color: PALETTE.muted }
            }
          }
        }
      });
    }
  } catch (err) {
    console.error('Error loading models charts:', err);
  }
}
