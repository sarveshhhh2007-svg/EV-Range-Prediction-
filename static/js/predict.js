/**
 * predict.js - Simplified EV Range Prediction Studio
 * Handles dynamic field rendering into polished cards, client-side validation,
 * prefilling example test cars, and running 3-model predictions.
 */

import { fetchJson, executePrediction } from "./apiClient.js?v=3.0";

let schemaLoaded = false;

const FIELD_ICONS = {
  battery_capacity_kWh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="16" height="10" rx="2" ry="2"/><line x1="22" y1="11" x2="22" y2="13"/><line x1="6" y1="11" x2="6" y2="13"/><line x1="10" y1="11" x2="10" y2="13"/></svg>`,
  top_speed_kmh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 0 0-10 10c0 4.4 2.9 8.2 7 9.5"/><path d="M15 21.5A10 10 0 0 0 22 12"/><path d="m14 14-3-3"/></svg>`,
  acceleration_0_100_s: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="13" r="8"/><path d="M12 9v4l2 2"/><path d="M10 2h4"/></svg>`,
  fast_charging_power_kw_dc: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  torque_nm: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
  efficiency_wh_per_km: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>`,
  seats: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
  cargo_volume_l: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>`,
  towing_capacity_kg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="5" width="22" height="14" rx="3"/><path d="M12 9v6"/><path d="M9 12h6"/></svg>`,
  length_mm: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h20"/><path d="m5 9-3 3 3 3"/><path d="m19 9 3 3-3 3"/></svg>`,
  width_mm: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h20"/><path d="m5 9-3 3 3 3"/><path d="m19 9 3 3-3 3"/></svg>`,
  height_mm: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20"/><path d="m9 5 3-3 3 3"/><path d="m9 19 3 3 3-3"/></svg>`,
  number_of_cells: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>`,
  default: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
};

function getFieldIcon(name) {
  return FIELD_ICONS[name] || FIELD_ICONS.default;
}

export async function initPredict() {
  const form = document.getElementById("predict-form");
  const mainContainer = document.getElementById("main-fields-container");
  const optionalContainer = document.getElementById("optional-fields-container");
  const detailsToggle = document.getElementById("details-toggle");
  const detailsCollapsed = document.getElementById("details-collapsed");
  const btnSubmit = document.getElementById("btn-predict-submit");
  const btnExample = document.getElementById("link-try-example");
  const btnClear = document.getElementById("link-clear-form");
  const resultBox = document.getElementById("prediction-result-box");
  const unusualWarning = document.getElementById("unusual-warning");

  // 1. Fetch Schema and Render Inputs
  try {
    const data = await fetchJson("/api/predict/schema");
    if (!data) throw new Error("Could not load schema");

    renderMainFields(data.main_fields || [], mainContainer);
    renderOptionalFields(data.optional_fields || [], optionalContainer);
    schemaLoaded = true;
    validateForm();
  } catch (err) {
    console.error("Schema error:", err);
    if (mainContainer) {
      mainContainer.innerHTML = `<div style="grid-column: 1/-1; color: var(--burgundy); padding: 16px;">Error loading prediction fields. Please refresh.</div>`;
    }
    return;
  }

  // 2. Collapsible Details Toggle
  if (detailsToggle && detailsCollapsed) {
    const toggleDetails = () => {
      const isOpen = detailsCollapsed.classList.toggle("open");
      detailsToggle.innerHTML = isOpen ? "▼ Hide extra details" : "▶ More details (optional)";
    };
    detailsToggle.addEventListener("click", toggleDetails);
    detailsToggle.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggleDetails();
      }
    });
  }

  // 3. Validation on input change
  if (form) {
    form.addEventListener("input", validateForm);
  }

  // 4. Try an Example
  if (btnExample) {
    btnExample.addEventListener("click", async () => {
      try {
        const originalText = btnExample.innerHTML;
        btnExample.textContent = "Loading...";
        const data = await fetchJson("/api/predict/example");
        if (!data) throw new Error("Could not load example");
        const inputs = data.inputs || {};

        // Fill main and optional fields
        let hasOptional = false;
        Object.entries(inputs).forEach(([key, val]) => {
          const el = document.getElementById(`field-${key}`);
          if (el) {
            el.value = (val !== null && val !== undefined) ? val : "";
            if (optionalContainer && optionalContainer.contains(el) && val !== null && val !== "") {
              hasOptional = true;
            }
          }
        });

        if (hasOptional && detailsCollapsed && !detailsCollapsed.classList.contains("open")) {
          detailsCollapsed.classList.add("open");
          if (detailsToggle) detailsToggle.innerHTML = "▼ Hide extra details";
        }

        validateForm();
        btnExample.innerHTML = originalText;

        // Automatically submit the example for instant feedback
        form.requestSubmit();
      } catch (err) {
        console.error("Example error:", err);
        btnExample.innerHTML = "✦ Try an example";
      }
    });
  }

  // 5. Clear Form
  if (btnClear) {
    btnClear.addEventListener("click", () => {
      if (form) form.reset();
      clearErrors();
      if (resultBox) resultBox.classList.remove("visible");
      if (unusualWarning) {
        unusualWarning.classList.remove("visible");
        unusualWarning.textContent = "";
      }
      if (detailsCollapsed) {
        detailsCollapsed.classList.remove("open");
        if (detailsToggle) detailsToggle.innerHTML = "▶ More details (optional)";
      }
      validateForm();
    });
  }

  // 6. Handle Form Submit
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!validateForm()) return;

      btnSubmit.disabled = true;
      btnSubmit.innerHTML = `
        <svg class="spin-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>
        Predicting Range...
      `;

      const formData = new FormData(form);
      const payload = {};
      for (const [key, value] of formData.entries()) {
        const trimmed = value.trim();
        if (trimmed !== "") {
          payload[key] = trimmed;
        }
      }

      try {
        const data = await executePrediction(payload);
        if (!data) {
          throw new Error("Failed to predict");
        }

        // Update results
        const bestLabel = document.getElementById("result-best-model-label");
        const bestValue = document.getElementById("result-best-value");
        if (bestLabel) bestLabel.textContent = `Best Model: ${data.best_model}`;
        if (bestValue) bestValue.textContent = data.best_prediction;

        // Update individual model bars
        const predictions = data.predictions || {};
        const bars = data.bars || {};

        // Linear Regression
        const lrVal = document.getElementById("val-lr");
        const lrFill = document.getElementById("fill-lr");
        if (lrVal) lrVal.textContent = `${predictions["Linear Regression"] || 0} km`;
        if (lrFill) lrFill.style.width = `${bars["Linear Regression"] || 0}%`;

        // Random Forest
        const rfVal = document.getElementById("val-rf");
        const rfFill = document.getElementById("fill-rf");
        if (rfVal) rfVal.textContent = `${predictions["Random Forest"] || 0} km`;
        if (rfFill) rfFill.style.width = `${bars["Random Forest"] || 0}%`;

        // Gradient Boosting
        const gbVal = document.getElementById("val-gb");
        const gbFill = document.getElementById("fill-gb");
        if (gbVal) gbVal.textContent = `${predictions["Gradient Boosting"] || 0} km`;
        if (gbFill) gbFill.style.width = `${bars["Gradient Boosting"] || 0}%`;

        // Warning handling
        if (unusualWarning) {
          if (data.unusual_note) {
            unusualWarning.textContent = data.unusual_note;
            unusualWarning.classList.add("visible");
          } else {
            unusualWarning.classList.remove("visible");
          }
        }

        // Show result box
        if (resultBox) {
          resultBox.classList.add("visible");
          resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      } catch (err) {
        console.error("Prediction error:", err);
        if (unusualWarning) {
          unusualWarning.textContent = err.message || "Error running prediction.";
          unusualWarning.classList.add("visible");
        }
      } finally {
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
          Predict EV Driving Range
        `;
      }
    });
  }

  // --- Render Helpers ---
  function renderMainFields(fields, container) {
    if (!container) return;
    container.innerHTML = fields.map(f => {
      const iconSvg = getFieldIcon(f.name);
      return `
        <div class="field-card ${f.required ? "required" : ""}">
          <div class="field-header">
            <label for="field-${f.name}" class="field-label">
              <span class="field-icon">${iconSvg}</span>
              <span>${f.label}</span>
              ${f.required ? '<span class="required-star" title="Required">*</span>' : ""}
            </label>
            ${f.unit ? `<span class="unit-badge">${f.unit}</span>` : ""}
          </div>
          <div class="input-control-wrap">
            <input
              type="number"
              id="field-${f.name}"
              name="${f.name}"
              class="field-input"
              step="any"
              placeholder="e.g. ${f.median}"
              ${f.required ? "required" : ""}
              data-label="${f.label}"
            >
          </div>
          <div class="field-footer">
            <span class="field-hint">Usual: <strong>${f.min} – ${f.max}</strong> ${f.unit || ""}</span>
            <span class="field-err" id="err-${f.name}"></span>
          </div>
        </div>
      `;
    }).join("");
  }

  function renderOptionalFields(fields, container) {
    if (!container) return;
    container.innerHTML = fields.map(f => {
      const iconSvg = getFieldIcon(f.name);
      if (f.type === "select") {
        const optionsHtml = (f.options || []).map(opt => `<option value="${opt}">${opt}</option>`).join("");
        return `
          <div class="field-card">
            <div class="field-header">
              <label for="field-${f.name}" class="field-label">
                <span class="field-icon">${iconSvg}</span>
                <span>${f.label}</span>
              </label>
              <span class="unit-badge">Select</span>
            </div>
            <div class="input-control-wrap">
              <select id="field-${f.name}" name="${f.name}" class="field-input field-select">
                <option value="">-- Any ${f.label} --</option>
                ${optionsHtml}
              </select>
            </div>
            <div class="field-footer">
              <span class="field-hint">Vehicle specification category</span>
            </div>
          </div>
        `;
      }
      return `
        <div class="field-card">
          <div class="field-header">
            <label for="field-${f.name}" class="field-label">
              <span class="field-icon">${iconSvg}</span>
              <span>${f.label}</span>
            </label>
            ${f.unit ? `<span class="unit-badge">${f.unit}</span>` : ""}
          </div>
          <div class="input-control-wrap">
            <input
              type="number"
              id="field-${f.name}"
              name="${f.name}"
              class="field-input"
              step="any"
              placeholder="e.g. ${f.median}"
              data-label="${f.label}"
            >
          </div>
          <div class="field-footer">
            <span class="field-hint">Usual: <strong>${f.min} – ${f.max}</strong> ${f.unit || ""}</span>
            <span class="field-err" id="err-${f.name}"></span>
          </div>
        </div>
      `;
    }).join("");
  }

  function validateForm() {
    if (!schemaLoaded) return false;
    let isValid = true;

    // Check battery capacity specifically
    const batteryInput = document.getElementById("field-battery_capacity_kWh");
    const batteryErr = document.getElementById("err-battery_capacity_kWh");
    if (batteryInput) {
      const val = batteryInput.value.trim();
      if (val === "") {
        isValid = false;
        if (batteryErr) {
          batteryErr.textContent = "";
          batteryErr.classList.remove("visible");
        }
      } else {
        const num = parseFloat(val);
        if (isNaN(num) || num <= 0) {
          isValid = false;
          if (batteryErr) {
            batteryErr.textContent = "Must be > 0";
            batteryErr.classList.add("visible");
          }
        } else if (batteryErr) {
          batteryErr.textContent = "";
          batteryErr.classList.remove("visible");
        }
      }
    }

    // Check all inputs for negative values
    const allNumeric = form.querySelectorAll("input[type='number']");
    allNumeric.forEach(input => {
      const val = input.value.trim();
      const errEl = document.getElementById(`err-${input.name}`);
      if (val !== "") {
        const num = parseFloat(val);
        if (num < 0) {
          isValid = false;
          if (errEl) {
            errEl.textContent = "No negatives";
            errEl.classList.add("visible");
          }
        } else if (errEl && input.name !== "battery_capacity_kWh") {
          errEl.textContent = "";
          errEl.classList.remove("visible");
        }
      } else if (errEl && input.name !== "battery_capacity_kWh") {
        errEl.textContent = "";
        errEl.classList.remove("visible");
      }
    });

    if (btnSubmit) {
      btnSubmit.disabled = !isValid;
    }
    return isValid;
  }

  function clearErrors() {
    const errs = document.querySelectorAll(".field-err");
    errs.forEach(err => {
      err.textContent = "";
      err.classList.remove("visible");
    });
  }
}

// Auto-run if loaded directly as a module
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initPredict);
} else {
  initPredict();
}
