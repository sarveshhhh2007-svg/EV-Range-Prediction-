/**
 * predict.js - Simplified EV Range Prediction Studio
 * Handles dynamic field rendering, client-side validation,
 * prefilling example test cars, and running 3-model predictions.
 */

let schemaLoaded = false;

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
    const res = await fetch("/api/predict/schema");
    if (!res.ok) throw new Error("Could not load schema");
    const data = await res.json();

    renderMainFields(data.main_fields || [], mainContainer);
    renderOptionalFields(data.optional_fields || [], optionalContainer);
    schemaLoaded = true;
    validateForm();
  } catch (err) {
    console.error("Schema error:", err);
    if (mainContainer) {
      mainContainer.innerHTML = `<div style="grid-column: 1/-1; color: var(--burgundy);">Error loading prediction fields. Please refresh.</div>`;
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
        btnExample.textContent = "Loading...";
        const res = await fetch("/api/predict/example");
        if (!res.ok) throw new Error("Could not load example");
        const data = await res.json();
        const inputs = data.inputs || {};

        // Fill main and optional fields
        let hasOptional = false;
        Object.entries(inputs).forEach(([key, val]) => {
          const el = document.getElementById(`field-${key}`);
          if (el) {
            el.value = (val !== null && val !== undefined) ? val : "";
            // Check if this input belongs to optional container
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
        btnExample.textContent = "Try an example";

        // Automatically submit the example for instant feedback
        form.requestSubmit();
      } catch (err) {
        console.error("Example error:", err);
        btnExample.textContent = "Try an example";
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
      btnSubmit.textContent = "Predicting...";

      const formData = new FormData(form);
      const payload = {};
      for (const [key, value] of formData.entries()) {
        const trimmed = value.trim();
        if (trimmed !== "") {
          payload[key] = trimmed;
        }
      }

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.message || "Failed to predict");
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
        btnSubmit.textContent = "Predict EV Driving Range";
      }
    });
  }

  // --- Helpers ---
  function renderMainFields(fields, container) {
    if (!container) return;
    container.innerHTML = fields.map(f => `
      <div class="field-group">
        <label for="field-${f.name}" class="field-label">
          ${f.label} ${f.unit ? `(${f.unit})` : ""} ${f.required ? '<span style="color:var(--burgundy); font-weight:700;">*</span>' : ""}
        </label>
        <input
          type="number"
          id="field-${f.name}"
          name="${f.name}"
          class="field-input"
          step="any"
          placeholder="e.g. ${f.median}"
          ${f.required ? "required" : ""}
          data-required="${f.required ? "true" : "false"}"
          data-label="${f.label}"
        >
        <div class="field-hint">Usual range: ${f.min} – ${f.max} ${f.unit}</div>
        <div class="field-err" id="err-${f.name}"></div>
      </div>
    `).join("");
  }

  function renderOptionalFields(fields, container) {
    if (!container) return;
    container.innerHTML = fields.map(f => {
      if (f.type === "select") {
        const optionsHtml = (f.options || []).map(opt => `<option value="${opt}">${opt}</option>`).join("");
        return `
          <div class="field-group">
            <label for="field-${f.name}" class="field-label">${f.label}</label>
            <select id="field-${f.name}" name="${f.name}" class="field-input">
              <option value="">-- Any ${f.label} --</option>
              ${optionsHtml}
            </select>
          </div>
        `;
      }
      return `
        <div class="field-group">
          <label for="field-${f.name}" class="field-label">
            ${f.label} ${f.unit ? `(${f.unit})` : ""}
          </label>
          <input
            type="number"
            id="field-${f.name}"
            name="${f.name}"
            class="field-input"
            step="any"
            placeholder="e.g. ${f.median}"
            data-label="${f.label}"
          >
          <div class="field-hint">Usual range: ${f.min} – ${f.max} ${f.unit || ""}</div>
          <div class="field-err" id="err-${f.name}"></div>
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
      } else {
        const num = parseFloat(val);
        if (isNaN(num) || num <= 0) {
          isValid = false;
          if (batteryErr) {
            batteryErr.textContent = "Battery capacity must be greater than 0.";
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
            errEl.textContent = "Cannot be negative.";
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
