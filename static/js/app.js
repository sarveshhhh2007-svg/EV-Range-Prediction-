/**
 * app.js - Main Application Controller
 * Handles loading screen orchestration, top navigation, data table preview,
 * pagination, search, summary metrics, and chart initialization.
 */

import { initInsightsCharts, initModelsCharts } from "./charts.js?v=3.0";
import { fetchJson, getCarsPreview } from "./apiClient.js?v=3.0";

// Global Data Table State
let currentPage = 1;
let currentSearch = "";
let totalPages = 1;

document.addEventListener("DOMContentLoaded", () => {
  initApp();
});

async function initApp() {
  const startTime = Date.now();
  const loadingScreen = document.getElementById("loading-screen");
  const loaderBar = document.getElementById("loader-bar");
  const loaderStatus = document.getElementById("loader-status");
  const loaderRetryBtn = document.getElementById("loader-retry-btn");

  // Timeout guard (8 seconds)
  let isReady = false;
  const timeoutId = setTimeout(() => {
    if (!isReady && loadingScreen) {
      if (loaderStatus) loaderStatus.textContent = "Loading is taking longer than usual.";
      if (loaderRetryBtn) loaderRetryBtn.style.display = "inline-block";
    }
  }, 8000);

  try {
    // 1. Initial status
    if (loaderStatus) loaderStatus.textContent = "Reading the dataset...";
    if (loaderBar) loaderBar.style.width = "35%";

    // 2. Fetch summary
    const summaryData = await fetchJson("/api/summary");
    if (!summaryData) {
      throw new Error("Application data unavailable");
    }
    populateSummary(summaryData);

    // 3. Update loading progress
    if (loaderStatus) loaderStatus.textContent = "Loading trained models...";
    if (loaderBar) loaderBar.style.width = "75%";

    // 4. Initialize charts & data table
    await Promise.all([
      initInsightsCharts(),
      initModelsCharts(),
      loadDataTable(1, ""),
      loadCleaningBullets()
    ]);

    // 5. Setup navigation and section visibility
    setupNavigation();
    setupSectionObserver();

    // 6. Complete loading screen with min 1200ms duration
    if (loaderStatus) loaderStatus.textContent = "Ready";
    if (loaderBar) loaderBar.style.width = "100%";

    const elapsed = Date.now() - startTime;
    const remainingTime = Math.max(0, 1200 - elapsed);

    setTimeout(() => {
      isReady = true;
      clearTimeout(timeoutId);
      if (loadingScreen) {
        loadingScreen.classList.add("fade-out");
        setTimeout(() => {
          loadingScreen.style.display = "none";
        }, 500);
      }
    }, remainingTime);

  } catch (err) {
    console.error("Initialization error:", err);
    clearTimeout(timeoutId);
    if (loaderStatus) loaderStatus.textContent = "Error loading application data.";
    if (loaderRetryBtn) loaderRetryBtn.style.display = "inline-block";
  }
}

// -------------------------------------------------------------
// Summary Section Metrics
// -------------------------------------------------------------
function populateSummary(data) {
  const carsEl = document.getElementById("home-count-cars");
  const specsEl = document.getElementById("home-count-specs");
  const modelsEl = document.getElementById("home-count-models");
  const bannerEl = document.getElementById("home-best-model-banner");

  if (carsEl && data.total_vehicles) carsEl.textContent = data.total_vehicles;
  if (specsEl && data.total_features) specsEl.textContent = data.total_features;
  if (modelsEl && data.total_models) modelsEl.textContent = data.total_models;
  if (bannerEl && data.best_model && data.best_model_mae !== undefined) {
    bannerEl.textContent = `Best model: ${data.best_model}, average error ${data.best_model_mae} km`;
  }
}

// -------------------------------------------------------------
// Data Table Preview & Pagination
// -------------------------------------------------------------
async function loadDataTable(page = 1, search = "") {
  const tbody = document.getElementById("cars-table-body");
  const pageIndicator = document.getElementById("page-indicator");
  const countIndicator = document.getElementById("data-total-count");
  const btnPrev = document.getElementById("btn-prev-page");
  const btnNext = document.getElementById("btn-next-page");

  if (!tbody) return;

  try {
    const data = await getCarsPreview(page, search, 10);

    currentPage = data.page;
    totalPages = data.total_pages;

    if (data.data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--muted);">No matching vehicles found.</td></tr>`;
    } else {
      tbody.innerHTML = data.data.map(car => `
        <tr>
          <td><strong>${escapeHtml(car.brand)}</strong></td>
          <td>${escapeHtml(car.model)}</td>
          <td class="tabular">${car.battery_capacity !== null ? car.battery_capacity : "—"}</td>
          <td class="tabular">${car.top_speed !== null ? car.top_speed : "—"}</td>
          <td class="tabular">${car.acceleration !== null ? car.acceleration : "—"}</td>
          <td class="tabular" style="font-weight:700; color:var(--burgundy);">${car.range !== null ? car.range : "—"}</td>
        </tr>
      `).join("");
    }

    if (pageIndicator) pageIndicator.textContent = `Page ${data.page} of ${data.total_pages}`;
    if (countIndicator) countIndicator.textContent = `${data.total_records} cars`;
    if (btnPrev) btnPrev.disabled = data.page <= 1;
    if (btnNext) btnNext.disabled = data.page >= data.total_pages;

  } catch (err) {
    console.error("Table preview error:", err);
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 20px; color: var(--burgundy);">Error loading vehicle data.</td></tr>`;
  }
}

// -------------------------------------------------------------
// Data Cleaning Bullets
// -------------------------------------------------------------
async function loadCleaningBullets() {
  const listEl = document.getElementById("cleaning-bullets-list");
  if (!listEl) return;

  try {
    const data = await fetchJson("/api/cleaning");
    if (data && Array.isArray(data.bullets) && data.bullets.length > 0) {
      listEl.innerHTML = data.bullets.map(b => `<li>${escapeHtml(b)}</li>`).join("");
    }
  } catch (err) {
    console.warn("Could not load cleaning bullets:", err);
  }
}

// -------------------------------------------------------------
// Top Navigation & Scroll Spy
// -------------------------------------------------------------
function setupNavigation() {
  const menuToggle = document.getElementById("menu-toggle");
  const navLinks = document.getElementById("nav-links");
  const links = document.querySelectorAll(".nav-item a");

  // Mobile menu toggle
  if (menuToggle && navLinks) {
    menuToggle.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });
  }

  // Smooth scroll and link click
  links.forEach(link => {
    link.addEventListener("click", (e) => {
      const targetId = link.getAttribute("href");
      if (targetId && targetId.startsWith("#")) {
        const targetEl = document.querySelector(targetId);
        if (targetEl) {
          e.preventDefault();
          targetEl.scrollIntoView({ behavior: "smooth" });
          if (navLinks) navLinks.classList.remove("open");
        }
      }
    });
  });

  // Data search input debounce
  const searchInput = document.getElementById("data-search");
  if (searchInput) {
    let debounceTimer = null;
    searchInput.addEventListener("input", (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        currentSearch = e.target.value.trim();
        currentPage = 1;
        loadDataTable(currentPage, currentSearch);
      }, 300);
    });
  }

  // Pagination buttons
  const btnPrev = document.getElementById("btn-prev-page");
  const btnNext = document.getElementById("btn-next-page");

  if (btnPrev) {
    btnPrev.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        loadDataTable(currentPage, currentSearch);
      }
    });
  }

  if (btnNext) {
    btnNext.addEventListener("click", () => {
      if (currentPage < totalPages) {
        currentPage++;
        loadDataTable(currentPage, currentSearch);
      }
    });
  }

  // Scroll Spy for navigation active link
  const sections = document.querySelectorAll("section[id]");
  window.addEventListener("scroll", () => {
    let currentId = "";
    const scrollPos = window.scrollY + 140;

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        currentId = sec.getAttribute("id");
      }
    });

    if (currentId) {
      links.forEach(link => {
        if (link.getAttribute("href") === `#${currentId}`) {
          link.classList.add("active");
        } else {
          link.classList.remove("active");
        }
      });
    }
  }, { passive: true });
}

// -------------------------------------------------------------
// Section Reveal Animations
// -------------------------------------------------------------
function setupSectionObserver() {
  const sections = document.querySelectorAll(".section");
  if (!("IntersectionObserver" in window)) {
    sections.forEach(sec => sec.classList.add("visible"));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  sections.forEach(sec => observer.observe(sec));
}

function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
