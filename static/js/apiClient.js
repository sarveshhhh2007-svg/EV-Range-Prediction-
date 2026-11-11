/**
 * apiClient.js - Universal Dual-Mode API Client
 * Seamlessly switches between live Flask backend (/api/...) and
 * 100% static GitHub Pages hosting (./static/data/*.json + client-side ML engine).
 */

import { loadModelsBundle, predictClient } from "./predictEngine.js?v=3.0";

const isStaticHost = window.location.hostname.endsWith("github.io") ||
                     window.location.protocol === "file:" ||
                     (!["localhost", "127.0.0.1"].includes(window.location.hostname) && !window.location.port);

const STATIC_MAP = {
  "/api/health": "./static/data/summary.json",
  "/api/summary": "./static/data/summary.json",
  "/api/dataset/preview": "./static/data/preview.json",
  "/api/cleaning": "./static/data/cleaning.json",
  "/api/insights": "./static/data/insights.json",
  "/api/models/metrics": "./static/data/models_metrics.json",
  "/api/predict/schema": "./static/data/schema.json",
  "/api/predict/example": "./static/data/example.json"
};

let cachedCars = null;

export async function fetchApi(endpoint) {
  const clean = endpoint.split("?")[0];

  // Try live server endpoint first if on localhost/custom server
  if (!isStaticHost) {
    try {
      const res = await fetch(endpoint);
      if (res.ok) return res;
    } catch (_) {
      // Fall through to static fallback
    }
  }

  // Fallback to static JSON file (for GitHub Pages)
  const staticUrl = STATIC_MAP[clean] || endpoint;
  return fetch(staticUrl);
}

export async function fetchJson(endpoint) {
  const res = await fetchApi(endpoint);
  if (!res.ok) throw new Error(`Failed to load ${endpoint}`);
  return res.json();
}

export async function getCarsPreview(page = 1, search = "", pageSize = 10) {
  // If running live Flask backend:
  if (!isStaticHost) {
    try {
      const url = `/api/dataset/preview?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(search)}`;
      const res = await fetch(url);
      if (res.ok) return await res.json();
    } catch (_) {
      // Fall through to client-side pagination
    }
  }

  // Static / GitHub Pages client-side pagination:
  if (!cachedCars) {
    const data = await fetchJson("/api/dataset/preview");
    cachedCars = data.data || [];
  }

  let filtered = cachedCars;
  if (search) {
    const q = search.toLowerCase();
    filtered = cachedCars.filter(c =>
      (c.brand && String(c.brand).toLowerCase().includes(q)) ||
      (c.model && String(c.model).toLowerCase().includes(q))
    );
  }

  const totalRecords = filtered.length;
  const totalPages = Math.max(1, Math.ceil(totalRecords / pageSize));
  const validPage = Math.min(Math.max(1, page), totalPages);
  const startIdx = (validPage - 1) * pageSize;
  const pageSlice = filtered.slice(startIdx, startIdx + pageSize);

  return {
    page: validPage,
    page_size: pageSize,
    total_records: totalRecords,
    total_pages: totalPages,
    columns: ["Brand", "Model", "Battery (kWh)", "Top Speed (km/h)", "0-100 km/h (s)", "Range (km)"],
    data: pageSlice
  };
}

export async function executePrediction(payload) {
  // If live backend is available:
  if (!isStaticHost) {
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) return await res.json();
    } catch (_) {
      // Fall through to client-side engine
    }
  }

  // Pure client-side ML execution on GitHub Pages
  const bundle = await loadModelsBundle("./static/data");
  if (!bundle) {
    throw new Error("Prediction engine bundle could not be loaded.");
  }
  return predictClient(payload, bundle);
}
