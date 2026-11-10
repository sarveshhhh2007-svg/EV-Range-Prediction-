/**
 * API Service Module.
 * Provides abstracted fetch methods, memory caching, error handling,
 * and background training polling.
 */

const API_CACHE = new Map();

export async function fetchJson(endpoint, options = {}, useCache = true) {
  const cacheKey = endpoint + JSON.stringify(options);
  if (useCache && !options.method && API_CACHE.has(cacheKey)) {
    return API_CACHE.get(cacheKey);
  }

  const defaultHeaders = {
    'Accept': 'application/json'
  };

  if (options.body && typeof options.body === 'object' && !(options.body instanceof FormData)) {
    defaultHeaders['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  const response = await fetch(endpoint, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers
    }
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const errorMsg = data?.error?.message || `HTTP ${response.status}: Request failed`;
    const err = new Error(errorMsg);
    err.status = response.status;
    err.code = data?.error?.code || 'API_ERROR';
    throw err;
  }

  if (useCache && (!options.method || options.method === 'GET')) {
    API_CACHE.set(cacheKey, data);
  }

  return data;
}

export function clearApiCache() {
  API_CACHE.clear();
}

export const API = {
  getHealth: () => fetchJson('/api/health', {}, false),
  getSummary: () => fetchJson('/api/summary'),
  getDatasetPreview: (params) => {
    const query = new URLSearchParams(params).toString();
    return fetchJson(`/api/dataset/preview?${query}`, {}, false);
  },
  getDatasetSchema: () => fetchJson('/api/dataset/schema'),
  getPreprocessing: () => fetchJson('/api/preprocessing'),
  getEDAStats: () => fetchJson('/api/eda'),
  getEDACorrelation: () => fetchJson('/api/eda/correlation'),
  getEDADistribution: () => fetchJson('/api/eda/distribution'),
  getEDAScatter: (feature) => fetchJson(`/api/eda/scatter?feature=${encodeURIComponent(feature)}`),
  getModelStatus: () => fetchJson('/api/models/status', {}, false),
  getModelMetrics: () => fetchJson('/api/models/metrics'),
  getModelImportance: () => fetchJson('/api/models/importance'),
  startTraining: () => fetchJson('/api/models/train', { method: 'POST' }, false),
  getTrainingStatus: () => fetchJson('/api/models/train/status', {}, false),
  getPredictSchema: () => fetchJson('/api/predict/schema'),
  getPredictExample: (index = null) => {
    const q = index !== null ? `?index=${index}` : '';
    return fetchJson(`/api/predict/example${q}`, {}, false);
  },
  predictRange: (payload) => fetchJson('/api/predict', { method: 'POST', body: payload }, false)
};

