// All requests go through the Vite proxy at /api, which forwards to the
// FastAPI service. The API key is read from localStorage so a deployment with
// auth enabled works without rebuilding the bundle.

const KEY_STORAGE = "llmsec_api_key";

export function getApiKey() {
  return localStorage.getItem(KEY_STORAGE) || "";
}

export function setApiKey(value) {
  if (value) localStorage.setItem(KEY_STORAGE, value);
  else localStorage.removeItem(KEY_STORAGE);
}

async function request(path, options = {}) {
  const key = getApiKey();
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(key ? { "X-API-Key": key } : {}),
      ...options.headers,
    },
  });

  if (!response.ok) {
    // FastAPI puts the reason in `detail`; fall back to the status line when
    // the body is not JSON (a proxy error, say).
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${response.status}: ${detail}`);
  }

  return response.json();
}

export const getHealth = () => request("/health");
export const getStats = () => request("/stats");
export const listAnalyses = (limit = 25) => request(`/analyses?limit=${limit}`);
export const getAnalysis = (id) => request(`/analyses/${id}`);

export const analyze = (payload) =>
  request("/analyze", { method: "POST", body: JSON.stringify(payload) });
