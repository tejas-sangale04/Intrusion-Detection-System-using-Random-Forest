import type { AlertsResponse, BatchResult, Decision, Explanation, Flow, Health, Metrics, ModelInfo } from "./types";

// Set VITE_API_URL to point the dashboard at another server.
export const API_URL: string = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://127.0.0.1:8000";
export const MAX_BATCH = 1000; // matches the API's default IDS_MAX_BATCH

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public details: { flow?: number; location?: string; error: string }[] = [],
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(0, `Cannot reach the API at ${API_URL}. Is uvicorn running?`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`;
    throw new ApiError(res.status, detail, body.errors ?? []);
  }
  return body as T;
}

const post = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),
  modelInfo: () => request<ModelInfo>("/model/info"),
  metrics: () => request<Metrics>("/metrics"),
  alerts: (limit = 100, decision?: Decision) =>
    request<AlertsResponse>(`/alerts?limit=${limit}${decision ? `&decision=${decision}` : ""}`),
  explanation: (id: string, k = 10) => request<Explanation>(`/explanations/${encodeURIComponent(id)}?k=${k}`),
  predictBatch: (flows: Flow[], source: string) => post<BatchResult>("/predict/batch", { flows, source }),
};

/** Score any number of flows by sending them in API-sized chunks. */
export async function scoreAll(flows: Flow[], source: string, onProgress?: (done: number) => void) {
  const results: BatchResult[] = [];
  for (let i = 0; i < flows.length; i += MAX_BATCH) {
    try {
      results.push(await api.predictBatch(flows.slice(i, i + MAX_BATCH), source));
    } catch (e) {
      // Re-number per-flow errors so they point at the row in the whole file.
      if (e instanceof ApiError) {
        e.details = e.details.map((d) => (d.flow === undefined ? d : { ...d, flow: d.flow + i }));
      }
      throw e;
    }
    onProgress?.(Math.min(i + MAX_BATCH, flows.length));
  }
  return results.flatMap((r) => r.predictions);
}
