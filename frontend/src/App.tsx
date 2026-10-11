import { useCallback, useEffect, useState } from "react";
import { api, API_URL, ApiError } from "./api";
import { AlertsPanel } from "./components/AlertsPanel";
import { ExplanationPanel } from "./components/ExplanationPanel";
import { Overview } from "./components/Overview";
import { ScorePanel } from "./components/ScorePanel";
import { activeReplays } from "./replay";
import type { Health, Metrics, ModelInfo } from "./types";

type Tab = "overview" | "score" | "alerts";
const METRICS_REFRESH_MS = 5000;
const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "score", label: "Score flows" },
  { id: "alerts", label: "Alerts" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [health, setHealth] = useState<Health | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const refresh = useCallback(() => {
    Promise.all([api.health(), api.modelInfo(), api.metrics()])
      .then(([h, i, m]) => { setHealth(h); setInfo(i); setMetrics(m); setError(null); })
      .catch((e: ApiError) => setError(e.message));
  }, []);
  useEffect(refresh, [refresh]);
  // Poll so the overview counts and the replay banner follow a running replay.
  useEffect(() => {
    const id = window.setInterval(() => {
      api.metrics().then(setMetrics).catch(() => {});
    }, METRICS_REFRESH_MS);
    return () => window.clearInterval(id);
  }, []);
  const replays = activeReplays(metrics?.runtime);

  const onScored = () => { refresh(); setRefreshKey((k) => k + 1); };

  return (
    <div className="app">
      <header className="top">
        <h1>Explainable Hybrid Network IDS</h1>
        <span className="small muted">
          API {health ? `v${health.version} at ${API_URL}` : error ? "unreachable" : "connecting…"}
        </span>
      </header>
      <div className="banner" role="note">
        <strong>Offline demo mode.</strong> Flows come from recorded CIC-IDS2017 data that you submit. Nothing is
        captured from a network, and no host is blocked. Decisions are model outputs for analyst review, not proof
        of an attack.
      </div>
      {replays.map((r) => (
        <div key={r.name} className="banner replay" role="status">
          <strong>Replay in progress: {r.name}.</strong> {r.flows.toLocaleString()} recorded flows sent so far
          ({r.known_attack.toLocaleString()} known attack, {r.anomaly.toLocaleString()} anomaly). This is a
          simulated feed from the dataset at a fixed rate, not live traffic.
        </div>
      ))}
      {error && (
        <p className="error" role="alert">
          {error} Start it with <code>uvicorn backend.main:app --port 8000</code> from the repository root.{" "}
          <button onClick={refresh}>Retry</button>
        </p>
      )}
      <nav className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {/* Panels stay mounted so scoring results and filters survive tab switches. */}
      <div hidden={tab !== "overview"}><Overview metrics={metrics} info={info} /></div>
      <div hidden={tab !== "score"}><ScorePanel onExplain={setSelected} selected={selected} onScored={onScored} /></div>
      <div hidden={tab !== "alerts"}><AlertsPanel onExplain={setSelected} selected={selected} refreshKey={refreshKey} active={tab === "alerts"} /></div>
      {selected && tab !== "overview" && <ExplanationPanel predictionId={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
