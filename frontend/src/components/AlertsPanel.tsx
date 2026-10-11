import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { AlertsResponse, Decision } from "../types";
import { DecisionBadge } from "./DecisionBadge";

const REFRESH_MS = 5000;

export function AlertsPanel({ onExplain, selected, refreshKey, active }: {
  onExplain: (id: string) => void; selected: string | null; refreshKey: number; active: boolean;
}) {
  const [filter, setFilter] = useState<Decision | "">("");
  const [auto, setAuto] = useState(true);
  const [data, setData] = useState<AlertsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.alerts(200, filter || undefined)
      .then((d) => { setData(d); setError(null); })
      .catch((e: ApiError) => setError(e.message));
  }, [filter]);

  useEffect(() => { load(); }, [load, refreshKey]);
  useEffect(() => {
    if (!auto || !active) return;
    const id = window.setInterval(load, REFRESH_MS);
    return () => window.clearInterval(id);
  }, [auto, active, load]);

  return (
    <section className="card">
      <h2>Alerts</h2>
      <p className="sub">
        Flows the API marked as <em>known attack</em> or <em>anomaly</em>, newest first. Alerts are records for an
        analyst. Nothing is blocked. The flow files carry no IP addresses, so no host or location is shown.
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <select aria-label="Filter alerts" value={filter} onChange={(e) => setFilter(e.target.value as Decision | "")}>
          <option value="">All alerts</option>
          <option value="known_attack">Known attacks</option>
          <option value="anomaly">Anomalies</option>
        </select>
        <label className="row small"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> Refresh every 5 s</label>
        <button onClick={load}>Refresh now</button>
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      {data && data.count === 0 && (
        <p className="muted">No alerts yet. Score some flows first; the API keeps alerts in memory until it restarts.</p>
      )}
      {data && data.count > 0 && (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Time (UTC)</th><th>Decision</th><th className="num">Attack prob.</th><th className="num">Anomaly score</th><th>Source</th><th /></tr></thead>
            <tbody>
              {data.alerts.map((a) => (
                <tr key={a.prediction_id} className={selected === a.prediction_id ? "selected" : ""}>
                  <td>{a.created_at.replace("T", " ").slice(0, 23)}</td>
                  <td><DecisionBadge decision={a.decision} /></td>
                  <td className="num">{a.attack_probability.toFixed(3)}</td>
                  <td className="num">{a.anomaly_score.toFixed(3)}</td>
                  <td>{a.source}</td>
                  <td><button onClick={() => onExplain(a.prediction_id)}>Explain</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && <p className="small muted" style={{ marginTop: 8 }}>{data.note}</p>}
    </section>
  );
}
