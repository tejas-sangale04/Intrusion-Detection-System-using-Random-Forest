import { useState } from "react";
import { ApiError, scoreAll } from "../api";
import { parseFlowsCsv } from "../csv";
import type { Decision, Flow, Prediction } from "../types";
import { DecisionBadge } from "./DecisionBadge";

const MAX_ROWS = 20000; // keeps the browser responsive; split larger files
const SHOWN_ROWS = 500;

interface Scored { prediction: Prediction; label: string | null }

export function ScorePanel({ onExplain, selected, onScored }: {
  onExplain: (id: string) => void; selected: string | null; onScored: () => void;
}) {
  const [results, setResults] = useState<Scored[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [onlyAlerts, setOnlyAlerts] = useState(false);

  async function run(flows: Flow[], labels: (string | null)[], source: string, what: string) {
    if (flows.length > MAX_ROWS) {
      setErrors([`${flows.length.toLocaleString()} rows; the dashboard sends at most ${MAX_ROWS.toLocaleString()} at a time. Split the file.`]);
      return;
    }
    setErrors([]);
    setResults([]);
    setStatus(`Scoring ${flows.length.toLocaleString()} flows from ${what}…`);
    setProgress({ done: 0, total: flows.length });
    try {
      const preds = await scoreAll(flows, source, (done) => setProgress({ done, total: flows.length }));
      setResults(preds.map((p, i) => ({ prediction: p, label: labels[i] })));
      setStatus(`Scored ${preds.length.toLocaleString()} flows from ${what}.`);
      onScored();
    } catch (e) {
      setStatus(null);
      if (e instanceof ApiError) {
        setErrors([e.message, ...e.details.slice(0, 10).map((d) =>
          `${d.flow !== undefined ? `Row ${d.flow + 1}: ` : d.location ? `${d.location}: ` : ""}${d.error}`)]);
      } else {
        setErrors([String(e)]);
      }
    } finally {
      setProgress(null);
    }
  }

  async function loadSample() {
    // Real test-split flows written by `python -m backend.examples`.
    const sample = (await import("../../../examples/batch.json")).default as { flows: Record<string, unknown>[] };
    const labels = sample.flows.map((f) => (typeof f.Label === "string" ? f.Label : null));
    const flows = sample.flows.map(({ Label: _drop, ...rest }) => rest as Flow);
    await run(flows, labels, "sample", "the sample file");
  }

  async function loadFile(file: File) {
    try {
      const { flows, labels } = parseFlowsCsv(await file.text());
      await run(flows, labels, "csv-upload", file.name);
    } catch (e) {
      setErrors([e instanceof Error ? e.message : String(e)]);
    }
  }

  const counts = results.reduce<Record<Decision, number>>(
    (acc, r) => ({ ...acc, [r.prediction.decision]: acc[r.prediction.decision] + 1 }),
    { benign: 0, anomaly: 0, known_attack: 0 },
  );
  const hasLabels = results.some((r) => r.label !== null);
  const filtered = onlyAlerts ? results.map((r, i) => ({ r, i })).filter(({ r }) => r.prediction.decision !== "benign")
    : results.map((r, i) => ({ r, i }));

  return (
    <section className="card">
      <h2>Score flows</h2>
      <p className="sub">
        Send CIC-IDS2017 flow records to the API. Use a CSV with the original column names (a slice of one of the
        dataset files works), or the six sample flows. This is offline scoring of recorded data, not live traffic.
      </p>
      <div className="row">
        <button onClick={loadSample} disabled={!!progress}>Score the sample flows</button>
        <label className="row">
          <span className="muted">or a CSV file:</span>
          <input type="file" accept=".csv,text/csv" disabled={!!progress}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void loadFile(f); e.target.value = ""; }} />
        </label>
      </div>
      {progress && (
        <div style={{ marginTop: 12 }}>
          <div className="progress" role="progressbar" aria-valuenow={progress.done} aria-valuemax={progress.total}>
            <div style={{ width: `${(100 * progress.done) / progress.total}%` }} />
          </div>
        </div>
      )}
      {status && <p>{status}</p>}
      {errors.length > 0 && (
        <div className="error" role="alert">{errors.map((e, i) => <div key={i}>{e}</div>)}</div>
      )}
      {results.length > 0 && (
        <>
          <p className="row">
            <DecisionBadge decision="benign" /> {counts.benign.toLocaleString()}
            <DecisionBadge decision="anomaly" /> {counts.anomaly.toLocaleString()}
            <DecisionBadge decision="known_attack" /> {counts.known_attack.toLocaleString()}
            <label className="row small" style={{ marginLeft: "auto" }}>
              <input type="checkbox" checked={onlyAlerts} onChange={(e) => setOnlyAlerts(e.target.checked)} /> Alerts only
            </label>
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">Row</th>{hasLabels && <th>Label in file</th>}<th>Decision</th>
                  <th className="num">Attack prob.</th><th className="num">Anomaly score</th><th>Imputed</th><th />
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, SHOWN_ROWS).map(({ r, i }) => (
                  <tr key={r.prediction.prediction_id} className={selected === r.prediction.prediction_id ? "selected" : ""}>
                    <td className="num">{i + 1}</td>
                    {hasLabels && <td>{r.label ?? "–"}</td>}
                    <td><DecisionBadge decision={r.prediction.decision} /></td>
                    <td className="num">{r.prediction.attack_probability.toFixed(3)}</td>
                    <td className="num">{r.prediction.anomaly_score.toFixed(3)}</td>
                    <td title={r.prediction.missing_values.join(", ")}>{r.prediction.missing_values.length || ""}</td>
                    <td><button onClick={() => onExplain(r.prediction.prediction_id)}>Explain</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filtered.length > SHOWN_ROWS && (
            <p className="small muted">Showing the first {SHOWN_ROWS} of {filtered.length.toLocaleString()} rows.</p>
          )}
          {hasLabels && (
            <p className="small muted">
              "Label in file" is the dataset's ground truth, shown for comparison only. It was not sent to the model.
            </p>
          )}
        </>
      )}
    </section>
  );
}
