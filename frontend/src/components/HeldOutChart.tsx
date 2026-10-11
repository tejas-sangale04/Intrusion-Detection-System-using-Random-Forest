import { useState } from "react";
import type { HeldOutRow } from "../types";
import { useTooltip } from "./Tooltip";

const pct = (v: number) => `${(100 * v).toFixed(1)}%`;
const SERIES = [
  { key: "classifier_recall", name: "Random Forest alone", color: "var(--series-1)" },
  { key: "hybrid_recall", name: "Hybrid (RF + Isolation Forest)", color: "var(--series-2)" },
] as const;

/** Recall on each attack family when that family was left out of training (Phases 4 and 5). */
export function HeldOutChart({ rows, source }: { rows: HeldOutRow[]; source: string }) {
  const fprs = [...new Set(rows.map((r) => r.target_fpr))].sort();
  const [fpr, setFpr] = useState(fprs[fprs.length - 1]);
  const [asTable, setAsTable] = useState(false);
  const { bind, node } = useTooltip();
  const shown = rows
    .filter((r) => r.target_fpr === fpr && r.model === "random_forest")
    .sort((a, b) => b.held_out_rows - a.held_out_rows);
  const benignFpr = shown[0]?.hybrid_fpr;

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <label className="small">
          Anomaly threshold set for{" "}
          <select value={fpr} onChange={(e) => setFpr(Number(e.target.value))}>
            {fprs.map((f) => <option key={f} value={f}>{pct(f)} false positives</option>)}
          </select>
        </label>
        <button onClick={() => setAsTable(!asTable)}>{asTable ? "Show chart" : "Show table"}</button>
      </div>
      {asTable ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Held-out family</th><th className="num">Flows</th><th className="num">RF alone</th><th className="num">Hybrid</th><th className="num">of which flagged as anomaly</th></tr></thead>
            <tbody>
              {shown.map((r) => (
                <tr key={r.held_out}><td>{r.held_out}</td><td className="num">{r.held_out_rows.toLocaleString()}</td>
                  <td className="num">{pct(r.classifier_recall)}</td><td className="num">{pct(r.hybrid_recall)}</td>
                  <td className="num">{pct(r.hybrid_recall_as_anomaly)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <>
          <div className="legend">
            {SERIES.map((s) => <span key={s.key}><span className="swatch" style={{ background: s.color }} />{s.name}</span>)}
          </div>
          <div className="bars">
            {shown.map((r) => (
              <FamilyBars key={r.held_out} r={r} bind={bind} />
            ))}
          </div>
          {node}
        </>
      )}
      <p className="small muted" style={{ marginTop: 8 }}>
        Each family was removed from the classifier's training data, then scored on the test split. At this
        threshold the hybrid flags {benignFpr !== undefined ? pct(benignFpr) : "?"} of benign test flows. Families
        with only a handful of test flows (see the counts) give very uncertain percentages. Source:{" "}
        <code>{source}</code>.
      </p>
    </div>
  );
}

function FamilyBars({ r, bind }: { r: HeldOutRow; bind: ReturnType<typeof useTooltip>["bind"] }) {
  return (
    <>
      <span className="name" title={r.held_out}>
        {r.held_out}<br /><span className="muted">{r.held_out_rows.toLocaleString()} flows</span>
      </span>
      <div className="track axis">
        {SERIES.map((s) => (
          <div key={s.key} className="bar-row bar-hit" {...bind(
            <><strong>{r.held_out}</strong> ({r.held_out_rows.toLocaleString()} test flows)<br />{s.name}: {pct(r[s.key])}</>,
          )}>
            <div className="bar" style={{ width: `${100 * r[s.key]}%`, background: s.color }} />
            <span className="v">{pct(r[s.key])}</span>
          </div>
        ))}
      </div>
    </>
  );
}
