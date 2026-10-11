import type { Contribution } from "../types";
import { useTooltip } from "./Tooltip";

const fmt = (v: number) => (Math.abs(v) >= 1e4 ? v.toExponential(2) : Number(v.toPrecision(4)).toLocaleString());

/** Diverging bars: right = pushed towards attack, left = towards benign. */
export function ShapChart({ contributions }: { contributions: Contribution[] }) {
  const { bind, node } = useTooltip();
  const max = Math.max(...contributions.map((c) => Math.abs(c.shap)), 1e-9);
  return (
    <div>
      <div className="legend">
        <span><span className="swatch" style={{ background: "var(--diverge-benign)" }} />Pushes towards benign</span>
        <span><span className="swatch" style={{ background: "var(--diverge-attack)" }} />Pushes towards attack</span>
      </div>
      <div className="diverge">
        {contributions.map((c) => {
          const width = `${(100 * Math.abs(c.shap)) / max}%`;
          const tip = bind(<><strong>{c.feature}</strong> = {fmt(c.value)}<br />SHAP {c.shap >= 0 ? "+" : ""}{c.shap.toFixed(4)}</>);
          return (
            <div key={c.feature} style={{ display: "contents" }}>
              <span className="name small" title={c.feature}>
                {c.feature} <span className="muted">= {fmt(c.value)}</span>
              </span>
              <div className="halves bar-hit" {...tip}>
                <div className="neg">
                  {c.shap < 0 && <div className="bar" style={{ width, background: "var(--diverge-benign)" }} />}
                </div>
                <div className="pos">
                  {c.shap > 0 && <div className="bar" style={{ width, background: "var(--diverge-attack)" }} />}
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {node}
    </div>
  );
}
