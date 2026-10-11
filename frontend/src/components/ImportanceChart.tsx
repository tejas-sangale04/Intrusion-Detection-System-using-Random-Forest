import { useTooltip } from "./Tooltip";

/** Mean |SHAP| per feature over the Phase 6 stratified sample. */
export function ImportanceChart({ rows, source, k = 10 }: { rows: Record<string, string | number>[]; source: string; k?: number }) {
  const { bind, node } = useTooltip();
  const top = rows
    .map((r) => ({ feature: String(r[""] ?? r.feature), value: Number(r.mean_abs_shap) }))
    .slice(0, k);
  const max = Math.max(...top.map((t) => t.value), 1e-9);
  return (
    <div>
      <div className="bars">
        {top.map((t) => (
          <FeatureBar key={t.feature} t={t} max={max} bind={bind} />
        ))}
      </div>
      {node}
      <p className="small muted" style={{ marginTop: 8 }}>
        Average size of each feature's contribution to the attack probability, on a stratified sample of test
        flows. It shows what the model relies on, not what causes attacks. Source: <code>{source}</code>.
      </p>
    </div>
  );
}

function FeatureBar({ t, max, bind }: { t: { feature: string; value: number }; max: number; bind: ReturnType<typeof useTooltip>["bind"] }) {
  return (
    <>
      <span className="name" title={t.feature}>{t.feature}</span>
      <div className="track axis">
        <div className="bar-row bar-hit" {...bind(<><strong>{t.feature}</strong><br />mean |SHAP| {t.value.toFixed(4)}</>)}>
          <div className="bar" style={{ width: `${(100 * t.value) / max}%`, background: "var(--series-1)" }} />
          <span className="v">{t.value.toFixed(3)}</span>
        </div>
      </div>
    </>
  );
}
