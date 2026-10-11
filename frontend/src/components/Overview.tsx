import type { Metrics, ModelInfo } from "../types";
import { DecisionBadge } from "./DecisionBadge";
import { HeldOutChart } from "./HeldOutChart";
import { ImportanceChart } from "./ImportanceChart";

const pct = (v: number) => `${(100 * v).toFixed(2)}%`;

export function Overview({ metrics, info }: { metrics: Metrics | null; info: ModelInfo | null }) {
  if (!metrics) return <p className="muted">Loading metrics…</p>;
  const rt = metrics.runtime;
  const off = metrics.offline_evaluation;
  const rf = off.classifier_baselines?.data.random_forest?.test;

  return (
    <>
      <section className="card">
        <h2>This server session</h2>
        <p className="sub">Counts of decisions since the API started. Submitted flows carry no ground truth, so these are not accuracy figures.</p>
        <div className="tiles">
          <Tile label="Flows scored" value={rt.flows_scored.toLocaleString()} />
          <Tile label={<DecisionBadge decision="benign" />} value={rt.decision_counts.benign.toLocaleString()} />
          <Tile label={<DecisionBadge decision="anomaly" />} value={rt.decision_counts.anomaly.toLocaleString()} />
          <Tile label={<DecisionBadge decision="known_attack" />} value={rt.decision_counts.known_attack.toLocaleString()} />
          <Tile label="Model time per flow" value={rt.mean_scoring_ms_per_flow === null ? "–" : `${rt.mean_scoring_ms_per_flow} ms`} />
        </div>
      </section>

      <section className="card">
        <h2>Offline evaluation (saved by the training scripts)</h2>
        <p className="sub">
          Random split of CIC-IDS2017 test flows. Train and test share attack campaigns, so these numbers describe
          recognising attack types seen in training.
        </p>
        {rf ? (
          <div className="tiles">
            <Tile label="Precision" value={pct(rf.precision)} />
            <Tile label="Recall" value={pct(rf.recall)} />
            <Tile label="F1" value={pct(rf.f1)} />
            <Tile label="False-positive rate" value={pct(rf.fpr)} />
            <Tile label="Test flows" value={rf.n.toLocaleString()} />
          </div>
        ) : <p className="muted">Classifier results file not found.</p>}
        {off.classifier_baselines && <p className="small muted">Source: <code>{off.classifier_baselines.source}</code> (Random Forest, threshold 0.5).</p>}
      </section>

      <div className="grid2">
        <section className="card">
          <h2>Attack families never seen in training</h2>
          <p className="sub">The harder test: how much of a left-out family is still caught.</p>
          {off.hybrid_held_out_families
            ? <HeldOutChart rows={off.hybrid_held_out_families.data} source={off.hybrid_held_out_families.source} />
            : <p className="muted">Phase 5 results file not found.</p>}
        </section>
        <section className="card">
          <h2>What the classifier relies on</h2>
          <p className="sub">Top 10 features by mean absolute SHAP value.</p>
          {off.global_feature_importance
            ? <ImportanceChart rows={off.global_feature_importance.data} source={off.global_feature_importance.source} />
            : <p className="muted">Phase 6 results file not found.</p>}
        </section>
      </div>

      {info && (
        <section className="card">
          <h2>Decision rules</h2>
          <ul>
            {info.hybrid_policy.map((p) => (
              <li key={p.decision}><DecisionBadge decision={p.decision} />: {p.rule.replace("anomaly threshold", `anomaly threshold (${info.anomaly_detector.threshold.toFixed(3)})`)}</li>
            ))}
          </ul>
          <p className="small muted">
            {info.classifier.type} with {info.classifier.n_estimators} trees; {info.anomaly_detector.type} threshold
            from the {info.anomaly_detector.threshold_rule}. The models use {info.model_features.length} of the{" "}
            {info.input_features.length} input columns; excluded: {info.excluded_features.join(", ") || "none"}.
          </p>
        </section>
      )}
      {off.missing_files.length > 0 && (
        <p className="small muted">Missing result files: {off.missing_files.join(", ")}</p>
      )}
    </>
  );
}

function Tile({ label, value }: { label: React.ReactNode; value: string }) {
  return (
    <div className="tile">
      <div className="value">{value}</div>
      <div className="label">{label}</div>
    </div>
  );
}
