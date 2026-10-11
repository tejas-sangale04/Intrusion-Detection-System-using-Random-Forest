import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api";
import type { Explanation } from "../types";
import { DecisionBadge } from "./DecisionBadge";
import { ShapChart } from "./ShapChart";

export function ExplanationPanel({ predictionId, onClose }: { predictionId: string; onClose: () => void }) {
  const [exp, setExp] = useState<Explanation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    let live = true;
    setExp(null);
    setError(null);
    ref.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    api.explanation(predictionId, 10)
      .then((e) => live && setExp(e))
      .catch((e: ApiError) => live && setError(e.message));
    return () => { live = false; };
  }, [predictionId]);

  return (
    <section className="card" aria-label="Explanation" ref={ref}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Why this decision?</h2>
        <button onClick={onClose}>Close</button>
      </div>
      <p className="small muted">Prediction {predictionId}</p>
      {error && <p className="error">{error}</p>}
      {!exp && !error && <p className="muted">Computing SHAP values (about half a second)…</p>}
      {exp && (
        <>
          <p className="row">
            <DecisionBadge decision={exp.decision} />
            <span>Attack probability <strong>{exp.attack_probability.toFixed(3)}</strong></span>
            <span className="muted">average over training traffic {exp.base_value.toFixed(3)}</span>
          </p>
          <ShapChart contributions={exp.top_features} />
          <p style={{ marginTop: 12 }}>{exp.summary}</p>
          <p className="small muted">{exp.scope}</p>
        </>
      )}
    </section>
  );
}
