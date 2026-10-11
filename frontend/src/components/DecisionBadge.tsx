import type { Decision } from "../types";

const STYLE: Record<Decision, { label: string; icon: string; color: string }> = {
  benign: { label: "Benign", icon: "✓", color: "var(--status-good)" },
  anomaly: { label: "Anomaly", icon: "?", color: "var(--status-serious)" },
  known_attack: { label: "Known attack", icon: "!", color: "var(--status-critical)" },
};

export const DECISION_HELP: Record<Decision, string> = {
  benign: "Classifier below 0.5 and anomaly score below its threshold.",
  anomaly: "Classifier below 0.5, but unusual compared with benign training traffic. Needs review; not confirmed malicious.",
  known_attack: "Classifier probability at or above 0.5: resembles an attack type seen in training.",
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  const s = STYLE[decision];
  return (
    <span className="badge" title={DECISION_HELP[decision]}>
      <span className="dot" style={{ background: s.color }} aria-hidden />
      <span className="icon" aria-hidden>{s.icon}</span>
      {s.label}
    </span>
  );
}
