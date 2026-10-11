// Mirrors backend/schemas.py and the /metrics and /model/info payloads.

export type Decision = "benign" | "anomaly" | "known_attack";
export type FeatureValue = number | null;
export type Flow = Record<string, FeatureValue>;

export interface Prediction {
  prediction_id: string;
  created_at: string;
  source: string;
  decision: Decision;
  attack_probability: number;
  anomaly_score: number;
  classifier_threshold: number;
  anomaly_threshold: number;
  missing_values: string[];
  explanation_url: string;
}

export interface BatchResult {
  count: number;
  decision_counts: Record<Decision, number>;
  predictions: Prediction[];
}

export interface Alert {
  prediction_id: string;
  created_at: string;
  source: string;
  decision: Decision;
  attack_probability: number;
  anomaly_score: number;
  missing_values: string[];
}

export interface AlertsResponse {
  count: number;
  alerts: Alert[];
  note: string;
}

export interface Contribution {
  feature: string;
  value: number;
  shap: number;
}

export interface Explanation {
  prediction_id: string;
  decision: Decision;
  attack_probability: number;
  base_value: number;
  top_features: Contribution[];
  summary: string;
  scope: string;
}

export interface Health {
  status: string;
  version: string;
  mode: string;
  models_loaded: boolean;
}

export interface ModelInfo {
  mode: string;
  classifier: { type: string; n_estimators: number; threshold: number };
  anomaly_detector: { type: string; threshold: number; threshold_rule: string };
  hybrid_policy: { rule: string; decision: Decision }[];
  input_features: string[];
  model_features: string[];
  excluded_features: string[];
}

export interface Runtime {
  uptime_seconds: number;
  prediction_requests: number;
  flows_scored: number;
  decision_counts: Record<Decision, number>;
  mean_scoring_ms_per_flow: number | null;
  predictions_in_memory: number;
  note: string;
}

export interface Sourced<T> {
  source: string;
  data: T;
}

export interface HeldOutRow {
  model: string;
  held_out: string;
  held_out_rows: number;
  classifier_recall: number;
  detector_recall: number;
  hybrid_recall: number;
  hybrid_recall_as_anomaly: number;
  classifier_fpr: number;
  hybrid_fpr: number;
  target_fpr: number;
}

export interface Metrics {
  offline_evaluation: {
    hybrid_held_out_families: Sourced<HeldOutRow[]> | null;
    global_feature_importance: Sourced<Record<string, string | number>[]> | null;
    classifier_baselines: Sourced<{ random_forest: { test: Record<string, number> } | null }> | null;
    missing_files: string[];
  };
  runtime: Runtime;
}
