import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HeldOutRow } from "../types";
import { AlertsPanel } from "./AlertsPanel";
import { DecisionBadge } from "./DecisionBadge";
import { HeldOutChart } from "./HeldOutChart";
import { ShapChart } from "./ShapChart";

afterEach(() => vi.unstubAllGlobals());

describe("DecisionBadge", () => {
  it("always shows a text label, not colour alone", () => {
    render(<DecisionBadge decision="known_attack" />);
    expect(screen.getByText("Known attack")).toBeInTheDocument();
  });
});

describe("HeldOutChart", () => {
  const row = (fam: string, fpr: number, c: number, h: number): HeldOutRow => ({
    model: "random_forest", held_out: fam, held_out_rows: 100, classifier_recall: c, detector_recall: 0,
    hybrid_recall: h, hybrid_recall_as_anomaly: h - c, classifier_fpr: 0.001, hybrid_fpr: fpr, target_fpr: fpr,
  });
  const rows = [row("DoS", 0.01, 0.006, 0.1), row("DoS", 0.05, 0.006, 0.54), row("Bot", 0.05, 0, 0.29)];

  it("shows the selected threshold and has a table view", () => {
    render(<HeldOutChart rows={rows} source="anomaly/hybrid_held_out_families.csv" />);
    expect(screen.getByText("54.0%")).toBeInTheDocument(); // defaults to the larger target FPR
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "0.01" } });
    expect(screen.getByText("10.0%")).toBeInTheDocument();
    expect(screen.queryByText("Bot")).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("Show table"));
    expect(screen.getByRole("table")).toBeInTheDocument();
  });
});

describe("ShapChart", () => {
  it("labels each feature with its value", () => {
    render(<ShapChart contributions={[{ feature: "Flow Duration", value: 12, shap: 0.3 }, { feature: "PSH Flag Count", value: 0, shap: -0.1 }]} />);
    expect(screen.getByText("Flow Duration")).toBeInTheDocument();
    expect(screen.getByText("Pushes towards attack")).toBeInTheDocument();
  });
});

describe("AlertsPanel", () => {
  it("lists alerts from the API and opens an explanation", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify({
      count: 1, note: "in memory",
      alerts: [{ prediction_id: "abc", created_at: "2026-10-11T08:00:00.000+00:00", source: "sample", decision: "anomaly",
        attack_probability: 0.1, anomaly_score: 0.7, missing_values: [] }],
    })))));
    const onExplain = vi.fn();
    render(<AlertsPanel onExplain={onExplain} selected={null} refreshKey={0} active={false} />);
    expect(await screen.findByText("Anomaly")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Explain"));
    expect(onExplain).toHaveBeenCalledWith("abc");
  });

  it("shows the API error when the server is down", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    render(<AlertsPanel onExplain={() => {}} selected={null} refreshKey={0} active={false} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the API");
  });
});
