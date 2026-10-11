import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, MAX_BATCH, scoreAll } from "./api";

const ok = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

afterEach(() => vi.unstubAllGlobals());

describe("scoreAll", () => {
  it("splits flows into API-sized batches and keeps their order", async () => {
    const sizes: number[] = [];
    vi.stubGlobal("fetch", vi.fn((_url: string, init: RequestInit) => {
      const n = JSON.parse(String(init.body)).flows.length;
      const start = sizes.reduce((a, b) => a + b, 0);
      sizes.push(n);
      return ok({ count: n, decision_counts: {}, predictions: Array.from({ length: n }, (_, i) => ({ prediction_id: String(start + i) })) });
    }));
    const preds = await scoreAll(Array.from({ length: MAX_BATCH + 5 }, () => ({ x: 1 })), "test");
    expect(sizes).toEqual([MAX_BATCH, 5]);
    expect(preds.map((p) => p.prediction_id).slice(-2)).toEqual([String(MAX_BATCH + 3), String(MAX_BATCH + 4)]);
  });

  it("reports per-flow errors with their row in the whole file", async () => {
    let call = 0;
    vi.stubGlobal("fetch", vi.fn(() => {
      call += 1;
      if (call === 1) return ok({ count: MAX_BATCH, decision_counts: {}, predictions: [] });
      return Promise.resolve(new Response(JSON.stringify({ detail: "Invalid flow features", errors: [{ flow: 2, error: "Missing" }] }), { status: 422 }));
    }));
    const err = await scoreAll(Array.from({ length: MAX_BATCH + 5 }, () => ({ x: 1 })), "test").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.details[0].flow).toBe(MAX_BATCH + 2);
  });

  it("explains an unreachable API", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    await expect(scoreAll([{ x: 1 }], "test")).rejects.toThrow(/Cannot reach the API/);
  });
});
