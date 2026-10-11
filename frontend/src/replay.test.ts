import { describe, expect, it } from "vitest";
import { activeReplays } from "./replay";
import type { Runtime } from "./types";

const runtime = (sources: Runtime["sources"]) => ({ sources } as Runtime);
const stats = (last_seen: string) => ({ flows: 10, benign: 8, anomaly: 1, known_attack: 1, last_seen });

describe("activeReplays", () => {
  it("lists only replay sources seen in the last 15 seconds", () => {
    const now = Date.parse("2026-10-11T10:00:20Z");
    const r = activeReplays(runtime({
      "replay:test": stats("2026-10-11T10:00:10Z"),
      "replay:old": stats("2026-10-11T09:59:00Z"),
      api: stats("2026-10-11T10:00:19Z"),
    }), now);
    expect(r.map((x) => x.name)).toEqual(["replay:test"]);
  });
});
