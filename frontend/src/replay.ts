import type { Runtime } from "./types";

export const REPLAY_PREFIX = "replay:";
const ACTIVE_WITHIN_MS = 15_000;

/** Replay sources that sent flows recently, newest first. */
export function activeReplays(runtime: Runtime | undefined, now: number = Date.now()) {
  if (!runtime) return [];
  return Object.entries(runtime.sources)
    .filter(([name, s]) => name.startsWith(REPLAY_PREFIX) && now - Date.parse(s.last_seen) < ACTIVE_WITHIN_MS)
    .map(([name, s]) => ({ name, ...s }))
    .sort((a, b) => Date.parse(b.last_seen) - Date.parse(a.last_seen));
}
