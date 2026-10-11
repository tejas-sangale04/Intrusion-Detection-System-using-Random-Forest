import type { Flow } from "./types";

/**
 * Parse a CIC-IDS2017-style CSV into flows.
 *
 * - Header names are trimmed (the original files have leading spaces), and a
 *   repeated name gets a ".1", ".2" ... suffix, as in the Python pipeline
 *   (CIC-IDS2017 has "Fwd Header Length" twice).
 * - Numbers become numbers. Empty cells, "NaN" and "Infinity" become null,
 *   because JSON has no Infinity; the API imputes null exactly as it treats
 *   Infinity, and lists those features in `missing_values`.
 * - Any other text is kept as-is, so the API rejects it with a clear error
 *   instead of the browser guessing a value.
 * - Columns named "Label" are kept so results can be compared with ground
 *   truth; the API ignores them.
 *
 * The files have no quoted fields, so a plain comma split is enough; a quoted
 * field is reported instead of being mis-split.
 */
export function parseFlowsCsv(text: string): { flows: Flow[]; labels: (string | null)[]; header: string[] } {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length < 2) throw new Error("The CSV needs a header row and at least one data row.");
  if (text.includes('"')) throw new Error("Quoted CSV fields are not supported; export the flows without quotes.");
  const header = dedupe(lines[0].replace(/^\uFEFF/, "").split(",").map((h) => h.trim()));
  const labelIdx = header.indexOf("Label");
  const flows: Flow[] = [];
  const labels: (string | null)[] = [];
  lines.slice(1).forEach((line, row) => {
    const cells = line.split(",");
    if (cells.length !== header.length) {
      throw new Error(`Row ${row + 1} has ${cells.length} values but the header has ${header.length}.`);
    }
    const flow: Record<string, unknown> = {};
    header.forEach((name, i) => {
      if (i === labelIdx) return;
      flow[name] = toValue(cells[i].trim());
    });
    flows.push(flow as Flow);
    labels.push(labelIdx >= 0 ? cells[labelIdx].trim() : null);
  });
  return { flows, labels, header };
}

function toValue(cell: string): number | null | string {
  if (cell === "" || /^[+-]?(nan|infinity|inf)$/i.test(cell)) return null;
  const n = Number(cell);
  return Number.isFinite(n) ? n : cell;
}

function dedupe(names: string[]): string[] {
  const seen = new Map<string, number>();
  return names.map((name) => {
    if (!seen.has(name)) {
      seen.set(name, 0);
      return name;
    }
    let n = seen.get(name)!;
    let unique: string;
    do {
      n += 1;
      unique = `${name}.${n}`;
    } while (seen.has(unique));
    seen.set(name, n);
    seen.set(unique, 0);
    return unique;
  });
}
