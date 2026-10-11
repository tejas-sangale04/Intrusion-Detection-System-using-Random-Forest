import { describe, expect, it } from "vitest";
import { parseFlowsCsv } from "./csv";

describe("parseFlowsCsv", () => {
  it("trims headers, keeps labels apart and maps Infinity/NaN/empty to null", () => {
    const text = " Destination Port, Flow Bytes/s, Flow Packets/s, Label\r\n80,Infinity,NaN,BENIGN\n443,12.5,,DDoS\n";
    const { flows, labels, header } = parseFlowsCsv(text);
    expect(header).toEqual(["Destination Port", "Flow Bytes/s", "Flow Packets/s", "Label"]);
    expect(flows).toEqual([
      { "Destination Port": 80, "Flow Bytes/s": null, "Flow Packets/s": null },
      { "Destination Port": 443, "Flow Bytes/s": 12.5, "Flow Packets/s": null },
    ]);
    expect(labels).toEqual(["BENIGN", "DDoS"]);
  });

  it("renames repeated headers like the Python pipeline", () => {
    const { header } = parseFlowsCsv(" Fwd Header Length, Flow Duration, Fwd Header Length\n1,2,3\n");
    expect(header).toEqual(["Fwd Header Length", "Flow Duration", "Fwd Header Length.1"]);
  });

  it("keeps unparseable text so the API can reject it", () => {
    const { flows } = parseFlowsCsv("a,b\n1,abc\n");
    expect(flows[0]).toEqual({ a: 1, b: "abc" });
  });

  it("rejects ragged rows, quoted fields and empty files", () => {
    expect(() => parseFlowsCsv("a,b\n1\n")).toThrow(/Row 1 has 1 values/);
    expect(() => parseFlowsCsv('a,b\n"1",2\n')).toThrow(/Quoted/);
    expect(() => parseFlowsCsv("a,b\n")).toThrow(/at least one data row/);
  });
});
