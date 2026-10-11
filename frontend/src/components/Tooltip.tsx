import { useState, type ReactNode } from "react";

export interface TipState { x: number; y: number; content: ReactNode }

/** One floating tooltip per chart, positioned at the pointer. */
export function useTooltip() {
  const [tip, setTip] = useState<TipState | null>(null);
  const bind = (content: ReactNode) => ({
    onMouseMove: (e: React.MouseEvent) => setTip({ x: e.clientX + 12, y: e.clientY + 12, content }),
    onMouseLeave: () => setTip(null),
  });
  const node = tip ? (
    <div className="tooltip" role="tooltip" style={{ left: tip.x, top: tip.y }}>
      {tip.content}
    </div>
  ) : null;
  return { bind, node };
}
