/**
 * Overlay legend for the detection view.
 *
 * Without this the colours carry no meaning: a reader sees red and blue boxes
 * and cannot tell whether the distinction is class, confidence or priority.
 */

import { useMemo } from "react";

import { threatColor } from "@/components/Indicators";
import type { Detection, ThreatLevel } from "@/types/api";

const THREAT_DESCRIPTIONS: Record<ThreatLevel, string> = {
  high: "Aircraft, helicopters, vessels",
  medium: "Vehicles, storage tanks",
  low: "Infrastructure and terrain features",
  unknown: "Unrecognised class",
};

interface Props {
  detections: Detection[];
  showLabels: boolean;
  onToggleLabels: () => void;
}

export function Legend({ detections, showLabels, onToggleLabels }: Props) {
  const rows = useMemo(() => {
    const byThreat = new Map<ThreatLevel, { count: number; classes: Set<string> }>();

    for (const detection of detections) {
      const entry = byThreat.get(detection.threat_level) ?? {
        count: 0,
        classes: new Set<string>(),
      };
      entry.count += 1;
      entry.classes.add(detection.class_name);
      byThreat.set(detection.threat_level, entry);
    }

    const order: ThreatLevel[] = ["high", "medium", "low", "unknown"];
    return order
      .filter((level) => byThreat.has(level))
      .map((level) => ({
        level,
        count: byThreat.get(level)?.count ?? 0,
        classes: [...(byThreat.get(level)?.classes ?? [])].sort(),
      }));
  }, [detections]);

  if (detections.length === 0) {
    return null;
  }

  return (
    <div className="legend">
      <div className="legend-head">
        <span>Priority</span>
        <button type="button" className="legend-toggle" onClick={onToggleLabels}>
          {showLabels ? "Hide labels" : "Show labels"}
        </button>
      </div>

      {rows.map((row) => (
        <div key={row.level} className="legend-row" title={THREAT_DESCRIPTIONS[row.level]}>
          <span className="legend-swatch" style={{ background: threatColor(row.level) }} />
          <span className="legend-label">{row.level}</span>
          <span className="legend-count">{row.count}</span>
          <span className="legend-classes">{row.classes.join(", ")}</span>
        </div>
      ))}
    </div>
  );
}
