/** Detection results: aggregate summary plus a filterable list. */

import { useMemo, useState } from "react";

import { InfoHint } from "@/components/InfoHint";
import { ThreatBadge, threatColor } from "@/components/Indicators";
import type { Detection, JobSummary, ThreatLevel } from "@/types/api";

interface Props {
  summary: JobSummary | null;
  detections: Detection[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

const THREAT_FILTERS: (ThreatLevel | "all")[] = ["all", "high", "medium", "low"];

export function DetectionPanel({ summary, detections, selectedId, onSelect }: Props) {
  const [filter, setFilter] = useState<ThreatLevel | "all">("all");

  const visible = useMemo(
    () =>
      filter === "all"
        ? detections
        : detections.filter((detection) => detection.threat_level === filter),
    [detections, filter],
  );

  const maxCount = summary?.by_class.reduce((max, entry) => Math.max(max, entry.count), 0) ?? 0;

  return (
    <>
      <div className="card">
        <h2>
          Findings
          <InfoHint>
            Counts after overlapping tile detections were merged. Priority comes
            from the asset class, not from the model: a moderately confident
            aircraft outranks a highly confident swimming pool.
          </InfoHint>
        </h2>

        {!summary || summary.total_detections === 0 ? (
          <p className="empty" style={{ padding: "12px 0" }}>
            No detections yet.
          </p>
        ) : (
          <>
            <div className="spread" style={{ marginBottom: 12 }}>
              <span style={{ fontSize: 22, fontWeight: 600 }}>
                {summary.total_detections}
              </span>
              <span className="mono dim">
                mean {(summary.mean_confidence * 100).toFixed(1)}%
                <InfoHint>
                  Average confidence across every detection in this run.
                </InfoHint>
              </span>
            </div>

            {summary.by_class.map((entry) => (
              <div key={`${entry.label}-${entry.threat_level}`} className="class-row">
                <span className="row">
                  <ThreatBadge level={entry.threat_level} />
                  {entry.display_name}
                </span>
                <span className="mono muted">{entry.count}</span>
                <div className="bar-track">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${maxCount > 0 ? (entry.count / maxCount) * 100 : 0}%`,
                      background: threatColor(entry.threat_level),
                    }}
                  />
                </div>
              </div>
            ))}
          </>
        )}
      </div>

      {detections.length > 0 && (
        <div className="card">
          <div className="spread" style={{ marginBottom: 10 }}>
            <h2 style={{ marginBottom: 0 }}>
              Targets
              <InfoHint>
                Ordered by confidence. Select one to see its dimensions, heading
                and position, and to highlight it in the scene.
              </InfoHint>
            </h2>
            <span className="mono dim">{visible.length}</span>
          </div>

          <div className="row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
            {THREAT_FILTERS.map((level) => (
              <button
                key={level}
                type="button"
                onClick={() => setFilter(level)}
                className={filter === level ? "primary" : ""}
                style={{ padding: "4px 10px", fontSize: 12 }}
              >
                {level}
              </button>
            ))}
          </div>

          <div className="detection-list">
            {visible.map((detection) => (
              <button
                key={detection.id}
                type="button"
                className={`detection-item${detection.id === selectedId ? " selected" : ""}`}
                onClick={() => onSelect(detection.id)}
              >
                <span className="spread">
                  <span className="row">
                    <ThreatBadge level={detection.threat_level} />
                    {detection.class_name}
                  </span>
                  <span className="mono muted">
                    {(detection.confidence * 100).toFixed(0)}%
                  </span>
                </span>
                <span className="mono dim">
                  {detection.centroid
                    ? `${detection.centroid.latitude.toFixed(5)}, ${detection.centroid.longitude.toFixed(5)}`
                    : `pixel ${detection.box.cx.toFixed(0)}, ${detection.box.cy.toFixed(0)}`}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
