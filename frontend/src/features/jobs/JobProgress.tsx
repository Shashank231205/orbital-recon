/**
 * Live view of an analysis run.
 *
 * Shows both an overall bar and the individual pipeline stages, so a long
 * inference pass reads as progress through known steps rather than a bar that
 * appears to have stalled.
 */

import { Spinner } from "@/components/Indicators";
import { STAGE_LABELS, STAGE_ORDER } from "@/hooks/useJobProgress";
import type { ProgressEvent } from "@/types/api";

interface Props {
  event: ProgressEvent | null;
  isStreaming: boolean;
  streamError: string | null;
}

export function JobProgress({ event, isStreaming, streamError }: Props) {
  if (!event) {
    return (
      <div className="card">
        <h2>Analysis</h2>
        <p className="empty" style={{ padding: "12px 0" }}>
          Upload a scene to begin.
        </p>
      </div>
    );
  }

  const failed = event.status === "failed";
  const done = event.status === "completed";
  const currentIndex = STAGE_ORDER.indexOf(event.stage);

  return (
    <div className="card">
      <div className="spread" style={{ marginBottom: 12 }}>
        <h2 style={{ marginBottom: 0 }}>Analysis</h2>
        <span className="mono dim">job {event.job_id}</span>
      </div>

      <div className="spread" style={{ marginBottom: 8 }}>
        <span className="row">
          {isStreaming && !done && !failed && <Spinner />}
          <span style={{ fontSize: 13 }}>
            {failed ? "Failed" : STAGE_LABELS[event.stage]}
          </span>
        </span>
        <span className="mono muted">{Math.round(event.progress * 100)}%</span>
      </div>

      <div
        className="progress-track"
        role="progressbar"
        aria-valuenow={Math.round(event.progress * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label="Analysis progress"
      >
        <div
          className={`progress-fill${failed ? " failed" : done ? " done" : ""}`}
          style={{ width: `${Math.max(event.progress * 100, 2)}%` }}
        />
      </div>

      {event.message && (
        <p className="mono dim" style={{ margin: "8px 0 0" }}>
          {event.message}
        </p>
      )}

      <div className="stage-list">
        {STAGE_ORDER.filter((stage) => stage !== "queued").map((stage) => {
          const index = STAGE_ORDER.indexOf(stage);
          const isActive = index === currentIndex && !done && !failed;
          const isComplete = index < currentIndex || done;

          return (
            <div
              key={stage}
              className={`stage-row${isActive ? " active" : isComplete ? " complete" : ""}`}
            >
              <span className="stage-marker">
                {isComplete ? "✓" : isActive ? "▸" : "·"}
              </span>
              {STAGE_LABELS[stage]}
            </div>
          );
        })}
      </div>

      {event.error && <p className="error-text">{event.error}</p>}
      {streamError && !done && !failed && <p className="error-text">{streamError}</p>}

      {done && (
        <p className="muted" style={{ margin: "10px 0 0", fontSize: 13 }}>
          {event.detection_count} asset{event.detection_count === 1 ? "" : "s"} detected.
        </p>
      )}
    </div>
  );
}
