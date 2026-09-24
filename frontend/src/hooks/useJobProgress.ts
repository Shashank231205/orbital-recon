/**
 * Subscribes to a job's progress stream.
 *
 * The connection closes as soon as the job reaches a terminal state. Without
 * that, `EventSource` would reconnect indefinitely to a finished job, since it
 * retries automatically whenever the server closes the stream.
 */

import { useEffect, useRef, useState } from "react";

import { progressStreamUrl } from "@/api/client";
import type { ProgressEvent } from "@/types/api";

export interface JobProgressState {
  event: ProgressEvent | null;
  isStreaming: boolean;
  streamError: string | null;
}

export function useJobProgress(jobId: number | null): JobProgressState {
  const [event, setEvent] = useState<ProgressEvent | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (jobId === null) {
      setEvent(null);
      setIsStreaming(false);
      setStreamError(null);
      return;
    }

    const source = new EventSource(progressStreamUrl(jobId));
    sourceRef.current = source;
    setIsStreaming(true);
    setStreamError(null);

    source.onmessage = (message) => {
      const update = JSON.parse(message.data) as ProgressEvent;
      setEvent(update);

      if (update.status === "completed" || update.status === "failed") {
        source.close();
        setIsStreaming(false);
      }
    };

    source.onerror = () => {
      // Fires both on a genuine failure and on the normal close after a
      // terminal update, so it only counts as an error while still streaming.
      if (source.readyState === EventSource.CLOSED) {
        setIsStreaming(false);
        return;
      }
      setStreamError("Lost connection to the analysis stream");
    };

    return () => {
      source.close();
      sourceRef.current = null;
      setIsStreaming(false);
    };
  }, [jobId]);

  return { event, isStreaming, streamError };
}

/** Human-readable label for each pipeline stage. */
export const STAGE_LABELS: Record<ProgressEvent["stage"], string> = {
  queued: "Queued",
  loading: "Loading scene",
  preprocessing: "Enhancing imagery",
  tiling: "Tiling scene",
  inference: "Detecting targets",
  merging: "Merging tiles",
  georeferencing: "Locating targets",
  persisting: "Saving results",
  done: "Complete",
};

export const STAGE_ORDER: ProgressEvent["stage"][] = [
  "queued",
  "loading",
  "preprocessing",
  "tiling",
  "inference",
  "merging",
  "georeferencing",
  "persisting",
  "done",
];
