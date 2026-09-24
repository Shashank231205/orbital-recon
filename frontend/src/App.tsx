/**
 * Application shell.
 *
 * Holds the active scene and job, and coordinates the panels around them.
 * Results are fetched once a job reports completion over its progress stream,
 * rather than polled, so the UI updates as soon as the pipeline finishes.
 */

import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { DetectionPanel } from "@/features/detections/DetectionPanel";
import { JobProgress } from "@/features/jobs/JobProgress";
import { SystemStatus } from "@/features/jobs/SystemStatus";
import { DetectionMap } from "@/features/map/DetectionMap";
import { PixelCanvas } from "@/features/map/PixelCanvas";
import { QueryPanel } from "@/features/query/QueryPanel";
import { UploadPanel } from "@/features/upload/UploadPanel";
import { useJobProgress } from "@/hooks/useJobProgress";
import type { Scene, UploadResponse } from "@/types/api";

export function App() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [jobId, setJobId] = useState<number | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const { event, isStreaming, streamError } = useJobProgress(jobId);
  const isComplete = event?.status === "completed";

  const { data: detections } = useQuery({
    queryKey: ["detections", jobId],
    queryFn: () => api.listDetections(jobId as number),
    enabled: jobId !== null && isComplete,
  });

  const { data: summary } = useQuery({
    queryKey: ["summary", jobId],
    queryFn: () => api.getSummary(jobId as number),
    enabled: jobId !== null && isComplete,
  });

  // Object URLs are retained by the browser until explicitly released.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleUploaded(result: UploadResponse, file: File) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));
    setScene(result.scene);
    setJobId(result.job.id);
    setSelectedId(null);
  }

  const items = useMemo(() => detections?.items ?? [], [detections]);
  const useMap = scene?.is_georeferenced === true;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <h1>Orbital Recon</h1>
          <span>satellite target detection</span>
        </div>
        <SystemStatus />
      </header>

      <div className="workspace">
        <aside className="panel">
          <UploadPanel onUploaded={handleUploaded} disabled={isStreaming} />
          <JobProgress event={event} isStreaming={isStreaming} streamError={streamError} />
          {scene && (
            <div className="card">
              <h2>Source</h2>
              <div className="spread">
                <span className="mono dim">dimensions</span>
                <span className="mono">
                  {scene.width} × {scene.height}
                </span>
              </div>
              <div className="spread">
                <span className="mono dim">modality</span>
                <span className="mono">{scene.modality.toUpperCase()}</span>
              </div>
              <div className="spread">
                <span className="mono dim">reference</span>
                <span className="mono">{scene.crs ?? "pixel space"}</span>
              </div>
            </div>
          )}
        </aside>

        <main className="stage">
          {useMap ? (
            <DetectionMap
              detections={items}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          ) : (
            <PixelCanvas
              imageUrl={previewUrl}
              detections={items}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
        </main>

        <aside className="panel right">
          <DetectionPanel
            summary={summary ?? null}
            detections={items}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
          <QueryPanel jobId={jobId} ready={isComplete === true} />
        </aside>
      </div>
    </div>
  );
}

