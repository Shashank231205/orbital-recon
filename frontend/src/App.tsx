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
import { InfoHint } from "@/components/InfoHint";
import { AnalysisPanel } from "@/features/detections/AnalysisPanel";
import { DetectionDetail } from "@/features/detections/DetectionDetail";
import { DetectionPanel } from "@/features/detections/DetectionPanel";
import { JobProgress } from "@/features/jobs/JobProgress";
import { SystemStatus } from "@/features/jobs/SystemStatus";
import { DetectionMap } from "@/features/map/DetectionMap";
import { Legend } from "@/features/map/Legend";
import { PixelCanvas } from "@/features/map/PixelCanvas";
import { QueryPanel } from "@/features/query/QueryPanel";
import { UploadPanel } from "@/features/upload/UploadPanel";
import { useJobProgress } from "@/hooks/useJobProgress";
import type { Scene, UploadResponse } from "@/types/api";

const MODALITY_NOTES: Record<string, string> = {
  eo: "Electro-optical: visible light. Contrast is equalised locally before detection.",
  ir: "Infrared: thermal. The wide sensor range is rescaled from its 2nd to 98th percentile, then locally equalised.",
  sar: "Synthetic-aperture radar: the Lee filter suppresses speckle while preserving the hard edges of man-made structures.",
};

export function App() {
  const [scene, setScene] = useState<Scene | null>(null);
  const [jobId, setJobId] = useState<number | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState(true);

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

  const { data: job } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId as number),
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
  const selected = useMemo(
    () => items.find((item) => item.id === selectedId) ?? null,
    [items, selectedId],
  );
  const useMapView = scene?.is_georeferenced === true;

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
                  {scene.width} &times; {scene.height}
                </span>
              </div>
              <div className="spread">
                <span className="mono dim">
                  modality
                  <InfoHint>
                    {MODALITY_NOTES[scene.modality] ?? "Sensor type of this scene."}
                  </InfoHint>
                </span>
                <span className="mono">{scene.modality.toUpperCase()}</span>
              </div>
              <div className="spread">
                <span className="mono dim">
                  reference
                  <InfoHint>
                    A coordinate reference lets every detection be placed on a map.
                    Without one, targets are reported in pixel space only.
                  </InfoHint>
                </span>
                <span className="mono">{scene.crs ?? "pixel space"}</span>
              </div>
            </div>
          )}

          {selected && (
            <DetectionDetail
              detection={selected}
              scene={scene}
              onClose={() => setSelectedId(null)}
            />
          )}
        </aside>

        <main className="stage">
          {useMapView ? (
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
              showLabels={showLabels}
            />
          )}
          <Legend
            detections={items}
            showLabels={showLabels}
            onToggleLabels={() => setShowLabels((shown) => !shown)}
          />
        </main>

        <aside className="panel right">
          <DetectionPanel
            summary={summary ?? null}
            detections={items}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
          <AnalysisPanel detections={items} job={job ?? null} />
          <QueryPanel jobId={jobId} ready={isComplete === true} />
        </aside>
      </div>
    </div>
  );
}
