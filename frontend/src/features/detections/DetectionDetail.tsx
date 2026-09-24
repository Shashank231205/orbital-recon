/**
 * Detail card for a single detection.
 *
 * Shows everything the pipeline recorded about one target, including the
 * measurements an analyst would otherwise have to infer from the box: its
 * dimensions, its heading, and where it sits.
 */

import { ThreatBadge } from "@/components/Indicators";
import type { Detection, Scene } from "@/types/api";

interface Props {
  detection: Detection | null;
  scene: Scene | null;
  onClose: () => void;
}

/** Heading in degrees clockwise from north, from the box rotation in radians. */
function headingDegrees(angleRadians: number): number {
  const degrees = (angleRadians * 180) / Math.PI;
  return ((degrees % 360) + 360) % 360;
}

function compassPoint(degrees: number): string {
  const points = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return points[Math.round(degrees / 45) % 8] ?? "N";
}

export function DetectionDetail({ detection, scene, onClose }: Props) {
  if (!detection) {
    return null;
  }

  const heading = headingDegrees(detection.box.angle);
  const longEdge = Math.max(detection.box.width, detection.box.height);
  const shortEdge = Math.min(detection.box.width, detection.box.height);

  return (
    <div className="card detail">
      <div className="spread" style={{ marginBottom: 10 }}>
        <span className="row">
          <ThreatBadge level={detection.threat_level} />
          <strong>{detection.class_name}</strong>
        </span>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close">
          &times;
        </button>
      </div>

      <div className="detail-grid">
        <span className="dim">Confidence</span>
        <span className="mono">{(detection.confidence * 100).toFixed(1)}%</span>

        <span className="dim">Extent</span>
        <span className="mono">
          {longEdge.toFixed(0)} &times; {shortEdge.toFixed(0)} px
        </span>

        <span className="dim">Heading</span>
        <span className="mono">
          {heading.toFixed(0)}&deg; {compassPoint(heading)}
        </span>

        <span className="dim">Centre</span>
        <span className="mono">
          {detection.box.cx.toFixed(0)}, {detection.box.cy.toFixed(0)} px
        </span>

        {detection.centroid && (
          <>
            <span className="dim">Position</span>
            <span className="mono">
              {detection.centroid.latitude.toFixed(6)},{" "}
              {detection.centroid.longitude.toFixed(6)}
            </span>
          </>
        )}

        <span className="dim">Detection ID</span>
        <span className="mono">{detection.id}</span>
      </div>

      <p className="detail-note">
        {detection.centroid
          ? "Position derived from the scene affine transform, reprojected to WGS84."
          : scene?.is_georeferenced === false
            ? "Scene carries no coordinate reference, so this target is located in pixel space only."
            : ""}
      </p>
    </div>
  );
}
