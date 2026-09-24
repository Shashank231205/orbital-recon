/** Small presentational primitives shared across panels. */

import type { ThreatLevel } from "@/types/api";

export function ThreatBadge({ level }: { level: ThreatLevel }) {
  return <span className={`badge ${level}`}>{level}</span>;
}

export function StatusDot({ state }: { state: "ok" | "bad" | "idle" }) {
  return <span className={`dot ${state}`} />;
}

export function Spinner() {
  return <span className="spinner" role="status" aria-label="Working" />;
}

/** Colour used for a threat level in charts and overlays. */
export function threatColor(level: ThreatLevel): string {
  switch (level) {
    case "high":
      return "#ff5c5c";
    case "medium":
      return "#ffb224";
    case "low":
      return "#4bb3fd";
    default:
      return "#8b98a8";
  }
}

export function formatDuration(ms: number | null): string {
  if (ms === null) {
    return "—";
  }
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(1)} ${units[unit]}`;
}
