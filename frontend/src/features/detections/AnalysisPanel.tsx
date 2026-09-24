/**
 * Derived statistics over a job's detections.
 *
 * The counts alone do not say whether a run is trustworthy. These figures
 * answer the questions an analyst asks next: how confident is the model
 * overall, are the targets a consistent size, and how tightly are they packed.
 * All are computed client-side from detections already fetched, so they cost no
 * extra request.
 */

import { useMemo } from "react";

import { InfoHint } from "@/components/InfoHint";
import { threatColor } from "@/components/Indicators";
import type { Detection, Job } from "@/types/api";

interface Props {
  detections: Detection[];
  job: Job | null;
}

const BUCKET_EDGES = [0.9, 0.8, 0.7, 0.6, 0.5, 0.0];

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
    : (sorted[middle] ?? 0);
}

export function AnalysisPanel({ detections, job }: Props) {
  const stats = useMemo(() => {
    if (detections.length === 0) {
      return null;
    }

    const confidences = detections.map((d) => d.confidence);
    const longEdges = detections.map((d) => Math.max(d.box.width, d.box.height));
    const aspects = detections.map(
      (d) => Math.max(d.box.width, d.box.height) / Math.max(Math.min(d.box.width, d.box.height), 1),
    );

    const buckets = BUCKET_EDGES.map((edge, index) => {
      const upper = index === 0 ? 1.01 : BUCKET_EDGES[index - 1] ?? 1.01;
      const count = confidences.filter((c) => c >= edge && c < upper).length;
      return {
        label: index === 0 ? "90%+" : `${(edge * 100).toFixed(0)}-${(upper * 100).toFixed(0)}%`,
        count,
      };
    });

    // A heading concentrated in one direction usually means moored or parked
    // assets rather than moving ones.
    const headings = detections.map((d) => {
      const degrees = ((d.box.angle * 180) / Math.PI) % 180;
      return degrees < 0 ? degrees + 180 : degrees;
    });
    const headingSpread = Math.max(...headings) - Math.min(...headings);

    return {
      buckets,
      maxBucket: Math.max(...buckets.map((b) => b.count), 1),
      medianConfidence: median(confidences),
      lowConfidenceShare: confidences.filter((c) => c < 0.5).length / confidences.length,
      medianLongEdge: median(longEdges),
      medianAspect: median(aspects),
      headingSpread,
      throughput:
        job?.duration_ms && job.duration_ms > 0
          ? (detections.length / job.duration_ms) * 1000
          : null,
    };
  }, [detections, job]);

  if (!stats) {
    return null;
  }

  return (
    <div className="card">
      <h2>
        Analysis
        <InfoHint>
          Derived from this run only. Every figure is computed from the detections
          listed below, not from a model estimate.
        </InfoHint>
      </h2>

      <div className="stat-row">
        <span className="dim">
          Median confidence
          <InfoHint>
            The middle value, which is less distorted by a handful of very weak or
            very strong detections than the mean.
          </InfoHint>
        </span>
        <span className="mono">{(stats.medianConfidence * 100).toFixed(1)}%</span>
      </div>

      <div className="stat-row">
        <span className="dim">
          Below 50%
          <InfoHint>
            Share of detections the model was unsure about. A high proportion
            suggests lowering the threshold admitted noise, or that the imagery is
            unlike the model's training data.
          </InfoHint>
        </span>
        <span className="mono">{(stats.lowConfidenceShare * 100).toFixed(0)}%</span>
      </div>

      <div className="stat-row">
        <span className="dim">
          Median extent
          <InfoHint>
            Length of the longer box edge in pixels. Consistent sizes across a
            scene indicate the model is finding one kind of object rather than
            fragmenting larger ones.
          </InfoHint>
        </span>
        <span className="mono">{stats.medianLongEdge.toFixed(0)} px</span>
      </div>

      <div className="stat-row">
        <span className="dim">
          Median elongation
          <InfoHint>
            Long edge divided by short edge. Vessels and aircraft are elongated;
            a value near 1 means the boxes are roughly square, which is unusual
            for those classes.
          </InfoHint>
        </span>
        <span className="mono">{stats.medianAspect.toFixed(2)}&times;</span>
      </div>

      <div className="stat-row">
        <span className="dim">
          Heading spread
          <InfoHint>
            Range of box orientations. A narrow spread means targets share an
            alignment, which is typical of moored vessels or parked aircraft.
          </InfoHint>
        </span>
        <span className="mono">{stats.headingSpread.toFixed(0)}&deg;</span>
      </div>

      {stats.throughput !== null && (
        <div className="stat-row">
          <span className="dim">
            Throughput
            <InfoHint>
              Detections produced per second of wall-clock analysis, including
              tiling and post-processing, not just model inference.
            </InfoHint>
          </span>
          <span className="mono">{stats.throughput.toFixed(0)}/s</span>
        </div>
      )}

      <h3 className="subhead">
        Confidence distribution
        <InfoHint>
          How detections spread across confidence bands. A distribution weighted
          toward the top bands is a stronger result than the same count spread
          evenly.
        </InfoHint>
      </h3>

      {stats.buckets.map((bucket) => (
        <div key={bucket.label} className="histogram-row">
          <span className="mono dim histogram-label">{bucket.label}</span>
          <div className="bar-track histogram-track">
            <div
              className="bar-fill"
              style={{
                width: `${(bucket.count / stats.maxBucket) * 100}%`,
                background: threatColor(bucket.count > 0 ? "low" : "unknown"),
              }}
            />
          </div>
          <span className="mono histogram-count">{bucket.count}</span>
        </div>
      ))}
    </div>
  );
}
