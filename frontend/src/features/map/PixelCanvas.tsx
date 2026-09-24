/**
 * Pixel-space view of detections.
 *
 * Scenes without georeferencing cannot be placed on a map, so detections are
 * drawn over the source image using their oriented box corners. The canvas is
 * scaled to fit while preserving aspect ratio, and box coordinates are scaled
 * by the same factor so overlays stay aligned at any display size.
 *
 * Labels are drawn only when a detection is large enough on screen to carry
 * one, or when it is selected. A dense harbour scene holds hundreds of
 * overlapping vessels, and labelling every one turns the image into a wall of
 * text that hides the imagery underneath.
 */

import { useEffect, useRef, useState } from "react";

import { threatColor } from "@/components/Indicators";
import type { Detection } from "@/types/api";

interface Props {
  imageUrl: string | null;
  detections: Detection[];
  selectedId: number | null;
  onSelect: (id: number) => void;
  showLabels: boolean;
}

const MAX_EDGE = 1400;

/** Minimum on-screen box width, in pixels, before a label is drawn. */
const LABEL_MIN_WIDTH = 46;

const LABEL_FONT = "600 11px ui-sans-serif, system-ui, sans-serif";

interface Placed {
  detection: Detection;
  points: [number, number][];
}

function drawLabel(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  colour: string,
): void {
  context.font = LABEL_FONT;
  const width = context.measureText(text).width;
  const paddingX = 4;
  const height = 15;

  // Keep the label inside the canvas when the box sits against an edge.
  const left = Math.max(1, Math.min(x, context.canvas.width - width - paddingX * 2 - 1));
  const top = y - height < 1 ? y + 2 : y - height;

  context.fillStyle = "rgba(11, 15, 20, 0.82)";
  context.fillRect(left, top, width + paddingX * 2, height);

  context.fillStyle = colour;
  context.fillRect(left, top, 2, height);

  context.fillStyle = "#e6edf3";
  context.textBaseline = "middle";
  context.fillText(text, left + paddingX + 2, top + height / 2);
}

export function PixelCanvas({
  imageUrl,
  detections,
  selectedId,
  onSelect,
  showLabels,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const placedRef = useRef<Placed[]>([]);
  const [scale, setScale] = useState(1);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    if (!imageUrl) {
      setIsLoaded(false);
      return;
    }

    const image = new Image();
    image.src = imageUrl;
    image.onload = () => {
      imageRef.current = image;
      setScale(Math.min(1, MAX_EDGE / Math.max(image.width, image.height)));
      setIsLoaded(true);
    };

    return () => {
      image.onload = null;
    };
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !isLoaded) {
      return;
    }

    canvas.width = image.width * scale;
    canvas.height = image.height * scale;

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.drawImage(image, 0, 0, canvas.width, canvas.height);

    const placed: Placed[] = [];

    // Outlines first, so no box is drawn over a label.
    for (const detection of detections) {
      if (detection.box.corners.length < 4) {
        continue;
      }

      const points = detection.box.corners.map(
        ([x, y]) => [x * scale, y * scale] as [number, number],
      );
      placed.push({ detection, points });

      const isSelected = detection.id === selectedId;
      const colour = threatColor(detection.threat_level);

      context.beginPath();
      points.forEach(([x, y], index) => {
        if (index === 0) {
          context.moveTo(x, y);
        } else {
          context.lineTo(x, y);
        }
      });
      context.closePath();

      context.strokeStyle = colour;
      context.lineWidth = isSelected ? 3 : 1.5;
      context.stroke();

      if (isSelected) {
        context.fillStyle = `${colour}55`;
        context.fill();
      }
    }

    placedRef.current = placed;

    if (!showLabels) {
      return;
    }

    for (const { detection, points } of placed) {
      const xs = points.map(([x]) => x);
      const ys = points.map(([, y]) => y);
      const boxWidth = Math.max(...xs) - Math.min(...xs);
      const isSelected = detection.id === selectedId;

      if (!isSelected && boxWidth < LABEL_MIN_WIDTH) {
        continue;
      }

      drawLabel(
        context,
        `${detection.class_name} ${(detection.confidence * 100).toFixed(0)}%`,
        Math.min(...xs),
        Math.min(...ys),
        threatColor(detection.threat_level),
      );
    }
  }, [detections, isLoaded, scale, selectedId, showLabels]);

  function handleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    // The canvas is laid out with CSS max-width, so its rendered size can
    // differ from its backing size; both factors are needed to recover the
    // canvas coordinate that was clicked.
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (canvas.width / rect.width);
    const y = (event.clientY - rect.top) * (canvas.height / rect.height);

    // Smallest match wins: in a dense scene a click often lands inside several
    // overlapping boxes, and the tightest one is what the eye is pointing at.
    let best: Placed | null = null;
    let bestArea = Infinity;

    for (const candidate of placedRef.current) {
      const xs = candidate.points.map(([px]) => px);
      const ys = candidate.points.map(([, py]) => py);
      const inside =
        x >= Math.min(...xs) &&
        x <= Math.max(...xs) &&
        y >= Math.min(...ys) &&
        y <= Math.max(...ys);

      if (!inside) {
        continue;
      }

      const area = (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
      if (area < bestArea) {
        best = candidate;
        bestArea = area;
      }
    }

    if (best) {
      onSelect(best.detection.id);
    }
  }

  if (!imageUrl) {
    return (
      <div className="canvas-stage">
        <p className="empty">Upload a scene to see detections.</p>
      </div>
    );
  }

  return (
    <div className="canvas-stage">
      <canvas ref={canvasRef} onClick={handleClick} />
    </div>
  );
}
