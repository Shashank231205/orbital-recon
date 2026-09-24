/**
 * Pixel-space view of detections.
 *
 * Scenes without georeferencing cannot be placed on a map, so detections are
 * drawn over the source image using their oriented box corners. The canvas is
 * scaled to fit while preserving aspect ratio, and box coordinates are scaled
 * by the same factor so overlays stay aligned at any display size.
 */

import { useEffect, useRef, useState } from "react";

import { threatColor } from "@/components/Indicators";
import type { Detection } from "@/types/api";

interface Props {
  imageUrl: string | null;
  detections: Detection[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

const MAX_EDGE = 1400;

export function PixelCanvas({ imageUrl, detections, selectedId, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
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

    for (const detection of detections) {
      const corners = detection.box.corners;
      if (corners.length < 4) {
        continue;
      }

      const isSelected = detection.id === selectedId;
      context.beginPath();
      corners.forEach(([x, y], index) => {
        const px = x * scale;
        const py = y * scale;
        if (index === 0) {
          context.moveTo(px, py);
        } else {
          context.lineTo(px, py);
        }
      });
      context.closePath();

      context.strokeStyle = threatColor(detection.threat_level);
      context.lineWidth = isSelected ? 3 : 1.5;
      context.stroke();

      if (isSelected) {
        context.fillStyle = `${threatColor(detection.threat_level)}55`;
        context.fill();
      }
    }
  }, [detections, isLoaded, scale, selectedId]);

  function handleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    // The canvas is laid out with CSS max-width, so its rendered size can
    // differ from its backing size; both factors are needed to recover the
    // original image coordinate that was clicked.
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) * (canvas.width / rect.width)) / scale;
    const y = ((event.clientY - rect.top) * (canvas.height / rect.height)) / scale;

    const hit = detections.find((detection) => {
      const xs = detection.box.corners.map(([cx]) => cx);
      const ys = detection.box.corners.map(([, cy]) => cy);
      return (
        x >= Math.min(...xs) &&
        x <= Math.max(...xs) &&
        y >= Math.min(...ys) &&
        y <= Math.max(...ys)
      );
    });

    if (hit) {
      onSelect(hit.id);
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
