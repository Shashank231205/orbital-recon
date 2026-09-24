/**
 * Geographic view of detections.
 *
 * Each detection is drawn as its true oriented footprint where one is
 * available, rather than a marker pin, because the footprint carries the
 * asset's size and heading. A centroid marker is used only when the footprint
 * is missing.
 */

import { useEffect, useMemo } from "react";
import { MapContainer, Polygon, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { LatLngBoundsExpression, LatLngExpression } from "leaflet";

import { threatColor } from "@/components/Indicators";
import type { Detection } from "@/types/api";

interface Props {
  detections: Detection[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}

/** Recentres the map whenever the set of plotted detections changes. */
function FitBounds({ bounds }: { bounds: LatLngBoundsExpression | null }) {
  const map = useMap();

  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 17 });
    }
  }, [bounds, map]);

  return null;
}

export function DetectionMap({ detections, selectedId, onSelect }: Props) {
  const plottable = useMemo(
    () => detections.filter((detection) => detection.footprint || detection.centroid),
    [detections],
  );

  const bounds = useMemo<LatLngBoundsExpression | null>(() => {
    const points: [number, number][] = [];

    for (const detection of plottable) {
      const source = detection.footprint ?? (detection.centroid ? [detection.centroid] : []);
      for (const point of source) {
        points.push([point.latitude, point.longitude]);
      }
    }

    return points.length > 0 ? points : null;
  }, [plottable]);

  if (plottable.length === 0) {
    return (
      <div className="map-root" style={{ display: "grid", placeItems: "center" }}>
        <p className="empty">
          No geographic positions available.
          <br />
          Upload a GeoTIFF to place detections on the map.
        </p>
      </div>
    );
  }

  return (
    <MapContainer className="map-root" center={[0, 0]} zoom={2} preferCanvas>
      <TileLayer
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attribution="Imagery &copy; Esri"
        maxZoom={19}
      />
      <FitBounds bounds={bounds} />

      {plottable.map((detection) => {
        const positions: LatLngExpression[] = (
          detection.footprint ??
          (detection.centroid
            ? [
                // A detection without a footprint still needs an outline, so a
                // small square is drawn around its centroid.
                { latitude: detection.centroid.latitude - 0.00004, longitude: detection.centroid.longitude - 0.00004 },
                { latitude: detection.centroid.latitude - 0.00004, longitude: detection.centroid.longitude + 0.00004 },
                { latitude: detection.centroid.latitude + 0.00004, longitude: detection.centroid.longitude + 0.00004 },
                { latitude: detection.centroid.latitude + 0.00004, longitude: detection.centroid.longitude - 0.00004 },
              ]
            : [])
        ).map((point) => [point.latitude, point.longitude]);

        const isSelected = detection.id === selectedId;
        const colour = threatColor(detection.threat_level);

        return (
          <Polygon
            key={detection.id}
            positions={positions}
            pathOptions={{
              color: colour,
              weight: isSelected ? 3 : 1.5,
              fillOpacity: isSelected ? 0.45 : 0.2,
            }}
            eventHandlers={{ click: () => onSelect(detection.id) }}
          >
            <Tooltip direction="top" offset={[0, -4]}>
              <strong>{detection.class_name}</strong>
              <br />
              {(detection.confidence * 100).toFixed(1)}% confidence
            </Tooltip>
          </Polygon>
        );
      })}
    </MapContainer>
  );
}
