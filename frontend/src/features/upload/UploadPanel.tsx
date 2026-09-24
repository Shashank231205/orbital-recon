/** Scene upload controls. */

import { useRef, useState } from "react";

import { api, RequestError } from "@/api/client";
import { Spinner } from "@/components/Indicators";
import type { Modality, UploadResponse } from "@/types/api";

const MODALITY_LABELS: Record<Modality, string> = {
  eo: "Electro-optical",
  ir: "Infrared",
  sar: "Synthetic-aperture radar",
};

interface Props {
  /** Receives the server response together with the file that produced it,
   *  so the caller can preview the scene without downloading it back. */
  onUploaded: (result: UploadResponse, file: File) => void;
  disabled: boolean;
}

export function UploadPanel({ onUploaded, disabled }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [modality, setModality] = useState<Modality>("eo");
  const [confidence, setConfidence] = useState(0.25);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function submit() {
    if (!file) {
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      onUploaded(await api.uploadScene({ file, modality, confidence }), file);
      setFile(null);
      if (inputRef.current) {
        inputRef.current.value = "";
      }
    } catch (cause) {
      setError(
        cause instanceof RequestError ? cause.message : "Upload failed unexpectedly",
      );
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="card">
      <h2>Scene</h2>

      <div className="field">
        <label htmlFor="scene-file">Image file</label>
        <input
          id="scene-file"
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          disabled={disabled || isUploading}
          onChange={(event) => {
            setFile(event.target.files?.[0] ?? null);
            setError(null);
          }}
        />
        <span className="dim mono">
          GeoTIFF carries coordinates; other formats stay in pixel space.
        </span>
      </div>

      <div className="field">
        <label htmlFor="scene-modality">Sensor modality</label>
        <select
          id="scene-modality"
          value={modality}
          disabled={disabled || isUploading}
          onChange={(event) => setModality(event.target.value as Modality)}
        >
          {Object.entries(MODALITY_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label htmlFor="scene-confidence">
          Confidence threshold: <span className="mono">{confidence.toFixed(2)}</span>
        </label>
        <input
          id="scene-confidence"
          type="range"
          min={0.05}
          max={0.9}
          step={0.05}
          value={confidence}
          disabled={disabled || isUploading}
          onChange={(event) => setConfidence(Number(event.target.value))}
        />
      </div>

      <button
        type="button"
        className="primary"
        disabled={!file || disabled || isUploading}
        onClick={() => void submit()}
        style={{ width: "100%" }}
      >
        {isUploading ? (
          <span className="row" style={{ justifyContent: "center" }}>
            <Spinner /> Uploading
          </span>
        ) : (
          "Analyse scene"
        )}
      </button>

      {error && (
        <p className="error-text" style={{ marginBottom: 0, marginTop: 10 }}>
          {error}
        </p>
      )}
    </div>
  );
}
