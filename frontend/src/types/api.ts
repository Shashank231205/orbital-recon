/** Types mirroring the backend API schemas. */

export type Modality = "eo" | "ir" | "sar";

export type JobStatus = "pending" | "running" | "completed" | "failed";

export type JobStage =
  | "queued"
  | "loading"
  | "preprocessing"
  | "tiling"
  | "inference"
  | "merging"
  | "georeferencing"
  | "persisting"
  | "done";

export type ThreatLevel = "high" | "medium" | "low" | "unknown";

export interface Scene {
  id: number;
  filename: string;
  modality: Modality;
  width: number;
  height: number;
  size_bytes: number;
  crs: string | null;
  is_georeferenced: boolean;
  created_at: string;
}

export interface Job {
  id: number;
  scene_id: number;
  status: JobStatus;
  stage: JobStage;
  progress: number;
  message: string | null;
  error: string | null;
  detection_count: number;
  duration_ms: number | null;
  created_at: string;
  updated_at: string;
}

export interface UploadResponse {
  scene: Scene;
  job: Job;
}

export interface GeoPoint {
  latitude: number;
  longitude: number;
}

export interface BoundingBox {
  cx: number;
  cy: number;
  width: number;
  height: number;
  angle: number;
  corners: [number, number][];
}

export interface Detection {
  id: number;
  class_name: string;
  confidence: number;
  threat_level: ThreatLevel;
  box: BoundingBox;
  centroid: GeoPoint | null;
  footprint: GeoPoint[] | null;
}

export interface DetectionList {
  job_id: number;
  total: number;
  items: Detection[];
}

export interface ClassCount {
  label: string;
  display_name: string;
  threat_level: ThreatLevel;
  count: number;
}

export interface JobSummary {
  job_id: number;
  scene_id: number;
  total_detections: number;
  mean_confidence: number;
  by_class: ClassCount[];
  by_threat: Record<string, number>;
}

export interface QueryResponse {
  job_id: number;
  question: string;
  answer: string;
  provider: string;
  model: string;
  latency_ms: number;
}

export interface BriefResponse {
  job_id: number;
  brief: string;
  provider: string;
  model: string;
  latency_ms: number;
}

export interface ProviderStatus {
  name: string;
  configured: boolean;
  healthy: boolean;
  consecutive_failures: number;
}

export interface Health {
  status: string;
  version: string;
  device: string;
  detector_loaded: boolean;
  llm_providers: ProviderStatus[];
}

/** One frame of the job progress stream. */
export interface ProgressEvent {
  job_id: number;
  status: JobStatus;
  stage: JobStage;
  progress: number;
  message: string | null;
  detection_count: number;
  error: string | null;
}

export interface ApiError {
  code: string;
  message: string;
  details: Record<string, unknown>;
}
