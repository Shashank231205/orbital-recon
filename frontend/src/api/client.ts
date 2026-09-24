/**
 * HTTP client for the Orbital Recon API.
 *
 * Errors from the backend carry a code and message; they are rethrown as
 * `RequestError` so callers can show the server's own wording rather than a
 * generic failure string.
 */

import type {
  BriefResponse,
  DetectionList,
  Health,
  Job,
  JobSummary,
  Modality,
  ProviderStatus,
  QueryResponse,
  Scene,
  ThreatLevel,
  UploadResponse,
} from "@/types/api";

const BASE_URL = "/api/v1";

export class RequestError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "RequestError";
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init);

  if (!response.ok) {
    let code = "http_error";
    let message = `Request failed with status ${response.status}`;

    try {
      const body = (await response.json()) as { code?: string; message?: string };
      code = body.code ?? code;
      message = body.message ?? message;
    } catch {
      // A non-JSON error body leaves the defaults in place.
    }

    throw new RequestError(response.status, code, message);
  }

  return (await response.json()) as T;
}

export interface UploadOptions {
  file: File;
  modality: Modality;
  confidence?: number;
}

export const api = {
  health: () => request<Health>("/health"),

  refreshProviders: () =>
    request<ProviderStatus[]>("/health/providers/refresh", { method: "POST" }),

  uploadScene: ({ file, modality, confidence }: UploadOptions) => {
    const form = new FormData();
    form.append("file", file);
    form.append("modality", modality);
    if (confidence !== undefined) {
      form.append("confidence", String(confidence));
    }
    return request<UploadResponse>("/scenes", { method: "POST", body: form });
  },

  getScene: (sceneId: number) => request<Scene>(`/scenes/${sceneId}`),

  getJob: (jobId: number) => request<Job>(`/jobs/${jobId}`),

  getSummary: (jobId: number) => request<JobSummary>(`/jobs/${jobId}/summary`),

  listDetections: (
    jobId: number,
    options: { minConfidence?: number; threatLevel?: ThreatLevel; limit?: number } = {},
  ) => {
    const params = new URLSearchParams();
    if (options.minConfidence !== undefined) {
      params.set("min_confidence", String(options.minConfidence));
    }
    if (options.threatLevel) {
      params.set("threat_level", options.threatLevel);
    }
    params.set("limit", String(options.limit ?? 1000));

    return request<DetectionList>(`/jobs/${jobId}/detections?${params.toString()}`);
  },

  query: (jobId: number, question: string) =>
    request<QueryResponse>(`/jobs/${jobId}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }),

  brief: (jobId: number) =>
    request<BriefResponse>(`/jobs/${jobId}/brief`, { method: "POST" }),
};

/** URL of a job's progress stream, consumed with `EventSource`. */
export function progressStreamUrl(jobId: number): string {
  return `${BASE_URL}/jobs/${jobId}/stream`;
}
