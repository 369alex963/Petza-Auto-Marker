export type Point = [number, number];

export interface Region {
  id: string;
  label: string;
  points: Point[];
  confidence: number | null;
  source: "auto" | "manual";
}

export interface Review {
  ok: boolean;
  problems: string[];
  summary: string;
}

export interface Marking {
  image: string;
  width: number;
  height: number;
  regions: Region[];
  model_version: string | null;
  created_at: string;
  status: "auto" | "reviewed";
  confidence: number | null;
  review: Review | null;
}

export interface Job {
  kind: string | null;
  status: "idle" | "running" | "done" | "error" | "stopped";
  progress: number;
  message: string;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface Status {
  device: {
    train_device: string;
    gpu_name: string | null;
    gpu_memory_gb?: number;
    cpu_threads: number;
    ort_providers: string[];
    inference_label: string;
  };
  model: {
    run_id: string;
    created_at: string;
    classes: string[];
    iou: number | null;
    dice: number | null;
    boundary_f: number | null;
    train_count: number;
    device: string;
  } | null;
  pool: { total: number; examples: number; reviewed: number; train: number; holdout: number; labels: string[] };
  settings: {
    images_dir: string;
    rules_path: string;
    examples_dir: string;
    output_dir: string;
    anthropic_api_key: boolean;
    reviewer_model: string;
    reviewer_enabled: boolean;
  };
  rules: { loaded: boolean; labels: string[]; text: string; review_below_confidence: number };
  job: Job;
}

export interface ImageItem {
  path: string;
  name: string;
  status: "auto" | "reviewed" | "foreign" | null;
  confidence: number | null;
  regions: number;
  needs_review: boolean;
  review: Review | null;
}

export interface Run {
  run_id: string;
  created_at: string;
  train_count: number;
  holdout_count: number;
  iou: number | null;
  dice: number | null;
  boundary_f: number | null;
  previous_iou: number | null;
  promoted: boolean;
  device: string;
  minutes: number;
  finetuned: boolean;
}

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (data?.detail) detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => call<Status>("GET", "/api/status"),
  saveSettings: (s: Partial<Record<string, string | boolean>>) => call<Status>("POST", "/api/settings", s),
  browse: (kind: "folder" | "file", initial = "") => call<{ path: string }>("POST", "/api/browse", { kind, initial }),
  images: () => call<{ items: ImageItem[] }>("GET", "/api/images"),
  marking: (path: string) =>
    call<{ marking: Marking; json_path: string; empty?: boolean }>("GET", `/api/marking?path=${encodeURIComponent(path)}`),
  saveMarking: (path: string, marking: Marking) =>
    call<{ ok: boolean; json_path: string }>("PUT", "/api/marking", { path, marking }),
  job: () => call<Job>("GET", "/api/jobs/current"),
  start: (kind: "import" | "train" | "mark" | "review", body?: unknown) => call<Job>("POST", `/api/jobs/${kind}`, body),
  cancel: () => call<Job>("POST", "/api/jobs/cancel"),
  runs: () => call<{ runs: Run[] }>("GET", "/api/runs"),
};

export const imageUrl = (path: string, max = 0) => `/api/image?path=${encodeURIComponent(path)}${max ? `&max=${max}` : ""}`;
