export type Point = [number, number];

export interface Region {
  id: string;
  label: string;
  points: Point[];
  confidence: number | null;
  source: "auto" | "manual";
}

export type Provider = "claude" | "openai" | "groq";
export type Connection = "account" | "api";

export interface Prompts {
  active: string;
  prompts: Record<string, string>;
  default_name: string;
  default_text: string;
}

export interface Review {
  ok: boolean;
  problems: string[];
  summary: string;
  via?: string;
}

export type Quality = "good" | "dark" | "low_contrast" | "blurry";

export interface AiChoice {
  id: number;
  keep: boolean;
  outline: "A" | "B" | "C";
  label: string;
  reason: string;
  agreed: string;
  kept_local_edge?: boolean;
}

export interface AiInfo {
  ok: boolean;
  skipped?: boolean;
  via?: string;
  runs?: number;
  choices?: AiChoice[];
  missed?: number;
  notes?: string;
  corrected?: boolean;
  seconds?: number;
  errors?: string[];
}

export interface Account {
  installed: boolean;
  signed_in: boolean;
  who: string | null;
  in_app: boolean;
}

export interface TryResult {
  size: [number, number];
  quality: Quality;
  seconds: number;
  used_ai: boolean;
  local: Region[];
  final: Region[];
  confidence: number;
  candidates: { id: number; label: string; outlines: Record<"A" | "B" | "C", Point[][]> }[];
  options_image: string | null;
  ai: AiInfo | null;
  errors: string[];
  saved: string | null;
  runs: number;
  provider: Provider;
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
  ai?: AiInfo | null;
  quality?: Quality | null;
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
  pool: { total: number; examples: number; reviewed: number; ai: number; train: number; holdout: number; labels: string[] };
  settings: {
    images_dir: string;
    rules_path: string;
    examples_dir: string;
    output_dir: string;
    anthropic_api_key: boolean;
    reviewer_model: string;
    reviewer_enabled: boolean;
    reviewer_provider: Provider;
    openai_api_key: boolean;
    openai_model: string;
    groq_api_key: boolean;
    groq_model: string;
    ai_runs: number;
    ai_edges: boolean;
    ai_scope: "unsure" | "all";
    openai_effort: string;
    ai_accounts_off: string[];
  };
  reviewer: { claude: Connection[]; openai: Connection[]; groq: Connection[]; ready: boolean; install: { claude: string; codex: string } };
  rules: { prompt: string; labels: string[]; review_below_confidence: number };
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
  ai: boolean | null;
  quality: Quality | null;
  approved_by?: "person" | "auto" | null;
  size?: [number, number];
  outline?: Point[][];
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
  current?: boolean;
  available?: boolean;
}

export interface HoldoutItem {
  id: string;
  name: string;
  kind: string;
  iou: number;
  dice: number;
  boundary_f: number;
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
  saveSettings: (s: Partial<Record<string, string | boolean | number>>) => call<Status>("POST", "/api/settings", s),
  browse: (kind: "folder" | "file", initial = "") => call<{ path: string }>("POST", "/api/browse", { kind, initial }),
  images: () => call<{ items: ImageItem[] }>("GET", "/api/images"),
  marking: (path: string) =>
    call<{ marking: Marking; json_path: string; empty?: boolean }>("GET", `/api/marking?path=${encodeURIComponent(path)}`),
  saveMarking: (path: string, marking: Marking) =>
    call<{ ok: boolean; json_path: string }>("PUT", "/api/marking", { path, marking }),
  job: () => call<Job>("GET", "/api/jobs/current"),
  start: (kind: "import" | "train" | "mark" | "review", body?: unknown) => call<Job>("POST", `/api/jobs/${kind}`, body),
  cancel: () => call<Job>("POST", "/api/jobs/cancel"),
  prompts: () => call<Prompts>("GET", "/api/prompts"),
  putPrompt: (name: string, text: string) => call<Prompts>("PUT", "/api/prompts", { name, text }),
  selectPrompt: (name: string) => call<Prompts>("POST", "/api/prompts/select", { name }),
  deletePrompt: (name: string) => call<Prompts>("POST", "/api/prompts/delete", { name }),
  testReviewer: () => call<{ ok: boolean; via: string | null; errors: string[] }>("POST", "/api/reviewer/test"),
  runs: () => call<{ runs: Run[] }>("GET", "/api/runs"),
  useRun: (runId: string) => call<Status>("POST", `/api/runs/${encodeURIComponent(runId)}/use`),
  holdout: () => call<{ items: HoldoutItem[] }>("GET", "/api/holdout"),
  approveSure: () => call<{ approved: number }>("POST", "/api/approve-sure"),
  tryOne: (path: string, save = false) => call<TryResult>("POST", "/api/try", { path, save }),
  accounts: () => call<Record<"claude" | "openai", Account>>("GET", "/api/ai/accounts"),
  connect: (provider: "claude" | "openai") => call<Job>("POST", "/api/ai/connect", { provider }),
  disconnect: (provider: "claude" | "openai") => call<Record<"claude" | "openai", Account>>("POST", "/api/ai/disconnect", { provider }),
};

export const compareUrl = (id: string, version: string, max = 900) => `/api/compare?id=${encodeURIComponent(id)}&max=${max}&v=${encodeURIComponent(version)}`;
export const QUALITY_TEXT: Record<Quality, string> = {
  good: "Good photo",
  dark: "Dark photo: brightened for the AI only",
  low_contrast: "Flat photo: contrast evened for the AI only",
  blurry: "Blurry photo: worth a look",
};

export const imageUrl = (path: string, max = 0) => `/api/image?path=${encodeURIComponent(path)}${max ? `&max=${max}` : ""}`;
