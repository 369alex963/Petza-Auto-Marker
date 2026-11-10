import { Component, useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { api, type Job, type Status } from "./api";
import { Heart, Stop } from "./icons";
import { Editor } from "./screens/Editor";
import { Mark } from "./screens/Mark";
import { Settings } from "./screens/Settings";
import { Setup } from "./screens/Setup";
import { Teach } from "./screens/Teach";
import { TryOne } from "./screens/TryOne";

export type Route = { name: "mark" | "try" | "teach" | "setup" | "settings" | "edit"; path?: string };

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  const [name, query] = h.split("?");
  const params = new URLSearchParams(query ?? "");
  if (name === "edit") return { name: "edit", path: params.get("path") ?? undefined };
  if (name === "try") return { name: "try", path: params.get("path") ?? undefined };
  if (name === "teach" || name === "setup" || name === "settings") return { name };
  return { name: "mark" };
}

export const go = (hash: string) => {
  window.location.hash = hash;
};
export const editHash = (path: string) => `#/edit?path=${encodeURIComponent(path)}`;

export interface AppCtx {
  status: Status | null;
  statusError: string | null;
  refresh: () => Promise<void>;
  job: Job | null;
  startJob: (kind: "import" | "train" | "mark" | "review", body?: unknown) => Promise<void>;
  jobVersion: number;
  toast: (msg: string) => void;
}

const TABS: { name: Route["name"]; label: string }[] = [
  { name: "mark", label: "Photos" },
  { name: "try", label: "Try one photo" },
  { name: "teach", label: "Teach" },
  { name: "setup", label: "Folders" },
  { name: "settings", label: "Settings" },
];

const JOB_LABEL: Record<string, string> = {
  import: "Importing examples",
  train: "Teaching",
  mark: "Marking photos",
  review: "Marking again with the AI",
  connect: "Connecting the AI account",
};

export function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [status, setStatus] = useState<Status | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [jobVersion, setJobVersion] = useState(0);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const lastStatus = useRef<string>("idle");

  useEffect(() => {
    const on = () => setRoute(parseHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const s = await api.status();
      setStatus(s);
      setJob(s.job);
      setStatusError(null);
    } catch (e) {
      setStatusError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll the job while one runs; when it finishes, everything refreshes once.
  useEffect(() => {
    const running = job?.status === "running";
    const t = window.setInterval(
      async () => {
        try {
          const j = await api.job();
          setJob(j);
          if (lastStatus.current === "running" && j.status !== "running") {
            setJobVersion((v) => v + 1);
            void refresh();
          }
          lastStatus.current = j.status;
        } catch {
          /* server restarting */
        }
      },
      running ? 600 : 4000,
    );
    return () => window.clearInterval(t);
  }, [job?.status, refresh]);

  const toast = useCallback((msg: string) => {
    setToastMsg(msg);
    window.setTimeout(() => setToastMsg(null), 2600);
  }, []);

  const startJob = useCallback(
    async (kind: "import" | "train" | "mark" | "review", body?: unknown) => {
      try {
        const j = await api.start(kind, body);
        lastStatus.current = "running";
        setJob(j);
      } catch (e) {
        toast((e as Error).message);
      }
    },
    [toast],
  );

  const ctx: AppCtx = { status, statusError, refresh, job, startJob, jobVersion, toast };
  const current = route.name === "edit" ? "mark" : route.name;

  return (
    <>
      <header className="bar">
        <div className="bar__in">
          <a className="wordmark" href="#/mark" aria-label="Wound-Recognizer, photos">
            Wound-Recognizer
          </a>
          <nav className="tabs" aria-label="Main">
            {TABS.map((t) => (
              <a key={t.name} className="tab" href={`#/${t.name}`} aria-current={current === t.name ? "page" : undefined}>
                {t.label}
              </a>
            ))}
          </nav>
        </div>
      </header>
      <JobStrip job={job} />
      {statusError && !status ? (
        <main className="page">
          <div className="note note--error" role="alert">
            <p>The app's engine is not answering ({statusError}). Is the black start window still open? Start it again with start.bat.</p>
            <button type="button" className="btn" onClick={() => void refresh()}>
              Try again
            </button>
          </div>
        </main>
      ) : (
        <Guard resetKey={window.location.hash}>
          {route.name === "edit" && route.path ? (
        <Editor key={route.path} path={route.path} ctx={ctx} />
      ) : route.name === "try" ? (
        <TryOne ctx={ctx} path={route.path} />
      ) : route.name === "teach" ? (
        <Teach ctx={ctx} />
      ) : route.name === "setup" ? (
        <Setup ctx={ctx} />
      ) : route.name === "settings" ? (
        <Settings ctx={ctx} />
      ) : (
        <Mark ctx={ctx} />
          )}
        </Guard>
      )}
      {toastMsg ? (
        <div className="toast" role="status">
          <span className="toast__heart">
            <Heart size={16} />
          </span>
          {toastMsg}
        </div>
      ) : null}
    </>
  );
}

/** A screen that fails shows a way back instead of a blank page. Saved files are untouched. */
class Guard extends Component<{ children: ReactNode; resetKey: string }, { error: string | null; key: string }> {
  state = { error: null as string | null, key: this.props.resetKey };
  static getDerivedStateFromError(e: Error) {
    return { error: e.message };
  }
  static getDerivedStateFromProps(p: { resetKey: string }, s: { key: string }) {
    return p.resetKey !== s.key ? { error: null, key: p.resetKey } : null;
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="page">
        <div className="note note--error" role="alert">
          <p>This screen ran into a problem ({this.state.error}). Nothing saved was lost.</p>
          <button type="button" className="btn" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </main>
    );
  }
}

function JobStrip({ job }: { job: Job | null }) {
  const [hideDone, setHideDone] = useState(false);
  useEffect(() => setHideDone(false), [job?.status, job?.kind]);
  if (!job || job.status === "idle" || hideDone) return null;
  const label = JOB_LABEL[job.kind ?? ""] ?? "Working";
  const pct = Math.round(job.progress * 100);
  if (job.status === "running") {
    return (
      <div className="job" role="status" aria-live="polite">
        <div className="job__in">
          <span className="job__msg">
            {label}: {job.message}
          </span>
          <div className="job__track" aria-hidden="true">
            <div className="job__fill" style={{ transform: `scaleX(${job.progress})` }} />
          </div>
          <span className="job__pct">{pct}%</span>
          <button type="button" className="btn btn--small" onClick={() => void api.cancel()}>
            <Stop /> Stop
          </button>
        </div>
      </div>
    );
  }
  const text =
    job.status === "error" ? `${label} stopped with a problem: ${job.error}` : job.status === "stopped" ? `${label} was stopped.` : `${label} finished. ${summarize(job)}`;
  return (
    <div className={`job ${job.status === "error" ? "job--error" : "job--done"}`} role="status">
      <div className="job__in">
        <span className="job__msg">{text}</span>
        <button type="button" className="btn btn--small btn--quiet" onClick={() => setHideDone(true)}>
          Close
        </button>
      </div>
    </div>
  );
}

function summarize(job: Job): string {
  const r = job.result as Record<string, unknown> | null;
  if (!r) return "";
  if (job.kind === "import") return `${r.imported} of ${r.total} examples imported${(r.no_marking as unknown[])?.length ? `, ${(r.no_marking as unknown[]).length} had no marking file` : ""}.`;
  if (job.kind === "mark" || job.kind === "review") {
    const ai = (r.ai_improved as number) || (r.ai_failed as number) ? ` The AI improved ${r.ai_improved}${r.ai_failed ? `, ${r.ai_failed} kept the local marking because the AI did not answer` : ""}.` : "";
    return `${r.marked} marked, ${r.needs_review} worth a look, ${r.skipped_reviewed} already approved.${ai}`;
  }
  if (job.kind === "connect") return "Finish signing in in the browser window that opened, then come back here.";
  if (job.kind === "train") {
    const iou = r.iou as number | null;
    return iou == null ? "" : `Accuracy ${(iou * 100).toFixed(1)}%. ${r.promoted ? "The new model is live." : "The previous model scored better and stays."}`;
  }
  return "";
}
