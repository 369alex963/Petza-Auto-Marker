import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, compareUrl, type HoldoutItem, type Run } from "../api";
import { Heart } from "../icons";

const pct = (v: number | null | undefined) => (v == null ? "" : `${(v * 100).toFixed(1)}%`);

export function Teach({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hard, setHard] = useState<HoldoutItem[]>([]);
  const [open, setOpen] = useState<HoldoutItem | null>(null);
  const [switching, setSwitching] = useState(false);
  const runId = ctx.status?.model?.run_id ?? "";

  useEffect(() => {
    api
      .runs()
      .then((r) => setRuns(r.runs))
      .catch((e: Error) => setError(e.message));
  }, [ctx.jobVersion, runId]);

  useEffect(() => {
    setOpen(null);
    if (!runId) return setHard([]);
    api
      .holdout()
      .then((r) => setHard(r.items.slice(0, 8)))
      .catch(() => setHard([]));
  }, [runId]);

  const switchTo = async (id: string) => {
    setSwitching(true);
    try {
      await api.useRun(id);
      await ctx.refresh();
      ctx.toast("Switched to that model");
    } catch (e) {
      ctx.toast((e as Error).message);
    } finally {
      setSwitching(false);
    }
  };

  const busy = ctx.job?.status === "running";
  const pool = s?.pool;
  const canTrain = (pool?.train ?? 0) >= 3;
  const model = s?.model;

  return (
    <main className="page">
      <div className="page__head">
        <div>
          <h1 className="page__title">Teach</h1>
          <p className="page__sub">
            The app learns from your marked examples and from every photo you approve. Some photos are kept aside and never learned from, so the score below is an honest test.
          </p>
        </div>
        <div className="toolbar">
          <button type="button" className="btn" disabled={busy || !s?.settings.examples_dir} onClick={() => void ctx.startJob("import")}>
            Import examples
          </button>
          <button type="button" className="btn btn--primary" disabled={busy || !canTrain} onClick={() => void ctx.startJob("train")}>
            Teach now
          </button>
        </div>
      </div>

      {!s?.settings.examples_dir ? (
        <div className="note">
          <p>Choose the folder of marked examples first.</p>
          <a className="btn" href="#/setup">
            Choose folders
          </a>
        </div>
      ) : !canTrain ? (
        <div className="note">
          <p>Import the marked examples to start. Teaching needs at least 3 photos to learn from.</p>
        </div>
      ) : null}

      <div className="cols">
        <div className={`diecut score ${model?.iou == null ? "score--none" : ""}`}>
          {model?.iou != null ? (
            <>
              <span className="score__fig">{pct(model.iou)}</span>
              <span className="score__label">Overlap with your own marking</span>
              <span className="score__sub">
                Measured on {pool?.holdout ?? 0} photos kept aside. Edge accuracy {pct(model.boundary_f)}.
              </span>
            </>
          ) : (
            <>
              <span className="score__label">No score yet</span>
              <span className="score__sub">{model ? "Add more examples so some can be kept aside for testing." : "Teach the app once to see how well it marks."}</span>
            </>
          )}
        </div>
        <section className="panel" aria-labelledby="pool">
          <h2 className="panel__title" id="pool">
            What it learns from
          </h2>
          <div className="stats">
            <div className="stat">
              <span className="stat__n">{pool?.examples ?? 0}</span>
              <span className="stat__l">Examples</span>
            </div>
            <div className="stat">
              <span className="stat__n">{pool?.reviewed ?? 0}</span>
              <span className="stat__l">Fixes</span>
            </div>
            <div className="stat">
              <span className="stat__n">{pool?.holdout ?? 0}</span>
              <span className="stat__l">Kept for testing</span>
            </div>
          </div>
          <p className="muted">
            Labels: {pool?.labels.join(", ") || "wound"}. Teaching takes about {s?.device.train_device === "cuda" ? "5 to 15 minutes on the graphics card" : "20 to 60 minutes on this processor"}.
          </p>
        </section>
      </div>

      {hard.length ? (
        <section aria-labelledby="hard">
          <h2 className="panel__title" id="hard">
            Hardest test photos
          </h2>
          <p className="muted" style={{ marginBottom: "var(--space-sm)" }}>
            Dark line: your outline. Pink line: the app’s. More examples like these help the most.
          </p>
          <div className="thumbs">
            {hard.map((h) => (
              <button type="button" key={h.id} className="thumb thumb--btn" aria-pressed={open?.id === h.id} onClick={() => setOpen(open?.id === h.id ? null : h)}>
                <span className="thumb__pic">
                  <img src={compareUrl(h.id, runId, 440)} alt={`Test photo ${h.name}`} loading="lazy" />
                  <span className="thumb__tag">{pct(h.iou)}</span>
                </span>
                <span className="thumb__name">{h.name}</span>
              </button>
            ))}
          </div>
          {open ? (
            <figure className="compare">
              <img src={compareUrl(open.id, runId, 1400)} alt={`Your outline and the app’s on ${open.name}`} />
              <figcaption className="muted">
                {open.name}: overlap {pct(open.iou)}, edges {pct(open.boundary_f)}.
              </figcaption>
            </figure>
          ) : null}
        </section>
      ) : null}

      <section aria-labelledby="hist">
        <h2 className="panel__title" id="hist" style={{ marginBottom: "var(--space-sm)" }}>
          Teaching history
        </h2>
        {error ? (
          <div className="note note--error" role="alert">
            <p>The history did not load: {error}</p>
          </div>
        ) : runs === null ? (
          <p className="muted">Loading</p>
        ) : runs.length === 0 ? (
          <p className="muted">Nothing taught yet. The first run will appear here.</p>
        ) : (
          <table className="runs">
            <thead>
              <tr>
                <th>When</th>
                <th>Photos</th>
                <th>Accuracy</th>
                <th>Before</th>
                <th>Edges</th>
                <th>Time</th>
                <th>Result</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td>{new Date(r.created_at).toLocaleString()}</td>
                  <td>{r.train_count}</td>
                  <td>{pct(r.iou)}</td>
                  <td>{pct(r.previous_iou)}</td>
                  <td>{pct(r.boundary_f)}</td>
                  <td>{r.minutes} min</td>
                  <td>{r.promoted ? "Better or equal, switched" : "Kept the older one"}</td>
                  <td>
                    {r.current ? (
                      <span className="pill pill--live">
                        <Heart size={12} /> In use
                      </span>
                    ) : r.available ? (
                      <button type="button" className="btn btn--small" disabled={busy || switching} onClick={() => void switchTo(r.run_id)}>
                        Use this one
                      </button>
                    ) : (
                      <span className="muted">Removed</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
