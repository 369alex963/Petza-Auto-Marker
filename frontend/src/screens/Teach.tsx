import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, type Run } from "../api";
import { Heart } from "../icons";

const pct = (v: number | null | undefined) => (v == null ? "" : `${(v * 100).toFixed(1)}%`);

export function Teach({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [runs, setRuns] = useState<Run[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .runs()
      .then((r) => setRuns(r.runs))
      .catch((e: Error) => setError(e.message));
  }, [ctx.jobVersion]);

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
            The app learns from Lauren’s marked examples and from every photo she approves. Some photos are kept aside and never learned from, so the score below is an honest test.
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
              <span className="score__label">Overlap with Lauren’s own marking</span>
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
              <span className="stat__l">Lauren’s fixes</span>
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
                  <td>
                    {r.promoted ? (
                      <span className="pill pill--live">
                        <Heart size={12} /> In use
                      </span>
                    ) : (
                      <span className="pill">Kept the older one</span>
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
