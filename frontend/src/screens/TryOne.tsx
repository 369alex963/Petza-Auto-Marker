import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { go } from "../App";
import { api, imageUrl, QUALITY_TEXT, type ImageItem, type Region, type TryResult } from "../api";

const OUTLINE_NAME = { A: "A, tight", B: "B, normal", C: "C, wide" } as const;
const PROVIDER_NAME = { claude: "Claude", openai: "ChatGPT", groq: "Groq" } as const;

function Outlined({ path, size, regions, label }: { path: string; size: [number, number]; regions: Region[]; label: string }) {
  return (
    <figure className="try__fig">
      <div className="try__pic">
        <img src={imageUrl(path, 1200)} alt="" />
        <svg viewBox={`0 0 ${size[0]} ${size[1]}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
          {regions.map((r, k) => (
            <polygon key={k} points={r.points.map((p) => p.join(",")).join(" ")} />
          ))}
        </svg>
      </div>
      <figcaption>{label}</figcaption>
    </figure>
  );
}

export function TryOne({ ctx, path: initial }: { ctx: AppCtx; path?: string }) {
  const s = ctx.status;
  const [items, setItems] = useState<ImageItem[] | null>(null);
  const [path, setPath] = useState(initial ?? "");
  const [result, setResult] = useState<TryResult | null>(null);
  const [working, setWorking] = useState<"try" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .images()
      .then((r) => {
        setItems(r.items);
        setPath((p) => p || r.items[0]?.path || "");
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  const run = async (save: boolean) => {
    setWorking(save ? "save" : "try");
    setError(null);
    if (!save) setResult(null);
    try {
      const r = await api.tryOne(path, save);
      setResult(r);
      if (save) ctx.toast("Saved. This is the photo's marking now.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setWorking(null);
    }
  };

  const busy = working !== null || ctx.job?.status === "running";
  const ai = result?.ai;
  const aiOn = !!s?.settings.reviewer_enabled && !!s?.reviewer.ready;

  return (
    <main className="page">
      <div className="page__head">
        <div>
          <h1 className="page__title">Try one photo</h1>
          <p className="page__sub">
            Exactly what “Mark all photos” does, on one photo, with every step shown. Check the result here, then mark all photos the same way.
          </p>
        </div>
      </div>

      <section className="panel">
        <div className="field">
          <label className="field__label" htmlFor="photo">
            Photo
          </label>
          <div className="field__row">
            <select id="photo" className="input" value={path} onChange={(e) => (setPath(e.target.value), setResult(null))} disabled={!items?.length}>
              {(items ?? []).map((i) => (
                <option key={i.path} value={i.path}>
                  {i.name}
                </option>
              ))}
            </select>
            <button type="button" className="btn btn--primary" disabled={!path || busy || !s?.model} onClick={() => void run(false)}>
              {working === "try" ? "Working, up to a minute" : "Try it"}
            </button>
          </div>
          <span className="field__help">
            {!s?.model
              ? "Teach the app first (Teach tab)."
              : aiOn
                ? `Uses ${PROVIDER_NAME[s.settings.reviewer_provider]}, ${s.settings.ai_runs} ${s.settings.ai_runs === 1 ? "run" : "runs"} per photo, and the active marking prompt “${s.rules.prompt}”. Change them in Settings.`
                : "The AI is off, so this shows the local model only. Turn the AI on in Settings."}
          </span>
        </div>
        {items && !items.length ? <p className="muted">No photos yet. Choose the photo folder in the Folders tab.</p> : null}
        {error ? (
          <div className="note note--error" role="alert">
            <p>{error}</p>
          </div>
        ) : null}
      </section>

      {result ? (
        <>
          <section className="panel" aria-label="Result">
            <dl className="facts">
              <dt>Photo quality</dt>
              <dd>{QUALITY_TEXT[result.quality]}</dd>
              <dt>AI</dt>
              <dd>
                {!result.used_ai
                  ? "Off: local model only"
                  : ai?.skipped
                    ? "Not asked: one confident wound in a good photo (Settings can ask about every photo)"
                    : ai?.ok
                    ? `${ai.via}, ${ai.runs} ${ai.runs === 1 ? "run" : "runs"}, ${ai.seconds} s${ai.missed ? `, found ${ai.missed} missed ${ai.missed === 1 ? "area" : "areas"}` : ""}`
                    : "Did not answer, so the local marking is kept"}
              </dd>
              <dt>Result</dt>
              <dd>
                {result.final.length} {result.final.length === 1 ? "area" : "areas"}, {Math.round(result.confidence * 100)}% sure, {result.seconds} s in all
              </dd>
            </dl>
            {result.errors.length ? (
              <div className="note note--error">
                <ul>
                  {result.errors.map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              </div>
            ) : null}
            {ai?.notes ? <p dir="auto">{ai.notes}</p> : null}
          </section>

          <div className="try">
            <Outlined path={path} size={result.size} regions={result.local} label="1. The local model alone" />
            {result.options_image ? (
              <figure className="try__fig">
                <div className="try__pic">
                  <img src={`data:image/jpeg;base64,${result.options_image}`} alt="The numbered outline options that were sent to the AI" />
                </div>
                <figcaption>2. What the AI was shown: A blue tight, B yellow normal, C white wide</figcaption>
              </figure>
            ) : null}
            <Outlined path={path} size={result.size} regions={result.final} label={result.used_ai && ai?.ok ? "3. Final, as the AI chose" : "Final"} />
          </div>

          {ai?.choices?.length ? (
            <section className="panel" aria-label="The AI's choices">
              <h2 className="panel__title">The AI's choice for each area</h2>
              <table className="runs">
                <thead>
                  <tr>
                    <th>Area</th>
                    <th>Decision</th>
                    <th>Outline</th>
                    <th>Agreed</th>
                    <th>Why</th>
                  </tr>
                </thead>
                <tbody>
                  {ai.choices.map((c) => (
                    <tr key={c.id}>
                      <td>{c.id}</td>
                      <td>{c.keep ? `Kept as ${c.label}` : "Removed"}</td>
                      <td>{c.keep ? `${OUTLINE_NAME[c.outline]}${c.kept_local_edge ? " (app's edge kept)" : ""}` : ""}</td>
                      <td>{c.agreed}</td>
                      <td dir="auto">{c.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ) : null}

          <div className="toolbar">
            <button type="button" className="btn" disabled={busy} onClick={() => void run(true)}>
              {working === "save" ? "Saving" : "Save this result"}
            </button>
            <button
              type="button"
              className="btn btn--primary"
              disabled={busy}
              onClick={() => {
                void ctx.startJob("mark");
                go("#/mark");
              }}
            >
              Mark all photos like this
            </button>
          </div>
        </>
      ) : null}
    </main>
  );
}
