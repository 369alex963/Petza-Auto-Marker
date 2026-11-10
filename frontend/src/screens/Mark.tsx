import { useEffect, useMemo, useState } from "react";
import { editHash, go, type AppCtx } from "../App";
import { api, imageUrl, QUALITY_TEXT, type ImageItem } from "../api";
import { Heart, Smiley, Sparkle } from "../icons";

type Filter = "all" | "look" | "approved" | "unmarked";

export function Mark({ ctx }: { ctx: AppCtx }) {
  const [items, setItems] = useState<ImageItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>(() => (new URLSearchParams(window.location.hash.split("?")[1]).get("show") as Filter) || "all");
  const s = ctx.status;
  const threshold = s?.rules.review_below_confidence ?? 0.75;

  useEffect(() => {
    let alive = true;
    setError(null);
    api
      .images()
      .then((r) => alive && setItems(r.items))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [ctx.jobVersion, s?.settings.images_dir]);

  const counts = useMemo(() => {
    const c = { all: 0, look: 0, approved: 0, unmarked: 0 };
    for (const i of items ?? []) {
      c.all++;
      if (i.status === "reviewed") c.approved++;
      else if (!i.status || i.status === "foreign") c.unmarked++;
      else if (i.needs_review) c.look++;
    }
    return c;
  }, [items]);

  const shown = (items ?? []).filter((i) =>
    filter === "all" ? true : filter === "approved" ? i.status === "reviewed" : filter === "unmarked" ? !i.status || i.status === "foreign" : i.needs_review,
  );
  // Worried ones first: that is where a human eye is needed.
  const ordered = filter === "look" ? [...shown].sort((a, b) => (a.confidence ?? 0) - (b.confidence ?? 0)) : shown;

  const setF = (f: Filter) => {
    setFilter(f);
    history.replaceState(null, "", f === "all" ? "#/mark" : `#/mark?show=${f}`);
  };

  const sure = (items ?? []).filter((i) => i.status === "auto" && i.regions > 0 && !i.needs_review).length;
  const [approving, setApproving] = useState(false);
  const approveSure = async () => {
    if (!window.confirm(`Approve the ${sure} photos the app is sure about? Their JSON files are kept as final and never overwritten. You can still open and change any of them.`)) return;
    setApproving(true);
    try {
      const r = await api.approveSure();
      ctx.toast(`${r.approved} photos approved`);
      setItems((await api.images()).items);
    } catch (e) {
      ctx.toast((e as Error).message);
    } finally {
      setApproving(false);
    }
  };

  const busy = ctx.job?.status === "running";
  const noModel = s && !s.model;

  return (
    <main className="page">
      <div className="page__head">
        <div>
          <h1 className="page__title">Photos</h1>
          <p className="page__path" title={s?.settings.images_dir}>
            <bdi>{!s ? " " : s.settings.images_dir || "No photo folder chosen yet."}</bdi>
          </p>
        </div>
        <div className="toolbar">
          {s?.settings.reviewer_enabled && s.reviewer.ready ? (
            <button type="button" className="btn" disabled={busy || !items?.length} onClick={() => void ctx.startJob("review")}>
              <Sparkle /> Mark again with the AI
            </button>
          ) : null}
          {sure > 0 ? (
            <button type="button" className="btn" disabled={busy || approving} onClick={() => void approveSure()}>
              <Heart size={16} /> Approve all sure ones ({sure})
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn--primary"
            disabled={busy || !items?.length || !s?.model}
            onClick={() => void ctx.startJob("mark")}
          >
            Mark all photos
          </button>
        </div>
      </div>

      {noModel ? (
        <div className="note">
          <p>Before marking, the app has to learn from your marked examples. Import them and press Teach; it takes a few minutes.</p>
          <a className="btn" href="#/teach">
            Go to Teach
          </a>
        </div>
      ) : null}

      {items && items.length > 0 ? (
        <div className="chips" role="group" aria-label="Show">
          {(
            [
              ["all", "All"],
              ["look", "Worth a look"],
              ["approved", "Approved"],
              ["unmarked", "Not marked"],
            ] as [Filter, string][]
          ).map(([f, label]) => (
            <button key={f} type="button" className="chip" aria-pressed={filter === f} onClick={() => setF(f)}>
              {label} <span className="chip__n">{counts[f]}</span>
            </button>
          ))}
        </div>
      ) : null}

      {error ? (
        <div className="note note--error" role="alert">
          <p>The photo list did not load: {error}</p>
          <button type="button" className="btn" onClick={() => void ctx.refresh()}>
            Try again
          </button>
        </div>
      ) : items === null || !s ? (
        <div className="thumbs" aria-busy="true" aria-label="Loading photos">
          {Array.from({ length: 8 }, (_, i) => (
            <div key={i} className="skeleton" />
          ))}
        </div>
      ) : !s?.settings.images_dir ? (
        <div className="empty">
          <h2 className="empty__title">Choose the photo folder</h2>
          <p>Point the app at the folder of wound photos. Each photo gets its marking saved as a JSON file with the same name.</p>
          <a className="btn btn--primary" href="#/setup">
            Choose folders
          </a>
        </div>
      ) : items.length === 0 ? (
        <div className="empty">
          <h2 className="empty__title">No photos in this folder</h2>
          <p>The folder has no JPG, PNG, WebP, BMP or TIFF files. Pick another one.</p>
          <a className="btn" href="#/setup">
            Change folder
          </a>
        </div>
      ) : ordered.length === 0 ? (
        <div className="empty">
          <p>Nothing here right now.</p>
          <button type="button" className="btn" onClick={() => setF("all")}>
            Show all photos
          </button>
        </div>
      ) : (
        <div className="thumbs">
          {ordered.map((i) => (
            <a key={i.path} className="thumb" href={editHash(i.path)} onClick={(e) => (e.preventDefault(), go(editHash(i.path)))}>
              <div className="thumb__pic">
                <img src={imageUrl(i.path, 480)} alt="" loading="lazy" width={480} height={360} />
                {i.outline && i.size ? (
                  <svg viewBox={`0 0 ${i.size[0]} ${i.size[1]}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
                    {i.outline.map((pts, k) => (
                      <polygon key={k} points={pts.map((p) => p.join(",")).join(" ")} />
                    ))}
                  </svg>
                ) : null}
                {i.status === "reviewed" ? (
                  <span className="thumb__tag thumb__tag--love">
                    <Heart size={14} /> Approved
                  </span>
                ) : i.status === "auto" ? (
                  <span className={`thumb__tag ${i.needs_review ? "thumb__tag--look" : ""}`}>
                    <Smiley confidence={i.confidence} size={16} threshold={threshold} />
                    {i.confidence != null ? `${Math.round(i.confidence * 100)}%` : ""}
                  </span>
                ) : null}
              </div>
              <span className="thumb__name">{i.name}</span>
              <span className="thumb__meta">
                {i.status === "reviewed"
                  ? `${i.regions} ${i.regions === 1 ? "area" : "areas"}, ${i.approved_by === "auto" ? "approved automatically" : "approved"}`
                  : i.status === "auto"
                    ? i.regions === 0
                      ? "No wound found, please check"
                      : i.quality === "blurry"
                        ? QUALITY_TEXT.blurry
                        : i.review && !i.review.ok
                          ? "AI flagged this one"
                          : `${i.regions} ${i.regions === 1 ? "area" : "areas"} marked${i.ai ? " with the AI" : ""}`
                    : i.status === "foreign"
                      ? "Has a JSON from another tool"
                      : "Not marked yet"}
              </span>
            </a>
          ))}
        </div>
      )}
    </main>
  );
}
