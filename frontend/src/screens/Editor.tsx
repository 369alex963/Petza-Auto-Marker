import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { editHash, go, type AppCtx } from "../App";
import { api, imageUrl, type ImageItem, type Marking, type Point, type Region } from "../api";
import { ChevronLeft, ChevronRight, Cursor, Eye, Fit, Heart, Lasso, Redo, Smiley, Trash, Undo } from "../icons";

type Tool = "edit" | "lasso";
type View = { x: number; y: number; w: number; h: number };
type Drag = { kind: "vertex"; region: number; index: number } | { kind: "pan"; sx: number; sy: number; view: View } | null;

const round1 = (n: number) => Math.round(n * 10) / 10;

/** Keep receiving moves while dragging, even outside the element. Never fatal. */
const capture = (e: React.PointerEvent) => {
  try {
    (e.target as Element).setPointerCapture(e.pointerId);
  } catch {
    /* pointer already released */
  }
};

export function Editor({ path, ctx }: { path: string; ctx: AppCtx }) {
  const [marking, setMarking] = useState<Marking | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [regions, setRegions] = useState<Region[]>([]);
  const [past, setPast] = useState<Region[][]>([]);
  const [future, setFuture] = useState<Region[][]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [tool, setTool] = useState<Tool>("edit");
  const [draft, setDraft] = useState<Point[]>([]);
  const [view, setView] = useState<View | null>(null);
  const [drag, setDrag] = useState<Drag>(null);
  const [hover, setHover] = useState<Point | null>(null);
  const [hideMarks, setHideMarks] = useState(false);
  const [spaceDown, setSpaceDown] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [imgReady, setImgReady] = useState(false);
  const [list, setList] = useState<ImageItem[]>([]);
  const svgRef = useRef<SVGSVGElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const dragStartRegions = useRef<Region[] | null>(null);

  const s = ctx.status;
  const threshold = s?.rules.review_below_confidence ?? 0.75;
  const labels = useMemo(() => {
    const set = new Set<string>([...(s?.rules.labels ?? []), ...(s?.model?.classes ?? []), ...(s?.pool.labels ?? []), ...regions.map((r) => r.label)]);
    if (set.size === 0) set.add("wound");
    return [...set];
  }, [s, regions]);

  // ---------- load ----------
  useEffect(() => {
    let alive = true;
    api
      .marking(path)
      .then((r) => {
        if (!alive) return;
        setMarking(r.marking);
        setRegions(r.marking.regions);
        setSelected(r.marking.regions.length ? 0 : null);
        if (!r.marking.regions.length) setTool("lasso");
      })
      .catch((e: Error) => alive && setError(e.message));
    api.images().then((r) => alive && setList(r.items)).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [path]);

  const fit = useCallback(() => {
    if (!marking || !stageRef.current) return;
    const { clientWidth: cw, clientHeight: ch } = stageRef.current;
    const scale = Math.min(cw / marking.width, ch / marking.height) * 0.94;
    const w = cw / scale;
    const h = ch / scale;
    setView({ x: (marking.width - w) / 2, y: (marking.height - h) / 2, w, h });
  }, [marking]);

  useEffect(() => {
    fit();
    const ro = new ResizeObserver(() => fit());
    if (stageRef.current) ro.observe(stageRef.current);
    return () => ro.disconnect();
  }, [fit]);

  // ---------- history ----------
  const commit = useCallback(
    (next: Region[], before?: Region[]) => {
      setPast((p) => [...p.slice(-99), before ?? regions]);
      setFuture([]);
      setRegions(next);
      setDirty(true);
    },
    [regions],
  );
  const undo = useCallback(() => {
    if (!past.length) return;
    setFuture([regions, ...future]);
    setRegions(past[past.length - 1]);
    setPast(past.slice(0, -1));
    setSelected((s) => (s != null && s < past[past.length - 1].length ? s : null));
    setDirty(true);
  }, [past, future, regions]);
  const redo = useCallback(() => {
    if (!future.length) return;
    setPast([...past, regions]);
    setRegions(future[0]);
    setFuture(future.slice(1));
    setSelected((s) => (s != null && s < future[0].length ? s : null));
    setDirty(true);
  }, [past, future, regions]);

  // ---------- geometry helpers ----------
  const toImage = useCallback((clientX: number, clientY: number): Point => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [p.x, p.y];
  }, []);
  const clamp = useCallback(
    ([x, y]: Point): Point => [round1(Math.min(Math.max(x, 0), marking?.width ?? x)), round1(Math.min(Math.max(y, 0), marking?.height ?? y))],
    [marking],
  );
  const pxPerUnit = view && stageRef.current ? stageRef.current.clientWidth / view.w : 1;
  const handleR = 6 / pxPerUnit;

  // ---------- pointer handling ----------
  const onWheel = (e: React.WheelEvent) => {
    if (!view) return;
    const [mx, my] = toImage(e.clientX, e.clientY);
    const k = Math.exp(e.deltaY * 0.0015);
    const w = Math.min(Math.max(view.w * k, 20), (marking?.width ?? 1000) * 8);
    const h = (view.h / view.w) * w;
    setView({ x: mx - ((mx - view.x) * w) / view.w, y: my - ((my - view.y) * h) / view.h, w, h });
  };

  const startPan = (e: React.PointerEvent) => {
    if (!view) return;
    capture(e);
    setDrag({ kind: "pan", sx: e.clientX, sy: e.clientY, view });
  };

  const onBackgroundDown = (e: React.PointerEvent) => {
    if (e.button === 1 || spaceDown || (e.button === 0 && tool === "edit")) {
      if (e.button === 0 && tool === "edit" && !spaceDown) setSelected(null);
      startPan(e);
      return;
    }
    if (e.button === 0 && tool === "lasso") addDraftPoint(e);
  };

  const addDraftPoint = (e: React.PointerEvent) => {
    const p = clamp(toImage(e.clientX, e.clientY));
    if (draft.length >= 3) {
      const [fx, fy] = draft[0];
      if (Math.hypot(p[0] - fx, p[1] - fy) < handleR * 2) {
        closeDraft();
        return;
      }
    }
    setDraft((d) => [...d, p]);
  };

  const closeDraft = useCallback(() => {
    if (draft.length < 3) return;
    const label = (selected != null && regions[selected]?.label) || labels[0] || "wound";
    const next = [...regions, { id: `r${Date.now().toString(36)}`, label, points: draft, confidence: null, source: "manual" as const }];
    commit(next);
    setSelected(next.length - 1);
    setDraft([]);
    setTool("edit");
  }, [draft, regions, selected, labels, commit]);

  const onVertexDown = (e: React.PointerEvent, region: number, index: number) => {
    e.stopPropagation();
    if (e.button === 2) return;
    if (tool === "lasso" || spaceDown) {
      if (tool === "lasso") addDraftPoint(e);
      return;
    }
    capture(e);
    dragStartRegions.current = regions;
    setSelected(region);
    setDrag({ kind: "vertex", region, index });
  };

  const onVertexMenu = (e: React.MouseEvent, region: number, index: number) => {
    e.preventDefault();
    const r = regions[region];
    if (r.points.length <= 3) {
      ctx.toast("An outline needs at least 3 points. Delete the whole area instead.");
      return;
    }
    commit(regions.map((x, i) => (i === region ? { ...x, points: x.points.filter((_, j) => j !== index), source: "manual" } : x)));
  };

  const onEdgeDown = (e: React.PointerEvent, region: number, index: number) => {
    e.stopPropagation();
    if (tool === "lasso" || spaceDown || e.button !== 0) {
      if (tool === "lasso") addDraftPoint(e);
      else if (spaceDown) startPan(e);
      return;
    }
    const p = clamp(toImage(e.clientX, e.clientY));
    const before = regions;
    const next = regions.map((x, i) => (i === region ? { ...x, points: [...x.points.slice(0, index + 1), p, ...x.points.slice(index + 1)], source: "manual" as const } : x));
    setPast((h) => [...h.slice(-99), before]);
    setFuture([]);
    setRegions(next);
    setDirty(true);
    dragStartRegions.current = null; // the insert is already in history
    capture(e);
    setSelected(region);
    setDrag({ kind: "vertex", region, index: index + 1 });
  };

  const onPolyDown = (e: React.PointerEvent, region: number) => {
    if (tool === "lasso") return; // let the background add the point
    e.stopPropagation();
    if (spaceDown || e.button === 1) {
      startPan(e);
      return;
    }
    setSelected(region);
  };

  const onMove = (e: React.PointerEvent) => {
    if (tool === "lasso") setHover(toImage(e.clientX, e.clientY));
    if (!drag || !view) return;
    if (drag.kind === "pan") {
      const k = drag.view.w / (stageRef.current?.clientWidth ?? 1);
      setView({ ...drag.view, x: drag.view.x - (e.clientX - drag.sx) * k, y: drag.view.y - (e.clientY - drag.sy) * k });
      return;
    }
    const p = clamp(toImage(e.clientX, e.clientY));
    setRegions((rs) => rs.map((r, i) => (i === drag.region ? { ...r, points: r.points.map((q, j) => (j === drag.index ? p : q)), source: "manual" } : r)));
  };

  const onUp = () => {
    const before = dragStartRegions.current;
    if (drag?.kind === "vertex" && before && before !== regions) {
      setPast((h) => [...h.slice(-99), before]);
      setFuture([]);
      setDirty(true);
    }
    dragStartRegions.current = null;
    setDrag(null);
  };

  // ---------- regions panel ----------
  const deleteRegion = useCallback(
    (i: number) => {
      commit(regions.filter((_, j) => j !== i));
      setSelected(null);
    },
    [regions, commit],
  );
  const setLabel = (i: number, label: string) => commit(regions.map((r, j) => (j === i ? { ...r, label, source: "manual" } : r)));

  // ---------- save & navigate ----------
  const index = list.findIndex((i) => i.path === path);
  const prev = index > 0 ? list[index - 1] : null;
  const next = index >= 0 && index < list.length - 1 ? list[index + 1] : null;

  const save = useCallback(
    async (thenNext: boolean) => {
      if (!marking) return;
      setSaving(true);
      try {
        await api.saveMarking(path, { ...marking, regions, status: "reviewed" });
        setDirty(false);
        ctx.toast(regions.length ? "Approved. The app learns from this one on the next Teach." : "Saved as no wound.");
        if (thenNext && next) go(editHash(next.path));
      } catch (e) {
        ctx.toast(`Could not save: ${(e as Error).message}`);
      } finally {
        setSaving(false);
      }
    },
    [marking, regions, path, next, ctx],
  );

  const leave = (hash: string) => {
    if (dirty && !window.confirm("You have changes that are not saved. Leave this photo anyway?")) return;
    go(hash);
  };

  // ---------- keyboard ----------
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest?.("input, select, textarea");
      if (typing) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void save(false);
      } else if (mod && e.key.toLowerCase() === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (mod && (e.key.toLowerCase() === "y" || (e.key.toLowerCase() === "z" && e.shiftKey))) {
        e.preventDefault();
        redo();
      } else if (e.key === " ") {
        e.preventDefault();
        setSpaceDown(true);
      } else if (e.key === "Enter") {
        if (tool === "lasso") closeDraft();
        else void save(true);
      } else if (e.key === "Escape") {
        if (draft.length) setDraft([]);
        else setTool("edit");
      } else if (e.key === "Backspace" && tool === "lasso" && draft.length) {
        setDraft((d) => d.slice(0, -1));
      } else if ((e.key === "Delete" || e.key === "Backspace") && selected != null) {
        deleteRegion(selected);
      } else if (e.key.toLowerCase() === "l") {
        setTool("lasso");
      } else if (e.key.toLowerCase() === "v") {
        setTool("edit");
        setDraft([]);
      } else if (e.key.toLowerCase() === "h") {
        setHideMarks((h) => !h);
      } else if (e.key === "0" || e.key.toLowerCase() === "f") {
        fit();
      } else if (e.key === "ArrowRight" && next) {
        leave(editHash(next.path));
      } else if (e.key === "ArrowLeft" && prev) {
        leave(editHash(prev.path));
      }
    };
    const up = (e: KeyboardEvent) => e.key === " " && setSpaceDown(false);
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  });

  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirty) e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  // ---------- render ----------
  if (error)
    return (
      <main className="page">
        <div className="note note--error" role="alert">
          <p>This photo could not be opened: {error}</p>
          <a className="btn" href="#/mark">
            Back to photos
          </a>
        </div>
      </main>
    );

  const conf = marking?.status === "reviewed" ? null : marking?.confidence ?? null;
  const mood = conf == null ? "" : conf >= threshold ? "mood--happy" : "";
  const stageClass = `stage ${tool === "lasso" && !spaceDown ? "stage--lasso" : ""} ${spaceDown ? "stage--pan" : ""} ${drag?.kind === "pan" ? "stage--panning" : ""}`;

  return (
    <main className="editor">
      <div className={stageClass} ref={stageRef}>
        <div className="stage__tools" role="toolbar" aria-label="Marking tools">
          <button type="button" className="btn btn--small btn--icon btn--quiet" aria-pressed={tool === "edit"} title="Move points (V)" aria-label="Move points" onClick={() => (setTool("edit"), setDraft([]))}>
            <Cursor />
          </button>
          <button type="button" className="btn btn--small btn--icon btn--quiet" aria-pressed={tool === "lasso"} title="Point lasso: draw a new area (L)" aria-label="Point lasso" onClick={() => setTool("lasso")}>
            <Lasso />
          </button>
          <button type="button" className="btn btn--small btn--icon btn--quiet" title="Undo (Ctrl+Z)" aria-label="Undo" disabled={!past.length} onClick={undo}>
            <Undo />
          </button>
          <button type="button" className="btn btn--small btn--icon btn--quiet" title="Redo (Ctrl+Y)" aria-label="Redo" disabled={!future.length} onClick={redo}>
            <Redo />
          </button>
          <button type="button" className="btn btn--small btn--icon btn--quiet" aria-pressed={hideMarks} title="Hide the marking to see the wound (H)" aria-label="Hide marking" onClick={() => setHideMarks((h) => !h)}>
            <Eye />
          </button>
          <button type="button" className="btn btn--small btn--icon btn--quiet" title="Fit to screen (F)" aria-label="Fit to screen" onClick={fit}>
            <Fit />
          </button>
        </div>
        {!imgReady ? <div className="stage__loading">Opening the photo</div> : null}
        {marking && view ? (
          <svg
            ref={svgRef}
            viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
            className={hideMarks ? "hidden-marks" : undefined}
            onWheel={onWheel}
            onPointerDown={onBackgroundDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerLeave={() => setHover(null)}
            onDoubleClick={() => tool === "lasso" && closeDraft()}
            onContextMenu={(e) => e.preventDefault()}
            role="application"
            aria-label={`Photo ${marking.image} with ${regions.length} marked ${regions.length === 1 ? "area" : "areas"}`}
          >
            <image href={imageUrl(path)} x={0} y={0} width={marking.width} height={marking.height} onLoad={() => setImgReady(true)} preserveAspectRatio="none" />
            {regions.map((r, ri) => (
              <polygon key={r.id + ri} points={r.points.map((p) => p.join(",")).join(" ")} className={`poly ${selected === ri ? "poly--sel" : ""}`} onPointerDown={(e) => onPolyDown(e, ri)} />
            ))}
            {selected != null && regions[selected] && tool === "edit" ? (
              <g>
                {regions[selected].points.map((p, i, pts) => {
                  const q = pts[(i + 1) % pts.length];
                  return <line key={`e${i}`} className="edge" x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]} onPointerDown={(e) => onEdgeDown(e, selected, i)} />;
                })}
                {regions[selected].points.map((p, i) => (
                  <circle
                    key={`v${i}`}
                    className={`vtx ${drag?.kind === "vertex" && drag.index === i ? "vtx--sel" : ""}`}
                    cx={p[0]}
                    cy={p[1]}
                    r={handleR}
                    onPointerDown={(e) => onVertexDown(e, selected, i)}
                    onContextMenu={(e) => onVertexMenu(e, selected, i)}
                  />
                ))}
              </g>
            ) : null}
            {draft.length ? (
              <g>
                <polyline className="poly poly--draft" points={[...draft, ...(hover ? [hover] : [])].map((p) => p.join(",")).join(" ")} />
                {draft.map((p, i) => (
                  <circle key={i} className={`vtx ${i === 0 ? "vtx--first" : ""}`} cx={p[0]} cy={p[1]} r={i === 0 ? handleR * 1.4 : handleR * 0.8} />
                ))}
              </g>
            ) : null}
          </svg>
        ) : null}
        <p className="stage__hint">
          {tool === "lasso"
            ? draft.length < 3
              ? "Click around the wound edge to place points."
              : "Click the first point, double-click or press Enter to close the outline."
            : "Drag a point to move it. Click the outline to add a point, right-click a point to remove it. Space and drag to pan, wheel to zoom."}
        </p>
      </div>

      <aside className="side" aria-label="Marking details">
        <div className="side__nav">
          <button type="button" className="btn btn--small btn--icon" aria-label="Previous photo" disabled={!prev} onClick={() => prev && leave(editHash(prev.path))}>
            <ChevronLeft />
          </button>
          <span className="side__count">{index >= 0 ? `${index + 1} of ${list.length}` : ""}</span>
          <button type="button" className="btn btn--small btn--icon" aria-label="Next photo" disabled={!next} onClick={() => next && leave(editHash(next.path))}>
            <ChevronRight />
          </button>
        </div>
        <div>
          <div className="side__name">{marking?.image ?? "Loading"}</div>
          <div className="muted">{marking ? `${marking.width} by ${marking.height} pixels` : ""}</div>
        </div>

        {marking?.status === "reviewed" ? (
          <div className="mood mood--happy">
            <Heart size={20} /> Approved by Lauren
          </div>
        ) : marking && marking.model_version ? (
          <div className={`mood ${mood}`}>
            <Smiley confidence={conf} size={26} threshold={threshold} />
            {conf == null ? "Not sure" : conf >= threshold ? `Looks good, ${Math.round(conf * 100)}% sure` : `Please check, ${Math.round(conf * 100)}% sure`}
          </div>
        ) : marking ? (
          <div className="mood">Not marked yet. Draw the outline with the point lasso.</div>
        ) : null}

        {marking?.review ? (
          <div className="review">
            <strong>{marking.review.ok ? "AI second opinion: looks right" : "AI second opinion: please check"}</strong>
            {marking.review.problems.length ? (
              <ul>
                {marking.review.problems.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            ) : (
              <span>{marking.review.summary}</span>
            )}
          </div>
        ) : null}

        <hr className="rule" />
        <ul className="regions" aria-label="Marked areas">
          {regions.length === 0 ? <li className="muted">No areas yet. Press L and click around the wound.</li> : null}
          {regions.map((r, i) => (
            <li key={r.id + i} className="region" aria-selected={selected === i} onClick={() => setSelected(i)}>
              <span className="region__swatch" aria-hidden="true" />
              <div>
                <select className="input" aria-label={`Label of area ${i + 1}`} value={r.label} onChange={(e) => setLabel(i, e.target.value)} onClick={(e) => e.stopPropagation()}>
                  {labels.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
                <div className="region__meta">
                  {r.points.length} points{r.source === "auto" && r.confidence != null ? `, ${Math.round(r.confidence * 100)}% sure` : ", edited"}
                </div>
              </div>
              <button type="button" className="btn btn--small btn--icon btn--quiet" aria-label={`Delete area ${i + 1}`} onClick={(e) => (e.stopPropagation(), deleteRegion(i))}>
                <Trash />
              </button>
            </li>
          ))}
        </ul>
        <button type="button" className="btn btn--small" onClick={() => (setTool("lasso"), setDraft([]))}>
          <Lasso /> Draw another area
        </button>

        <div className="side__actions">
          <button type="button" className="btn btn--primary" disabled={saving || !marking} onClick={() => void save(true)}>
            <Heart /> {next ? "Approve and next" : "Approve"}
          </button>
          <button type="button" className="btn" disabled={saving || !marking || !dirty} onClick={() => void save(false)}>
            Save changes
          </button>
          <a className="btn btn--quiet" href="#/mark" onClick={(e) => (e.preventDefault(), leave("#/mark"))}>
            Back to all photos
          </a>
          <dl className="keys">
            <dt>
              <kbd>L</kbd> / <kbd>V</kbd>
            </dt>
            <dd>lasso / move points</dd>
            <dt>
              <kbd>Enter</kbd>
            </dt>
            <dd>close outline, or approve</dd>
            <dt>
              <kbd>Ctrl Z</kbd>
            </dt>
            <dd>undo</dd>
            <dt>
              <kbd>H</kbd>
            </dt>
            <dd>hide marking</dd>
            <dt>
              <kbd>Arrows</kbd>
            </dt>
            <dd>previous / next photo</dd>
          </dl>
        </div>
      </aside>
    </main>
  );
}
