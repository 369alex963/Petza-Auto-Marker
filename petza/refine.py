"""AI-guided marking: the local model draws, the online AI decides.

Vision AIs judge a photo well but give imprecise coordinates; the local model is
pixel-precise but cannot read the written rules. So per photo:

1. The local model's probability map gives candidate regions, each with up to
   three precise outlines: A tight, B normal (the local result), C wide.
2. The AI sees the photo and the numbered options and picks, by the prompt rules,
   which regions are wound and which outline follows the true edge. It may point
   at a missed wound with a rough box; the local model then outlines it precisely.
3. With 2 or 3 runs, every decision is a majority vote.
4. Any failure keeps the local result. The AI never draws points.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from . import quality, reviewer
from .geometry import mask_to_polygons, polygons_to_mask
from .infer import predict_window
from .rules import Rules
from .schema import Region

MAX_SIDE = 1568      # longest side of what is sent (the providers' sweet spot)
FIND = 0.25          # regions are searched at this probability, so faint wounds are candidates too
TIGHT, WIDE = 0.7, 0.3
SURE_EDGE = 0.75     # tight area / wide area at or above this: the local edge is sharp, keep it
MISSED = 0.2         # threshold inside a box the AI says holds a missed wound
MAX_CANDIDATES = 8   # more would crowd the picture; the rest keep the local result
OPTION_NAMES = {"A": "tight", "B": "normal", "C": "wide"}
COLORS = {"A": (40, 120, 255), "B": (255, 225, 0), "C": (255, 255, 255)}  # RGB: blue, yellow, white

PROMPT = """You are improving an automatic wound marking on a medical photo, following the clinician's rules.

Image 1 is the photo.
Image 2 shows the candidate regions a precise local segmentation model found{zoom}. Each region has a white number
and up to three outline options, all precise to the pixel:
  A (blue) tight: only the most certain wound pixels.
  B (yellow) normal: the model's own result.
  C (white) wide: also the uncertain border around it.
Options available per region: {listing}

The clinician's rules:
---
{rules}
---

For every numbered region decide, by the rules and what you see in image 1:
- keep: true only if it is wound by the rules; false if it is not (for example dry skin, healed skin, plain red skin,
  the pink outer rim, a shadow, a reflection or not a wound at all).
- outline: the listed option that follows the true wound edge best by the rules. Only pick an option listed for that region.
- label: one of {labels}.
- reason: a few words.
If image 1 shows a wound that has no candidate at all, add it to missed with a rough box [x0, y0, x1, y1]
as fractions from 0 to 1 of image 1's width and height. Otherwise leave missed empty.
Write reason and notes briefly, in the language the clinician's rules are written in."""


class RegionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema for OpenAI
    id: int
    keep: bool
    outline: Literal["A", "B", "C"]
    label: str
    reason: str


class MissedWound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    box: list[float]
    reason: str


class ChoiceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    regions: list[RegionChoice]
    missed: list[MissedWound]
    notes: str


@dataclass
class Candidate:
    id: int
    label: str
    outlines: dict[str, list[list[list[float]]]]  # option -> polygons (a region may split at a stricter threshold)
    area: float
    edge: float = 0.0  # tight area / wide area: near 1 means the local model is sure where the edge is

    @property
    def default(self) -> tuple[bool, str]:
        """What the local model alone does: keep it if it has a normal outline."""
        return bool(self.outlines.get("B")), "B"

    def options(self) -> list[str]:
        return [o for o in "ABC" if self.outlines.get(o)]


@dataclass
class Result:
    regions: list[Region]
    confidence: float
    quality: str
    ai: dict | None = None
    errors: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    options_jpeg: bytes | None = None
    local_regions: list[Region] = field(default_factory=list)


# ---------- candidates ----------

def candidates(probs: np.ndarray, classes: list[str], rules: Rules, spacing: float | None) -> list[Candidate]:
    h, w = probs.shape[:2]
    fg = 1 - probs[..., 0]
    normal = probs.argmax(-1) > 0
    poly = dict(min_area_px=rules.min_area_for(h, w), simplify_tolerance_px=rules.tolerance_for(h, w), point_spacing_px=spacing)
    n, comp, stats, _ = cv2.connectedComponentsWithStats((fg >= FIND).astype(np.uint8))
    found = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < poly["min_area_px"]:
            continue
        x, y, bw, bh = stats[i, :4]
        win = (slice(y, y + bh), slice(x, x + bw))  # work inside the region's box: fast on big photos
        region = comp[win] == i

        def outline(mask):
            return [[[px + x, py + y] for px, py in p] for p in mask_to_polygons(mask, **poly)]

        outlines = {"A": outline(region & (fg[win] >= TIGHT)), "B": outline(region & normal[win]), "C": outline(region & (fg[win] >= WIDE))}
        if not any(outlines.values()):
            continue
        cls = int(probs[win][region][:, 1:].mean(0).argmax())
        tight, wide = int((region & (fg[win] >= TIGHT)).sum()), int((region & (fg[win] >= WIDE)).sum())
        found.append(Candidate(0, classes[cls] if cls < len(classes) else "wound", outlines, float(stats[i, cv2.CC_STAT_AREA]),
                               tight / wide if wide else 0.0))
    found.sort(key=lambda c: -c.area)
    for k, c in enumerate(found):
        c.id = k + 1
    return found


# ---------- what is sent ----------

def _jpeg(rgb: np.ndarray) -> bytes:
    """Fresh JPEG from pixels only: no EXIF, GPS, date, camera or file name."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def _fit(rgb: np.ndarray, side: int = MAX_SIDE) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    s = min(1.0, side / max(h, w))
    return (cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else rgb.copy()), s


def render_options(rgb: np.ndarray, cands: list[Candidate]) -> tuple[bytes, bool]:
    """The photo with every candidate's numbered A/B/C outlines. Zoomed in when
    the regions are small, so the AI sees the edges. Returns (jpeg, zoomed)."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1, zoomed = 0, 0, w, h, False
    pts_all = [p for c in cands for o in c.outlines.values() for poly in o for p in poly]
    if pts_all:
        a = np.asarray(pts_all)
        bx0, by0, bx1, by1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
        mx, my = max(64, 0.3 * (bx1 - bx0)), max(64, 0.3 * (by1 - by0))
        cx0, cy0, cx1, cy1 = int(max(0, bx0 - mx)), int(max(0, by0 - my)), int(min(w, bx1 + mx)), int(min(h, by1 + my))
        if (cx1 - cx0) * (cy1 - cy0) < 0.5 * w * h:
            x0, y0, x1, y1, zoomed = cx0, cy0, cx1, cy1, True
    crop = rgb[y0:y1, x0:x1]
    s = min(3.0, MAX_SIDE / max(crop.shape[:2]))
    view = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    view = np.ascontiguousarray(view)
    thick = max(2, round(max(view.shape[:2]) / 600))
    for opt in "CBA":  # tight on top
        for c in cands:
            for poly in c.outlines.get(opt) or []:
                p = ((np.asarray(poly) - [x0, y0]) * s).round().astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(view, [p], True, COLORS[opt], thick, cv2.LINE_AA)
    for c in cands:
        pts = np.asarray([p for o in c.outlines.values() for poly in o for p in poly])
        tx, ty = int((pts[:, 0].min() - x0) * s), int(max(28, (pts[:, 1].min() - y0) * s - 10))
        for color, t in (((0, 0, 0), thick + 4), ((255, 255, 255), thick)):
            cv2.putText(view, str(c.id), (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, t, cv2.LINE_AA)
    return _jpeg(view), zoomed


def build_question(sent_rgb: np.ndarray, cands: list[Candidate], rules_text: str, labels: list[str]) -> tuple[list[bytes], str]:
    clean, _ = _fit(sent_rgb)
    options, zoomed = render_options(sent_rgb, cands)
    listing = "; ".join(f"region {c.id}: {', '.join(c.options())}" for c in cands) or "none (no candidates were found)"
    prompt = PROMPT.format(
        zoom=" (zoomed in on the regions)" if zoomed else "",
        listing=listing,
        rules=rules_text.strip() or "(no written rules: mark the full visible wound edge)",
        labels=", ".join(labels or ["wound"]),
    )
    return [_jpeg(clean), options], prompt


# ---------- deciding ----------

def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _clean_box(box) -> list[float] | None:
    if len(box) != 4:
        return None
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in box)
    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
    return [x0, y0, x1, y1] if (x1 - x0) > 0.005 and (y1 - y0) > 0.005 else None


def vote(answers: list[ChoiceResult], cands: list[Candidate], labels: list[str]) -> tuple[list[dict], list[list[float]]]:
    """Majority per region (a tie keeps the local result) and missed boxes that
    more than half of the answers agree on."""
    n = len(answers)
    decisions = []
    for c in cands:
        votes = []
        for a in answers:
            v = next((r for r in a.regions if r.id == c.id), None)
            keep, opt = (v.keep, v.outline) if v else c.default
            if opt not in c.options():
                opt = next(o for o in ("B", "C", "A") if o in c.options())
            votes.append((keep, opt, v.label if v and v.label in labels else c.label, v.reason if v else ""))
        yes = sum(1 for k, *_ in votes if k)
        keep = c.default[0] if yes * 2 == n else yes * 2 > n
        keepers = [v for v in votes if v[0] == keep] or votes
        ranked = Counter(o for _, o, _, _ in keepers).most_common()
        best = [o for o, k in ranked if k == ranked[0][1]]
        opt = "B" if "B" in best else best[0]
        label = Counter(lb for _, _, lb, _ in keepers).most_common(1)[0][0]
        reason = next((r for k, o, _, r in keepers if o == opt and r), next((r for *_, r in keepers if r), ""))
        decisions.append({"id": c.id, "keep": keep, "outline": opt, "label": label, "reason": reason,
                          "agreed": f"{sum(1 for v in votes if v[0] == keep and (not keep or v[1] == opt))}/{n}"})
    boxes = [(i, b) for i, a in enumerate(answers) for m in a.missed if (b := _clean_box(m.box))]
    accepted: list[list[float]] = []
    for i, b in boxes:
        support = {j for j, o in boxes if _box_iou(b, o) > 0.3}
        if len(support) * 2 > n and not any(_box_iou(b, x) > 0.3 for x in accepted):
            accepted.append(b)
    return decisions, accepted


def _outline_missed(run, size: int, img: np.ndarray, box, taken: np.ndarray, classes, rules: Rules, spacing) -> list[Region]:
    """Run the local model again on a box the AI pointed at, and outline what it finds there."""
    h, w = img.shape[:2]
    bx0, by0, bx1, by1 = int(box[0] * w), int(box[1] * h), int(math.ceil(box[2] * w)), int(math.ceil(box[3] * h))
    mx, my = max(48, (bx1 - bx0) // 4), max(48, (by1 - by0) // 4)
    x0, y0, x1, y1 = max(0, bx0 - mx), max(0, by0 - my), min(w, bx1 + mx), min(h, by1 + my)
    probs = predict_window(run, img[y0:y1, x0:x1], size, tta=True)
    fg = 1 - probs[..., 0]
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = fg >= MISSED
    inside = np.zeros((h, w), bool)
    inside[by0:by1, bx0:bx1] = True
    mask &= inside & ~taken
    polys = mask_to_polygons(mask, min_area_px=rules.min_area_for(h, w), simplify_tolerance_px=rules.tolerance_for(h, w), point_spacing_px=spacing)
    out = []
    for p in polys:
        m = polygons_to_mask([p], h, w)[y0:y1, x0:x1]
        cls = int(probs[m][:, 1:].mean(0).argmax()) if m.any() else 0
        out.append(Region(id="", label=classes[cls] if cls < len(classes) else "wound", points=p,
                          confidence=round(float(fg[m].mean()) if m.any() else 0.0, 4)))
    return out


def _confidence(probs: np.ndarray, regions: list[Region], h: int, w: int) -> list[Region]:
    fg = 1 - probs[..., 0]
    for r in regions:
        if r.confidence is None:
            m = polygons_to_mask([r.points], h, w)
            r.confidence = round(float(fg[m].mean()) if m.any() else 0.0, 4)
    return regions


# ---------- one photo, start to end ----------

def mark(pred, img: np.ndarray, rules: Rules, s=None, use_ai: bool = False, keep_views: bool = False) -> Result:
    """The full marking of one photo: quality check, local model, and (when
    `use_ai`) the AI's choices. Used by "Mark all", "Mark this photo again" and
    "Try one photo", so all three behave the same."""
    status = quality.check(img)
    local, conf, probs = pred.predict_full(img, rules)
    result = Result(regions=local, confidence=conf, quality=status, local_regions=local)
    if not use_ai or s is None:
        return result
    started = time.time()
    h, w = img.shape[:2]
    spacing = rules.point_spacing_px or (pred.meta.get("spacing_frac") * (h * h + w * w) ** 0.5 if pred.meta.get("spacing_frac") else None)
    cands = candidates(probs, pred.classes, rules, spacing)
    asked, rest = cands[:MAX_CANDIDATES], cands[MAX_CANDIDATES:]
    corrected = quality.needs_correction(status)
    sent = quality.correct(img) if corrected else img
    images, prompt = build_question(sent, asked, rules.text, pred.classes)
    result.candidates, result.options_jpeg = (cands, images[1]) if keep_views else ([], None)
    runs = max(1, min(3, int(getattr(s, "ai_runs", 1) or 1)))
    answers, vias = [], []
    for k in range(runs):
        answer, via, errors = reviewer.ask(images, prompt, ChoiceResult, s, salt=f"run{k}")
        result.errors += errors
        if answer is None:
            break
        answers.append(answer)
        vias.append(via)
    if len(answers) < runs:  # a failed run: keep the local result, never half a vote
        result.ai = {"ok": False, "errors": result.errors[-3:], "seconds": round(time.time() - started, 1)}
        return result
    decisions, boxes = vote(answers, asked, pred.classes)
    for c, d in zip(asked, decisions):
        # Measured on real diabetic photos (held-out professional markings): the AI's
        # tighter or wider outline was less exact than the local edge, so by default
        # the AI decides wound or not and finds missed wounds, and the edge stays local.
        # With ai_edges on, it may move the edge only where the local edge is unsure.
        free = getattr(s, "ai_edges", False) and c.edge < SURE_EDGE
        if d["keep"] and d["outline"] != "B" and not free and c.outlines.get("B"):
            d["outline"], d["kept_local_edge"] = "B", True
    chosen: list[Region] = []
    for c, d in zip(asked, decisions):
        if d["keep"]:
            chosen += [Region(id="", label=d["label"], points=p) for p in c.outlines[d["outline"]]]
    for c in rest:
        chosen += [Region(id="", label=c.label, points=p) for p in c.outlines.get("B") or []]
    taken = polygons_to_mask([r.points for r in chosen], h, w) if chosen else np.zeros((h, w), bool)
    added = 0
    for b in boxes:
        found = _outline_missed(pred.run, pred.size, img, b, taken, pred.classes, rules, spacing)
        added += len(found)
        chosen += found
        if found:
            taken |= polygons_to_mask([r.points for r in found], h, w)
    chosen = _confidence(probs, chosen, h, w)
    chosen.sort(key=lambda r: -len(r.points))
    if rules.max_regions:
        chosen = chosen[: rules.max_regions]
    for i, r in enumerate(chosen):
        r.id = f"r{i + 1}"
    result.regions = chosen
    result.confidence = round(min((r.confidence or 0) for r in chosen), 4) if chosen else conf
    result.ai = {
        "ok": True, "via": vias[0], "runs": runs, "choices": decisions, "missed": added,
        "notes": next((a.notes for a in answers if a.notes), ""), "corrected": corrected,
        "seconds": round(time.time() - started, 1),
    }
    return result


def test_question() -> tuple[list[bytes], str, type[BaseModel]]:
    """A tiny real question for Test connection: one round sore on plain skin."""
    from . import prompts

    img = np.full((360, 360, 3), (205, 160, 140), np.uint8)
    cv2.circle(img, (180, 180), 60, (150, 40, 50), -1)
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = lambda r: [[[180 + r * math.cos(a), 180 + r * math.sin(a)] for a in t]]  # noqa: E731
    cand = Candidate(1, "wound", {"A": ring(50), "B": ring(61), "C": ring(75)}, 1.0)
    images, prompt = build_question(img, [cand], prompts.active_rules().text, ["wound"])
    return images, prompt, ChoiceResult
