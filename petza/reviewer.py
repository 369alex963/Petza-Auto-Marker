"""Optional second opinion from Claude: is this marking right by the rules?

Local-first: with no API key, no network or any API error this returns None
and the app carries on. It never moves points; it only flags images for
Lauren to look at.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from pydantic import BaseModel

from .schema import Marking

MAX_SIDE = 1568

PROMPT = """You are checking an automatic wound marking on a medical photo.
The outline drawn in bright green is the automatic marking (one closed polygon per marked region, label written next to it).

Marking rules from the clinician:
---
{rules}
---

Check the marking against the rules and the photo. Report a problem only when you can see it, for example:
the outline cuts into the wound or leaves part of it out, it includes healthy skin, a wound is missed,
something that is not a wound is marked, or the label is wrong.
Set ok to true when the marking is acceptable as is."""


class ReviewResult(BaseModel):
    ok: bool
    problems: list[str]
    summary: str


def overlay(img: np.ndarray, marking: Marking) -> bytes:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    thick = max(2, round(max(small.shape[:2]) / 500))
    for r in marking.regions:
        pts = (np.asarray(r.points) * scale).round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(bgr, [pts], True, (0, 255, 0), thick, cv2.LINE_AA)
        x, y = pts[:, 0, 0].min(), max(20, pts[:, 0, 1].min() - 8)
        cv2.putText(bgr, r.label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def review(img: np.ndarray, marking: Marking, rules_text: str, api_key: str, model: str) -> ReviewResult | None:
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=120, max_retries=2)
        data = base64.standard_b64encode(overlay(img, marking)).decode("utf-8")
        response = client.messages.parse(
            model=model,
            max_tokens=4000,
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": PROMPT.format(rules=rules_text.strip() or "(no written rules: mark the full visible wound edge)")},
            ]}],
            output_format=ReviewResult,
        )
        if response.stop_reason == "refusal":
            return None
        return response.parsed_output
    except Exception:
        return None  # offline, bad key, rate limit...: the local result stands
