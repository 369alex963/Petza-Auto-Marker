"""Optional second opinion from an online AI: is this marking right by the rules?

Two providers, Claude and ChatGPT (OpenAI). Each one is reached through the
account already signed in on this PC when possible (the Claude Code CLI, or the
OpenAI Codex CLI), and otherwise through an API key.

Local-first: with no connection, no network or any error this returns None and
the app carries on. It never moves points; it only flags images to look at.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from .schema import Marking, Region

MAX_SIDE = 1568
PROVIDERS = ("claude", "openai")
CLI_TIMEOUT = 300
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

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

FILE_NOTE = "\n\nThe photo with the marking is the file {name} in the current folder. Open it and look at it before you answer."


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")  # strict JSON schema for OpenAI
    ok: bool
    problems: list[str]
    summary: str


SCHEMA = ReviewResult.model_json_schema()


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


# ---------- connections ----------

@lru_cache(maxsize=None)
def _cli(name: str) -> str | None:
    return shutil.which(name)


def connections(s) -> dict:
    """What can be used right now, in the order it is tried."""
    claude = [m for m, ok in (("account", _cli("claude")), ("api", s.anthropic_api_key)) if ok]
    openai = [m for m, ok in (("account", _cli("codex")), ("api", s.openai_api_key)) if ok]
    order = claude if s.reviewer_provider == "claude" else openai
    return {"claude": claude, "openai": openai, "ready": bool(order)}


def _run(cmd: list[str], prompt: str, cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=prompt, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=CLI_TIMEOUT, creationflags=NO_WINDOW)


def _claude_account(jpeg: bytes, prompt: str, model: str) -> ReviewResult:
    """Claude Code CLI, signed in with the Claude account (no API key)."""
    with tempfile.TemporaryDirectory(prefix="wr-review-") as d:
        (Path(d) / "marking.jpg").write_bytes(jpeg)
        cmd = [_cli("claude"), "-p", "--output-format", "json", "--json-schema", json.dumps(SCHEMA),
               "--model", model, "--tools", "Read", "--allowedTools", "Read", "--setting-sources", "",  # skip plugins and hooks: 14 s instead of 2 min
               "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
        r = _run(cmd, prompt + FILE_NOTE.format(name="marking.jpg"), d)
        data = json.loads(r.stdout or "{}")
        if r.returncode or data.get("is_error"):
            raise RuntimeError((data.get("result") or r.stderr or "Claude did not answer").strip()[:300])
        out = data.get("structured_output") or json.loads(data["result"])
        return ReviewResult.model_validate(out)


def _openai_account(jpeg: bytes, prompt: str, model: str) -> ReviewResult:
    """OpenAI Codex CLI, signed in with the ChatGPT account (no API key)."""
    with tempfile.TemporaryDirectory(prefix="wr-review-") as d:
        img, schema, answer = Path(d) / "marking.jpg", Path(d) / "schema.json", Path(d) / "answer.json"
        img.write_bytes(jpeg)
        schema.write_text(json.dumps(SCHEMA), encoding="utf-8")
        cmd = [_cli("codex"), "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--image", str(img),
               "--output-schema", str(schema), "--output-last-message", str(answer)]
        if model:
            cmd += ["--model", model]
        r = _run(cmd + ["-"], prompt, d)
        if r.returncode or not answer.is_file():
            raise RuntimeError((r.stderr or r.stdout or "ChatGPT did not answer").strip()[-300:])
        return ReviewResult.model_validate_json(answer.read_text(encoding="utf-8"))


def _claude_api(jpeg: bytes, prompt: str, model: str, key: str) -> ReviewResult:
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=120, max_retries=2)
    response = client.messages.parse(
        model=model,
        max_tokens=4000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.standard_b64encode(jpeg).decode("utf-8")}},
            {"type": "text", "text": prompt},
        ]}],
        output_format=ReviewResult,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("Claude declined to review this photo.")
    return response.parsed_output


def _openai_api(jpeg: bytes, prompt: str, model: str, key: str) -> ReviewResult:
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=120, max_retries=2)
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.standard_b64encode(jpeg).decode("utf-8")},
        ]}],
        text_format=ReviewResult,
    )
    if response.output_parsed is None:
        raise RuntimeError("ChatGPT declined to review this photo.")
    return response.output_parsed


def _attempts(s):
    if s.reviewer_provider == "openai":
        if _cli("codex"):
            yield "ChatGPT account", lambda j, p: _openai_account(j, p, s.openai_model)
        if s.openai_api_key:
            yield "OpenAI API key", lambda j, p: _openai_api(j, p, s.openai_model, s.openai_api_key)
    else:
        if _cli("claude"):
            yield "Claude account", lambda j, p: _claude_account(j, p, s.reviewer_model)
        if s.anthropic_api_key:
            yield "Claude API key", lambda j, p: _claude_api(j, p, s.reviewer_model, s.anthropic_api_key)


def review_verbose(img: np.ndarray, marking: Marking, rules_text: str, s) -> tuple[ReviewResult | None, str | None, list[str]]:
    """Try the account first, then the API key. Returns (result, via, errors)."""
    jpeg = overlay(img, marking)
    prompt = PROMPT.format(rules=rules_text.strip() or "(no written rules: mark the full visible wound edge)")
    errors = []
    for via, call in _attempts(s):
        try:
            return call(jpeg, prompt), via, errors
        except Exception as e:  # offline, signed out, bad key, rate limit...
            errors.append(f"{via}: {str(e).strip()[:300] or type(e).__name__}")
    return None, None, errors


def review(img: np.ndarray, marking: Marking, rules_text: str, s) -> dict | None:
    result, via, _ = review_verbose(img, marking, rules_text, s)
    return {**result.model_dump(), "via": via} if result else None


def test(s) -> dict:
    """A tiny real request: a plain skin-coloured square with one marked circle."""
    img = np.full((320, 320, 3), (200, 150, 130), np.uint8)
    cv2.circle(img, (160, 160), 60, (150, 40, 50), -1)
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = [[160 + 62 * np.cos(a), 160 + 62 * np.sin(a)] for a in t]
    m = Marking(image="test.jpg", width=320, height=320, regions=[Region(id="r1", points=pts)])
    if not connections(s)["ready"]:
        return {"ok": False, "via": None, "errors": ["Nothing to connect with yet: sign in to the app on this PC, or add an API key."]}
    result, via, errors = review_verbose(img, m, "Trace the outer edge of the dark round wound.", s)
    return {"ok": result is not None, "via": via, "errors": errors}
