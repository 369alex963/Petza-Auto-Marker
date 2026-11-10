"""The online AI: how the app reaches Claude, ChatGPT or Groq.

Each provider is reached through the account signed in on this PC when possible
(the official Claude Code or Codex command-line app, the same one the official
VS Code extensions run), and otherwise through an API key. Groq is API key only.

`ask` sends images + a prompt and returns a validated JSON answer. What to ask
lives in refine.py. Local-first: with no connection, no network or any error
the caller gets None and keeps the local result.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from pydantic import BaseModel

from . import config

PROVIDERS = ("claude", "openai", "groq")
GROQ_URL = "https://api.groq.com/openai/v1"  # OpenAI-compatible
CLI_TIMEOUT = 300
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
RATE_WAITS = (20, 40, 60)  # seconds to wait after a "too many requests" answer

FILE_NOTE = "\n\nThe images are the files {names} in the current folder, in that order. Open and look at each one before you answer."


# ---------- connections ----------

INSTALL = {  # official pages, shown in Settings when the app is missing
    "claude": "https://code.claude.com/docs/en/setup",
    "codex": "https://openai.com/codex/",
}


def _newest(paths) -> list[Path]:
    return sorted((p for p in paths if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def _cli(name: str) -> str | None:
    """The official AI command-line app, wherever it is on this PC (the app's own
    tools folder, its installer's folder, or inside the VS Code / Cursor extension)."""
    own = config.TOOLS / name / f"{name}.exe"
    if own.is_file():
        return str(own)
    found = shutil.which(name)
    if found:
        return found
    home = Path.home()
    local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    roaming = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
    exe = f"{name}.exe"
    spots = [home / ".local" / "bin" / exe, local / "Microsoft" / "WinGet" / "Links" / exe,
             roaming / "npm" / f"{name}.cmd", local / "Programs" / name / exe]
    if name == "claude":  # native installer, or the copy inside the VS Code / Cursor extension
        for ide in (".vscode", ".vscode-insiders", ".cursor", ".windsurf"):
            spots += _newest((home / ide / "extensions").glob("anthropic.claude-code-*/resources/native-binary/claude.exe"))
    else:  # the ChatGPT / Codex desktop app keeps codex.exe in a versioned folder
        spots += _newest((local / "OpenAI" / "Codex" / "bin").glob("*/codex.exe"))
        for ide in (".vscode", ".vscode-insiders", ".cursor", ".windsurf"):
            spots += _newest((home / ide / "extensions").glob("openai.chatgpt-*/bin/*/codex.exe"))
    return next((str(p) for p in spots if p.is_file()), None)


def _run(cmd: list[str], prompt: str, cwd: str, timeout: int = CLI_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=prompt, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, creationflags=NO_WINDOW)


def account(provider: str) -> dict:
    """Is the official app here, and is it signed in? {installed, signed_in, who}"""
    name = "claude" if provider == "claude" else "codex"
    exe = _cli(name)
    if not exe:
        return {"installed": False, "signed_in": False, "who": None}
    try:
        if name == "claude":
            r = _run([exe, "auth", "status"], "", tempfile.gettempdir(), timeout=30)
            data = json.loads(r.stdout or "{}")
            return {"installed": True, "signed_in": bool(data.get("loggedIn")), "who": data.get("email")}
        r = _run([exe, "login", "status"], "", tempfile.gettempdir(), timeout=30)
        text = (r.stdout + r.stderr).strip()
        return {"installed": True, "signed_in": r.returncode == 0 and "logged in" in text.lower(), "who": text.splitlines()[-1] if text else None}
    except Exception:
        return {"installed": True, "signed_in": False, "who": None}


def connections(s) -> dict:
    """What can be used right now, in the order it is tried."""
    claude = [m for m, ok in (("account", _cli("claude") and "claude" not in s.ai_accounts_off), ("api", s.anthropic_api_key)) if ok]
    openai = [m for m, ok in (("account", _cli("codex") and "openai" not in s.ai_accounts_off), ("api", s.openai_api_key)) if ok]
    groq = ["api"] if s.groq_api_key else []
    order = {"claude": claude, "openai": openai, "groq": groq}.get(s.reviewer_provider, [])
    return {"claude": claude, "openai": openai, "groq": groq, "ready": bool(order), "install": INSTALL}


# ---------- "Connect account": download the official app, then its own sign-in ----------

CLAUDE_DOWNLOADS = "https://downloads.claude.ai/claude-code-releases"  # what Anthropic's installer uses
CODEX_RELEASE = "https://api.github.com/repos/openai/codex/releases/latest"  # OpenAI's official releases


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Wound-Recognizer"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, target: Path, progress=lambda m: None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Wound-Recognizer"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total, done = int(r.headers.get("Content-Length") or 0), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                progress(f"Downloading {done * 100 // total}%")
    os.replace(tmp, target)


def install(provider: str, progress=lambda m: None) -> str:
    """Download the official command-line app into tools/ (checked against the
    publisher's checksum or size). Returns its path."""
    arm = os.environ.get("PROCESSOR_ARCHITECTURE") == "ARM64"
    if provider == "claude":
        version = _fetch(f"{CLAUDE_DOWNLOADS}/latest").decode().strip()
        if not version or len(version) > 40 or "<" in version:
            raise RuntimeError("Could not reach Anthropic's download service.")
        platform = "win32-arm64" if arm else "win32-x64"
        manifest = json.loads(_fetch(f"{CLAUDE_DOWNLOADS}/{version}/manifest.json"))
        checksum = manifest["platforms"][platform]["checksum"].lower()
        target = config.TOOLS / "claude" / "claude.exe"
        _download(f"{CLAUDE_DOWNLOADS}/{version}/{platform}/claude.exe", target, progress)
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            target.unlink()
            raise RuntimeError("The downloaded Claude Code file did not match Anthropic's checksum. Nothing was installed.")
        return str(target)
    release = json.loads(_fetch(CODEX_RELEASE))
    want = f"codex-{'aarch64' if arm else 'x86_64'}-pc-windows-msvc.exe"
    asset = next((a for a in release.get("assets", []) if a.get("name") == want), None)
    if not asset:
        raise RuntimeError("Could not find the Codex app for Windows on OpenAI's release page.")
    target = config.TOOLS / "codex" / "codex.exe"
    _download(asset["browser_download_url"], target, progress)
    digest = (asset.get("digest") or "").removeprefix("sha256:").lower()
    if (digest and hashlib.sha256(target.read_bytes()).hexdigest() != digest) or target.stat().st_size != asset.get("size", target.stat().st_size):
        target.unlink()
        raise RuntimeError("The downloaded Codex file did not match OpenAI's release. Nothing was installed.")
    return str(target)


def sign_in(provider: str) -> None:
    """Start the official sign-in: it opens the browser on claude.ai / chatgpt.com.
    The app never sees the password or the token."""
    name = "claude" if provider == "claude" else "codex"
    exe = _cli(name)
    if not exe:
        raise RuntimeError("The official app is not on this computer yet.")
    cmd = [exe, "auth", "login"] if name == "claude" else [exe, "login"]
    # Its own console window, so the user can see the sign-in link and any code it shows.
    subprocess.Popen(cmd, cwd=tempfile.gettempdir(), creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)


def sign_out(provider: str) -> None:
    name = "claude" if provider == "claude" else "codex"
    exe = _cli(name)
    if exe:
        _run([exe, "auth", "logout"] if name == "claude" else [exe, "logout"], "", tempfile.gettempdir(), timeout=60)


# ---------- asking ----------

def _schema(model: type[BaseModel]) -> dict:
    return model.model_json_schema()


def _names(n: int) -> list[str]:
    return [f"image{i + 1}.jpg" for i in range(n)]


def _claude_account(images: list[bytes], prompt: str, model: str, schema: type[BaseModel]) -> BaseModel:
    """Claude Code CLI, signed in with the Claude account (no API key)."""
    with tempfile.TemporaryDirectory(prefix="wr-ai-") as d:
        names = _names(len(images))
        for n, data in zip(names, images):
            (Path(d) / n).write_bytes(data)
        cmd = [_cli("claude"), "-p", "--output-format", "json", "--json-schema", json.dumps(_schema(schema)),
               "--model", model, "--tools", "Read", "--allowedTools", "Read", "--setting-sources", "",  # skip plugins and hooks: 14 s instead of 2 min
               "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
        r = _run(cmd, prompt + FILE_NOTE.format(names=", ".join(names)), d)
        data = json.loads(r.stdout or "{}")
        if r.returncode or data.get("is_error"):
            raise RuntimeError((data.get("result") or r.stderr or "Claude did not answer").strip()[:300])
        out = data.get("structured_output") or json.loads(data["result"])
        return schema.model_validate(out)


def _openai_account(images: list[bytes], prompt: str, model: str, schema: type[BaseModel], effort: str = "low") -> BaseModel:
    """OpenAI Codex CLI, signed in with the ChatGPT account (no API key)."""
    with tempfile.TemporaryDirectory(prefix="wr-ai-") as d:
        schema_file, answer = Path(d) / "schema.json", Path(d) / "answer.json"
        schema_file.write_text(json.dumps(_schema(schema)), encoding="utf-8")
        # Lean: no user config, plugins or AGENTS.md (they cost thousands of tokens a photo).
        cmd = [_cli("codex"), "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-user-config",
               "-c", f"model_reasoning_effort={effort}", "-c", "project_doc_max_bytes=0",
               "--output-schema", str(schema_file), "--output-last-message", str(answer), "--ephemeral"]
        for n, data in zip(_names(len(images)), images):
            (Path(d) / n).write_bytes(data)
            cmd += ["--image", str(Path(d) / n)]
        if model:
            cmd += ["--model", model]
        r = _run(cmd + ["-"], prompt, d)
        if r.returncode or not answer.is_file():
            raise RuntimeError((r.stderr or r.stdout or "ChatGPT did not answer").strip()[-300:])
        return schema.model_validate_json(answer.read_text(encoding="utf-8"))


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("utf-8")


def _claude_api(images: list[bytes], prompt: str, model: str, key: str, schema: type[BaseModel]) -> BaseModel:
    import anthropic

    client = anthropic.Anthropic(api_key=key, timeout=180, max_retries=2)
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(j)}} for j in images]
    response = client.messages.parse(
        model=model,
        max_tokens=6000,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content + [{"type": "text", "text": prompt}]}],
        output_format=schema,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError("Claude declined to look at this photo.")
    return response.parsed_output


def _openai_api(images: list[bytes], prompt: str, model: str, key: str, schema: type[BaseModel], effort: str = "low") -> BaseModel:
    from openai import OpenAI

    client = OpenAI(api_key=key, timeout=180, max_retries=2)
    content = [{"type": "input_text", "text": prompt}]
    content += [{"type": "input_image", "image_url": "data:image/jpeg;base64," + _b64(j)} for j in images]
    response = client.responses.parse(model=model, input=[{"role": "user", "content": content}], text_format=schema,
                                      reasoning={"effort": effort})
    if response.output_parsed is None:
        raise RuntimeError("ChatGPT declined to look at this photo.")
    return response.output_parsed


def _groq_api(images: list[bytes], prompt: str, model: str, key: str, schema: type[BaseModel]) -> BaseModel:
    """Groq (free plan available): OpenAI-compatible chat API with JSON mode."""
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=GROQ_URL, timeout=180, max_retries=2)
    shape = "Answer with only a JSON object that follows this JSON schema:\n" + json.dumps(_schema(schema))
    content = [{"type": "text", "text": prompt + "\n\n" + shape}]
    content += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(j)}} for j in images]
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_completion_tokens=3000,
    )
    text = response.choices[0].message.content or ""
    return schema.model_validate_json(text[text.find("{"): text.rfind("}") + 1])


def _attempts(s):
    if s.reviewer_provider == "groq":
        if s.groq_api_key:
            yield "Groq API key", lambda i, p, sc: _groq_api(i, p, s.groq_model, s.groq_api_key, sc)
        return
    if s.reviewer_provider == "openai":
        if _cli("codex") and "openai" not in s.ai_accounts_off:
            yield "ChatGPT account", lambda i, p, sc: _openai_account(i, p, s.openai_model, sc, s.openai_effort)
        if s.openai_api_key:
            yield "OpenAI API key", lambda i, p, sc: _openai_api(i, p, s.openai_model, s.openai_api_key, sc, s.openai_effort)
    else:
        if _cli("claude") and "claude" not in s.ai_accounts_off:
            yield "Claude account", lambda i, p, sc: _claude_account(i, p, s.reviewer_model, sc)
        if s.anthropic_api_key:
            yield "Claude API key", lambda i, p, sc: _claude_api(i, p, s.reviewer_model, s.anthropic_api_key, sc)


def _rate_limited(e: Exception) -> bool:
    text = str(e).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _model_name(s) -> str:
    return {"groq": s.groq_model, "openai": f"{s.openai_model}/{s.openai_effort}"}.get(s.reviewer_provider, s.reviewer_model)


def ask(images: list[bytes], prompt: str, schema: type[BaseModel], s, salt: str = "") -> tuple[BaseModel | None, str | None, list[str]]:
    """Account first, then API key. The same question again gets the saved answer
    (no time, no quota). Returns (answer, via, errors)."""
    h = hashlib.sha1()
    for part in (*images, prompt.encode(), s.reviewer_provider.encode(), _model_name(s).encode(), salt.encode()):
        h.update(hashlib.sha1(part).digest())
    cached = config.CACHE / "ai" / f"{h.hexdigest()}.json"
    if cached.is_file():
        try:
            saved = json.loads(cached.read_text(encoding="utf-8"))
            return schema.model_validate(saved["answer"]), saved["via"] + " (saved answer)", []
        except Exception:
            pass
    errors = []
    for via, call in _attempts(s):
        for wait in (*RATE_WAITS, None):
            try:
                answer = call(images, prompt, schema)
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps({"via": via, "answer": answer.model_dump()}, ensure_ascii=False), encoding="utf-8")
                return answer, via, errors
            except Exception as e:  # offline, signed out, bad key, rate limit...
                if wait is not None and _rate_limited(e):
                    time.sleep(wait)  # free plans allow a few requests a minute: wait, then ask again
                    continue
                errors.append(f"{via}: {str(e).strip()[:300] or type(e).__name__}")
                break
    return None, None, errors


def missing_message(s) -> str:
    return {
        "openai": "ChatGPT: the official Codex app is not on this computer and no OpenAI API key is saved. Press Connect ChatGPT account, or add a key.",
        "groq": "Groq: no Groq API key is saved yet. Get a free one at console.groq.com.",
    }.get(s.reviewer_provider, "Claude: the official Claude Code app is not on this computer and no Anthropic API key is saved. Press Connect Claude account, or add a key.")


def test(s) -> dict:
    """One small real request through the same path as marking."""
    if not connections(s)["ready"]:
        return {"ok": False, "via": None, "errors": [missing_message(s)]}
    from .refine import test_question

    images, prompt, schema = test_question()
    answer, via, errors = ask(images, prompt, schema, s, salt=str(time.time()))  # never a saved answer
    return {"ok": answer is not None, "via": via, "errors": errors}
