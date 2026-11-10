# Wound-Recognizer

A local app that marks wounds on medical photos. Each mark is an outline of points, saved as one JSON file per photo for the other medical app. It learns from your marked examples and from every photo you correct. Everything runs on this computer.

## Start

Double-click `start.bat`. The first time, it installs everything into `.venv` inside this folder, which takes a while once. Then the app opens in the browser at http://127.0.0.1:8765. Keep the black window open while you use it.

`setup.bat` installs the right build for the PC: CUDA on NVIDIA cards, DirectML on other graphics cards, CPU otherwise.

## Use

1. **Folders:** choose the folder of marked examples, the photo folder, and optionally an output folder.
2. **Teach:** press **Import examples**, then **Teach now**. Some photos are kept aside and never learned from. The score is how well the app's outline overlaps your outline on those photos.
3. **Photos:** press **Mark all photos**. The smiley shows how sure the app is. "Worth a look" lists the doubtful photos first.
   The **Hardest test photos** on Teach show your outline (dark) next to the app's (pink), so you can see where it still struggles. Any earlier model can be switched back on from the history.
4. **Editor:** drag points, click the outline to add a point, right-click a point to remove it, press `L` to draw a new area with the point lasso. Press **Approve** (the heart) to save. Approved photos are never overwritten, and they teach the app on the next Teach. **Mark this photo** re-runs the app on just the open photo.

## AI-guided marking (optional)

The local model draws, the online AI decides (`petza/refine.py`). Per photo:

1. `quality.py` checks brightness, contrast and sharpness. A dark or flat photo is corrected only in the copy that is sent.
2. From the local probability map, each candidate region (found at p ≥ 0.25) gets three pixel-precise outlines: A tight (p ≥ 0.7), B normal (argmax, the local result) and C wide (p ≥ 0.3).
3. The AI receives:
   - the photo, at most 1024 px and re-encoded, so it carries no EXIF and no file name;
   - a numbered A/B/C overlay, zoomed in when the regions are small;
   - the active prompt.
4. It answers with strict JSON: per region `keep`, `outline`, `label` and `reason`, plus `missed` boxes. A missed box is re-predicted locally at p ≥ 0.2 and outlined.
5. By default the edge stays local (B). On real diabetic photos the AI's A/C choice was less exact (IoU 0.792 against 0.820). Setting `ai_edges` allows A/C only where the edge is unsure (tight/wide area < 0.75).
6. Quota:
   - `openai_model` defaults to `gpt-6-luna` with `openai_effort` `low`.
   - The Codex call uses `--ignore-user-config -c project_doc_max_bytes=0`, and images are at most 1024 px.
   - `ai_scope` `unsure` (the default) skips clear-cut photos: one candidate with a B outline, confidence at or above the threshold, good quality.
   - `ai_accounts_off` is the app-only Disconnect.
7. Settings `ai_runs` sets 1, 2 or 3 questions per photo, decided by majority vote; a tie keeps the local result.
8. Answers are cached in `data/cache/ai`. Rate limits are waited out. Any failure keeps the local result.
9. Applied outlines join the training pool as kind `ai`. They are never held out, so the promote gate still scores against human markings only.

The same code runs for **Mark all**, **Mark this photo again** and **Try one photo** (`POST /api/try`).

Connections (`petza/reviewer.py`):
- Claude uses the official Claude Code CLI; ChatGPT uses the official Codex CLI. These are the same binaries the official VS Code extensions run.
- **Connect account** downloads the CLI into `tools/` if it is missing (Anthropic's release manifest with a sha256 check, or OpenAI's GitHub release with a size check). It then runs the CLI's own `auth login` / `login`.
- The fallback is an API key. Groq always uses an API key.

Measure whether the AI helps on your own held-out photos: `.venv\Scripts\python -m petza.compare_ai`.

## Marking prompt

The prompt the online AI reads lives in the app (Settings, "Marking prompt for the AI"), saved in `data/prompts.json`; several named prompts, one active, a Hebrew default in `petza/prompts.py`. An optional block at the top sets the measurable rules:

```
---
labels: [wound]
review_below_confidence: 0.75
min_area_px: 400
point_spacing_px: 12
max_regions: 3
---
Trace the outer edge of the wound bed...
```

When `point_spacing_px` is not set, the app copies the point spacing of the examples.

## JSON output

```json
{"image": "a.jpg", "width": 4032, "height": 3024,
 "regions": [{"id": "r1", "label": "wound", "points": [[x, y], ...], "confidence": 0.94, "source": "auto"}],
 "model_version": "m-...", "status": "auto | reviewed", "confidence": 0.94,
 "quality": "good | dark | low_contrast | blurry", "ai": {"ok": true, "via": "...", "runs": 1, "choices": [...], "missed": 0, "notes": "..."}}
```

Points are pixels in the upright photo (EXIF rotation applied), origin top-left.

The examples importer reads this format, LabelMe, COCO, and most JSON files that hold point lists. A converter for the other app's exact format gets added once a sample file arrives.

## Developer

- Tests: `.venv\Scripts\python -m pytest`. Includes an end-to-end train, mark and score run on synthetic wounds.
- Frontend: `cd frontend && npm install && npm run build`. The built `frontend/dist` is committed, so the app itself needs no Node.
