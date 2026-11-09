# Petza Auto-Marker

A local app that marks wounds on medical photos. Each mark is an outline of points, saved as one JSON file per photo for the other medical app. It learns from Lauren's marked examples and from every photo she corrects. Everything runs on this computer.

## Start

Double-click `start.bat`. The first time, it installs everything into `.venv` inside this folder, which takes a while once. Then the app opens in the browser at http://127.0.0.1:8765. Keep the black window open while you use it.

`setup.bat` installs the right build for the PC: CUDA on NVIDIA cards, DirectML on other graphics cards, CPU otherwise.

## Use

1. **Folders:** choose the photo folder, the rules file (`.md` or `.txt`, optional), the folder of marked examples, and optionally an output folder.
2. **Teach:** press **Import examples**, then **Teach now**. Some photos are kept aside and never learned from. The score is how well the app's outline overlaps Lauren's outline on those photos.
3. **Photos:** press **Mark all photos**. The smiley shows how sure the app is. "Worth a look" lists the doubtful photos first.
   The **Hardest test photos** on Teach show Lauren's outline (dark) next to the app's (pink), so you can see where it still struggles. Any earlier model can be switched back on from the history.
4. **Editor:** drag points, click the outline to add a point, right-click a point to remove it, press `L` to draw a new area with the point lasso. Press **Approve** (the heart) to save. Approved photos are never overwritten, and they teach the app on the next Teach. **Mark this photo** re-runs the app on just the open photo.

## Rules file

Plain text for people. An optional block at the top sets the measurable rules:

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
 "model_version": "m-...", "status": "auto | reviewed", "confidence": 0.94}
```

Points are pixels in the upright photo (EXIF rotation applied), origin top-left.

The examples importer reads this format, LabelMe, COCO, and most JSON files that hold point lists. A converter for the other app's exact format gets added once a sample file arrives.

## Developer

- Tests: `.venv\Scripts\python -m pytest`. Includes an end-to-end train, mark and score run on synthetic wounds.
- Frontend: `cd frontend && npm install && npm run build`. The built `frontend/dist` is committed, so the app itself needs no Node.
- Spec: `superpowers/specs/`. Plan: `superpowers/plans/`.
