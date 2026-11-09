import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api } from "../api";

export function Settings({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [enabled, setEnabled] = useState(false);
  const [key, setKey] = useState("");
  const [model, setModel] = useState("claude-opus-5-5");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!s) return;
    setEnabled(s.settings.reviewer_enabled);
    setModel(s.settings.reviewer_model);
  }, [s?.settings.reviewer_enabled, s?.settings.reviewer_model]); // eslint-disable-line react-hooks/exhaustive-deps

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      const body: Record<string, string | boolean> = { reviewer_enabled: enabled, reviewer_model: model };
      if (key.trim()) body.anthropic_api_key = key.trim();
      await api.saveSettings(body);
      setKey("");
      await ctx.refresh();
      ctx.toast("Settings saved");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const removeKey = async () => {
    await api.saveSettings({ anthropic_api_key: "", reviewer_enabled: false });
    await ctx.refresh();
    ctx.toast("Key removed");
  };

  return (
    <main className="page">
      <div className="page__head">
        <div>
          <h1 className="page__title">Settings</h1>
          <p className="page__sub">The app marks everything on this computer. An online AI can add a second opinion that flags doubtful photos; it never moves a point.</p>
        </div>
      </div>
      <div className="cols">
        <form className="panel" onSubmit={save}>
          <h2 className="panel__title">AI second opinion</h2>
          <label className="switch">
            <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
            Check each new marking with Claude after "Mark all"
          </label>
          <div className="field">
            <label className="field__label" htmlFor="key">
              Anthropic API key
            </label>
            <div className="field__row">
              <input
                id="key"
                name="key"
                type="password"
                className="input"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder={s?.settings.anthropic_api_key ? "A key is saved. Type to replace it." : "sk-ant-..."}
                autoComplete="off"
              />
              {s?.settings.anthropic_api_key ? (
                <button type="button" className="btn" onClick={() => void removeKey()}>
                  Remove
                </button>
              ) : null}
            </div>
            <span className="field__help">Stored only in this folder (data\settings.json). Photos are sent to Anthropic only when this is on.</span>
          </div>
          <div className="field">
            <label className="field__label" htmlFor="model">
              Model
            </label>
            <select id="model" name="model" className="input" value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="claude-opus-5-5">Claude Opus 5.5 (most careful)</option>
              <option value="claude-sonnet-5-5">Claude Sonnet 5.5 (faster, lower cost)</option>
            </select>
            <span className="field__help" />
          </div>
          {error ? (
            <div className="note note--error" role="alert">
              <p>{error}</p>
            </div>
          ) : null}
          <div>
            <button type="submit" className="btn btn--primary" disabled={saving}>
              {saving ? "Saving" : "Save settings"}
            </button>
          </div>
        </form>
        <section className="panel" aria-labelledby="out">
          <h2 className="panel__title" id="out">
            The JSON files
          </h2>
          <p>Each photo gets a JSON file with the same name. Points are pixel positions in the photo as it is shown upright, measured from the top-left corner.</p>
          <pre className="rules-text">{`{
  "image": "wound_001.jpg",
  "width": 4032, "height": 3024,
  "regions": [
    { "label": "wound",
      "points": [[1520.0, 980.5], [1544.0, 975.0], ...],
      "confidence": 0.94 }
  ],
  "status": "auto"  (or "reviewed" once approved)
}`}</pre>
        </section>
      </div>
    </main>
  );
}
