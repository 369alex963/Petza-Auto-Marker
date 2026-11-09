import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, type Provider } from "../api";

const ACCOUNT_APP: Record<Provider, { app: string; signin: string }> = {
  claude: { app: "Claude Code", signin: "Install Claude Code and sign in with the Claude account once." },
  openai: { app: "Codex", signin: "Install the OpenAI Codex app and sign in with the ChatGPT account once (codex login)." },
};

type TestResult = { ok: boolean; via: string | null; errors: string[] };

export function Settings({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [enabled, setEnabled] = useState(false);
  const [provider, setProvider] = useState<Provider>("claude");
  const [key, setKey] = useState("");
  const [claudeModel, setClaudeModel] = useState("claude-opus-5-5");
  const [openaiModel, setOpenaiModel] = useState("gpt-6-astra");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [test, setTest] = useState<TestResult | null>(null);

  useEffect(() => {
    if (!s) return;
    setEnabled(s.settings.reviewer_enabled);
    setProvider(s.settings.reviewer_provider);
    setClaudeModel(s.settings.reviewer_model);
    setOpenaiModel(s.settings.openai_model);
  }, [s?.settings.reviewer_enabled, s?.settings.reviewer_provider, s?.settings.reviewer_model, s?.settings.openai_model]); // eslint-disable-line react-hooks/exhaustive-deps

  const keyField = provider === "claude" ? "anthropic_api_key" : "openai_api_key";
  const hasKey = provider === "claude" ? s?.settings.anthropic_api_key : s?.settings.openai_api_key;
  const hasAccount = s?.reviewer[provider].includes("account") ?? false;

  const persist = async () => {
    const body: Record<string, string | boolean> = {
      reviewer_enabled: enabled,
      reviewer_provider: provider,
      reviewer_model: claudeModel,
      openai_model: openaiModel.trim() || "gpt-6-astra",
    };
    if (key.trim()) body[keyField] = key.trim();
    await api.saveSettings(body);
    setKey("");
    await ctx.refresh();
  };

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await persist();
      ctx.toast("Settings saved");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const runTest = async () => {
    setTesting(true);
    setTest(null);
    setError(null);
    try {
      await persist();
      setTest(await api.testReviewer());
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setTesting(false);
    }
  };

  const removeKey = async () => {
    await api.saveSettings({ [keyField]: "" });
    await ctx.refresh();
    ctx.toast("Key removed");
  };

  const pick = (p: Provider) => {
    setProvider(p);
    setTest(null);
    setKey("");
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
            Check each new marking after “Mark all”
          </label>

          <fieldset className="field">
            <legend className="field__label">Which AI</legend>
            <div className="field__row">
              <label className="switch">
                <input type="radio" name="provider" checked={provider === "claude"} onChange={() => pick("claude")} />
                Claude
              </label>
              <label className="switch">
                <input type="radio" name="provider" checked={provider === "openai"} onChange={() => pick("openai")} />
                ChatGPT
              </label>
            </div>
          </fieldset>

          <div className="field">
            <span className="field__label">Account on this computer</span>
            {hasAccount ? (
              <span>The {ACCOUNT_APP[provider].app} app is installed. Its signed-in account is used first, no key needed.</span>
            ) : (
              <span className="muted">Not found. {ACCOUNT_APP[provider].signin} Or add an API key below.</span>
            )}
          </div>

          <div className="field">
            <label className="field__label" htmlFor="key">
              {provider === "claude" ? "Anthropic API key" : "OpenAI API key"}
              {hasAccount ? " (backup)" : ""}
            </label>
            <div className="field__row">
              <input
                id="key"
                name="key"
                type="password"
                className="input"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder={hasKey ? "A key is saved. Type to replace it." : provider === "claude" ? "sk-ant-..." : "sk-..."}
                autoComplete="off"
              />
              {hasKey ? (
                <button type="button" className="btn" onClick={() => void removeKey()}>
                  Remove
                </button>
              ) : null}
            </div>
            <span className="field__help">Used only when the account above is missing or fails. Stored only in this folder (data\settings.json).</span>
          </div>

          <div className="field">
            <label className="field__label" htmlFor="model">
              Model
            </label>
            {provider === "claude" ? (
              <select id="model" name="model" className="input" value={claudeModel} onChange={(e) => setClaudeModel(e.target.value)}>
                <option value="claude-opus-5-5">Claude Opus 5.5 (most careful)</option>
                <option value="claude-sonnet-5-5">Claude Sonnet 5.5 (faster)</option>
              </select>
            ) : (
              <input id="model" name="model" className="input" value={openaiModel} onChange={(e) => setOpenaiModel(e.target.value)} placeholder="gpt-6-astra" />
            )}
            <span className="field__help">Photos are sent to {provider === "claude" ? "Anthropic" : "OpenAI"} only when the second opinion is used.</span>
          </div>

          {error ? (
            <div className="note note--error" role="alert">
              <p>{error}</p>
            </div>
          ) : null}
          {test ? (
            <div className={`note ${test.ok ? "" : "note--error"}`} role="status">
              <p>{test.ok ? `Connected through the ${test.via}.` : "Could not connect."}</p>
              {test.errors.length ? (
                <ul>
                  {test.errors.map((x, i) => (
                    <li key={i}>{x}</li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}
          <div className="toolbar">
            <button type="submit" className="btn btn--primary" disabled={saving || testing}>
              {saving ? "Saving" : "Save settings"}
            </button>
            <button type="button" className="btn" disabled={saving || testing} onClick={() => void runTest()}>
              {testing ? "Testing, up to a minute" : "Test connection"}
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
