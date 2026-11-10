import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, type Provider } from "../api";
import { PromptEditor } from "./PromptEditor";

const ACCOUNT_APP: Record<"claude" | "openai", { app: string; signin: string }> = {
  claude: { app: "Claude Code", signin: "Install Claude Code and sign in with your Claude account once." },
  openai: { app: "the ChatGPT Codex app", signin: "Install the Codex app and sign in with your ChatGPT account once." },
};

const KEY_FIELD: Record<Provider, "anthropic_api_key" | "openai_api_key" | "groq_api_key"> = {
  claude: "anthropic_api_key",
  openai: "openai_api_key",
  groq: "groq_api_key",
};
const KEY_LABEL: Record<Provider, string> = { claude: "Anthropic API key", openai: "OpenAI API key", groq: "Groq API key" };
const KEY_HINT: Record<Provider, string> = { claude: "sk-ant-...", openai: "sk-...", groq: "gsk_..." };
const COMPANY: Record<Provider, string> = { claude: "Anthropic", openai: "OpenAI", groq: "Groq" };

type TestResult = { ok: boolean; via: string | null; errors: string[] };

export function Settings({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [enabled, setEnabled] = useState(false);
  const [provider, setProvider] = useState<Provider>("claude");
  const [key, setKey] = useState("");
  const [models, setModels] = useState<Record<Provider, string>>({ claude: "claude-opus-5-5", openai: "gpt-6-astra", groq: "qwen/qwen3.8-27b" });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [test, setTest] = useState<TestResult | null>(null);

  useEffect(() => {
    if (!s) return;
    setEnabled(s.settings.reviewer_enabled);
    setProvider(s.settings.reviewer_provider);
    setModels({ claude: s.settings.reviewer_model, openai: s.settings.openai_model, groq: s.settings.groq_model });
  }, [s?.settings.reviewer_enabled, s?.settings.reviewer_provider, s?.settings.reviewer_model, s?.settings.openai_model, s?.settings.groq_model]); // eslint-disable-line react-hooks/exhaustive-deps

  const hasKey = s ? s.settings[KEY_FIELD[provider]] : false;
  const hasAccount = provider !== "groq" && (s?.reviewer[provider].includes("account") ?? false);

  const persist = async () => {
    const body: Record<string, string | boolean> = {
      reviewer_enabled: enabled,
      reviewer_provider: provider,
      reviewer_model: models.claude,
      openai_model: models.openai.trim() || "gpt-6-astra",
      groq_model: models.groq.trim() || "qwen/qwen3.8-27b",
    };
    if (key.trim()) body[KEY_FIELD[provider]] = key.trim();
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
    await api.saveSettings({ [KEY_FIELD[provider]]: "" });
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
            After “Mark all photos”, also ask the AI to look at every new marking and flag the ones that may be wrong (about 10 to 20 seconds per photo)
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
              <label className="switch">
                <input type="radio" name="provider" checked={provider === "groq"} onChange={() => pick("groq")} />
                Groq (free)
              </label>
            </div>
          </fieldset>

          {provider === "groq" ? (
            <div className="note">
              <p>
                <strong>Groq is free.</strong> It needs a free API key, no credit card:
              </p>
              <ol>
                <li>
                  Open{" "}
                  <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer">
                    console.groq.com/keys
                  </a>{" "}
                  and sign up with Google or an email address.
                </li>
                <li>Press “Create API Key”, give it any name, and copy the key (it starts with gsk_).</li>
                <li>Paste it below, then press Test connection.</li>
              </ol>
              <p className="muted">
                Free plan limits for the vision model: 30 requests a minute, 1,000 a day, 8,000 tokens a minute and 200,000 a day. Each photo uses about 3,000 to 4,000 tokens,
                so expect about 2 photos a minute and about 50 photos a day. Your exact limits are on{" "}
                <a href="https://console.groq.com/settings/limits" target="_blank" rel="noreferrer">
                  Groq’s limits page
                </a>
                .
              </p>
            </div>
          ) : (
            <div className="field">
              <span className="field__label">Account on this computer</span>
              {hasAccount ? (
                <span>{provider === "claude" ? "Claude Code" : "The Codex app"} is installed. Its signed-in account is used first, no key needed.</span>
              ) : (
                <span className="muted">
                  Not found on this computer. {ACCOUNT_APP[provider].signin}{" "}
                  <a href={provider === "claude" ? s?.reviewer.install.claude : s?.reviewer.install.codex} target="_blank" rel="noreferrer">
                    Download {ACCOUNT_APP[provider].app} from the official site
                  </a>
                  , sign in, then press Test connection. Or add an API key below.
                </span>
              )}
            </div>
          )}

          <div className="field">
            <label className="field__label" htmlFor="key">
              {KEY_LABEL[provider]}
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
                placeholder={hasKey ? "A key is saved. Type to replace it." : KEY_HINT[provider]}
                autoComplete="off"
              />
              {hasKey ? (
                <button type="button" className="btn" onClick={() => void removeKey()}>
                  Remove
                </button>
              ) : null}
            </div>
            <span className="field__help">
              {provider === "groq" ? "" : "Used only when the account above is missing or fails. "}Stored only in this folder (data\settings.json).
            </span>
          </div>

          <div className="field">
            <label className="field__label" htmlFor="model">
              Model
            </label>
            {provider === "claude" ? (
              <select id="model" name="model" className="input" value={models.claude} onChange={(e) => setModels((m) => ({ ...m, claude: e.target.value }))}>
                <option value="claude-opus-5-5">Claude Opus 5.5 (most careful)</option>
                <option value="claude-sonnet-5-5">Claude Sonnet 5.5 (faster)</option>
              </select>
            ) : (
              <input
                id="model"
                name="model"
                className="input"
                value={models[provider]}
                onChange={(e) => setModels((m) => ({ ...m, [provider]: e.target.value }))}
                spellCheck={false}
              />
            )}
            <span className="field__help">Photos are sent to {COMPANY[provider]} only when the second opinion is used.</span>
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

        <section className="panel" aria-label="Marking prompt for the AI">
          <PromptEditor onChange={() => void ctx.refresh()} toast={ctx.toast} />
        </section>
      </div>

      <section className="panel" aria-labelledby="out" style={{ marginTop: "var(--space-lg)" }}>
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
    </main>
  );
}
