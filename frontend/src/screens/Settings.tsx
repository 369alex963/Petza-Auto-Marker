import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api, type Account, type Provider } from "../api";
import { DemoNote } from "./DemoNote";
import { PromptEditor } from "./PromptEditor";

const ACCOUNT_APP: Record<"claude" | "openai", { app: string; site: string; name: string }> = {
  claude: { app: "Claude Code", site: "claude.ai", name: "Claude" },
  openai: { app: "Codex", site: "chatgpt.com", name: "ChatGPT" },
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
  const [provider, setProvider] = useState<Provider>("openai");
  const [key, setKey] = useState("");
  const [models, setModels] = useState<Record<Provider, string>>({ claude: "claude-haiku-5-5", openai: "gpt-6-luna", groq: "qwen/qwen3.8-27b" });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [test, setTest] = useState<TestResult | null>(null);
  const [runs, setRuns] = useState(1);
  const [edges, setEdges] = useState(false);
  const [scope, setScope] = useState<"unsure" | "all">("unsure");
  const [accounts, setAccounts] = useState<Record<"claude" | "openai", Account> | null>(null);
  const [waiting, setWaiting] = useState(false);

  const loadAccounts = () => api.accounts().then(setAccounts).catch(() => undefined);
  useEffect(() => {
    void loadAccounts();
  }, [ctx.jobVersion]);
  // After "Connect", check every few seconds until the sign-in in the browser is done.
  useEffect(() => {
    if (!waiting) return;
    const t = window.setInterval(() => void loadAccounts(), 4000);
    const stop = window.setTimeout(() => setWaiting(false), 5 * 60 * 1000);
    return () => {
      window.clearInterval(t);
      window.clearTimeout(stop);
    };
  }, [waiting]);

  useEffect(() => {
    if (!s) return;
    setEnabled(s.settings.reviewer_enabled);
    setProvider(s.settings.reviewer_provider);
    setModels({ claude: s.settings.reviewer_model, openai: s.settings.openai_model, groq: s.settings.groq_model });
    setRuns(s.settings.ai_runs || 1);
    setEdges(!!s.settings.ai_edges);
    setScope(s.settings.ai_scope || "unsure");
  }, [s?.settings.reviewer_enabled, s?.settings.reviewer_provider, s?.settings.reviewer_model, s?.settings.openai_model, s?.settings.groq_model, s?.settings.ai_runs, s?.settings.ai_edges, s?.settings.ai_scope]); // eslint-disable-line react-hooks/exhaustive-deps

  const hasKey = s ? s.settings[KEY_FIELD[provider]] : false;
  const acc = provider !== "groq" ? accounts?.[provider] : undefined;
  const hasAccount = !!acc?.signed_in && acc.in_app;
  const offInApp = !!acc?.signed_in && !acc.in_app;

  useEffect(() => {
    if (waiting && hasAccount && provider !== "groq") {
      setWaiting(false);
      ctx.toast(`${ACCOUNT_APP[provider].name} account connected`);
      void ctx.refresh();
    }
  }, [waiting, hasAccount]); // eslint-disable-line react-hooks/exhaustive-deps

  const connect = async () => {
    if (provider === "groq") return;
    setError(null);
    try {
      await api.connect(provider);
      setWaiting(true);
      await ctx.refresh();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const disconnect = async () => {
    if (provider === "groq") return;
    setAccounts(await api.disconnect(provider));
    await ctx.refresh();
    ctx.toast(`This app no longer uses your ${ACCOUNT_APP[provider].name} account`);
  };

  const persist = async () => {
    const body: Record<string, string | boolean | number> = {
      reviewer_enabled: enabled,
      ai_runs: runs,
      ai_edges: edges,
      reviewer_provider: provider,
      reviewer_model: models.claude,
      openai_model: models.openai.trim() || "gpt-6-luna",
      ai_scope: scope,
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
          <p className="page__sub">
            The marking is done on this computer by the app's own AI model. An online AI can make it more exact: it decides by your marking prompt which areas are
            really wound, removes the ones that are not, and points at wounds the app missed, which the app then outlines precisely. It is saved with nothing for you to do.
          </p>
        </div>
      </div>
      <div className="cols">
        <form className="panel" onSubmit={save}>
          <h2 className="panel__title">Online AI that improves the marking</h2>
          <label className="switch">
            <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
            Let the AI improve every marking automatically (by the marking prompt)
          </label>

          <fieldset className="field">
            <legend className="field__label">Which AI</legend>
            <div className="field__row">
              <label className="switch">
                <input type="radio" name="provider" checked={provider === "openai"} onChange={() => pick("openai")} />
                ChatGPT
              </label>
              <label className="switch">
                <input type="radio" name="provider" checked={provider === "claude"} onChange={() => pick("claude")} />
                Claude
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
              <span className="field__label">{ACCOUNT_APP[provider].name} account</span>
              <div className="account">
                {!accounts ? (
                  <span className="muted">Checking</span>
                ) : hasAccount ? (
                  <>
                    <span>Connected{acc?.who ? `: ${acc.who}` : ""}. No key needed.</span>
                    <button type="button" className="btn btn--small" onClick={() => void disconnect()}>
                      Disconnect from this app
                    </button>
                  </>
                ) : (
                  <>
                    <button type="button" className="btn btn--primary" disabled={waiting || ctx.job?.status === "running"} onClick={() => void connect()}>
                      {waiting ? "Waiting for the sign-in" : `Connect ${ACCOUNT_APP[provider].name} account`}
                    </button>
                    <span className="muted">
                      {waiting
                        ? `Sign in on ${ACCOUNT_APP[provider].site} in the window that opened.`
                        : offInApp
                          ? "Disconnected from this app. Your account stays signed in everywhere else."
                          : acc?.installed
                          ? `Opens ${ACCOUNT_APP[provider].site} to sign in.`
                          : `Downloads the official ${ACCOUNT_APP[provider].app} app, then opens ${ACCOUNT_APP[provider].site} to sign in.`}
                    </span>
                  </>
                )}
              </div>
              <span className="field__help">
                Like the official VS Code extension, the app signs in through {COMPANY[provider]}’s own {ACCOUNT_APP[provider].app} app and never sees your password.{" "}
                <a href={provider === "claude" ? s?.reviewer.install.claude : s?.reviewer.install.codex} target="_blank" rel="noreferrer">
                  About {ACCOUNT_APP[provider].app}
                </a>
              </span>
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
                <option value="claude-haiku-5-5">Claude Haiku 5.5: fast and affordable, hundreds of photos (recommended)</option>
                <option value="claude-sonnet-5-5">Claude Sonnet 5.5: stronger, uses the quota faster</option>
                <option value="claude-opus-5-5">Claude Opus 5.5: strongest, uses the quota much faster</option>
              </select>
            ) : provider === "openai" ? (
              <select id="model" name="model" className="input" value={models.openai} onChange={(e) => setModels((m) => ({ ...m, openai: e.target.value }))}>
                <option value="gpt-6-luna">GPT-6-Luna: fast and affordable, hundreds of photos (recommended)</option>
                <option value="gpt-6.1-sol">GPT-6.1-Sol: stronger, uses the quota faster</option>
                <option value="gpt-6-astra">GPT-6-Astra: strongest, uses the quota much faster</option>
                {!["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"].includes(models.openai) ? <option value={models.openai}>{models.openai}</option> : null}
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
            <span className="field__help">Photos are sent to {COMPANY[provider]} only while the switch above is on.</span>
          </div>

          <fieldset className="field">
            <legend className="field__label">Questions per photo</legend>
            <div className="field__row">
              {[1, 2, 3].map((n) => (
                <label key={n} className="switch">
                  <input type="radio" name="runs" checked={runs === n} onChange={() => setRuns(n)} />
                  {n === 1 ? "1 (default)" : n}
                </label>
              ))}
            </div>
            <span className="field__help">
              With 2 or 3, the AI is asked again and the majority answer wins. That is steadier, but every extra question takes the same time again (about 15 to 30
              seconds a photo) and uses that much more of the AI’s quota.
            </span>
          </fieldset>

          <fieldset className="field">
            <legend className="field__label">Which photos to ask about</legend>
            <label className="switch">
              <input type="radio" name="scope" checked={scope === "unsure"} onChange={() => setScope("unsure")} />
              Only photos that are not clear-cut (recommended)
            </label>
            <label className="switch">
              <input type="radio" name="scope" checked={scope === "all"} onChange={() => setScope("all")} />
              Every photo
            </label>
            <span className="field__help">
              A photo with one confident wound and good quality is not sent: measured on real photos, the AI added nothing there. That keeps the quota for the
              photos that need it: extra areas, faint or missed wounds, low confidence, dark or blurry photos.
            </span>
          </fieldset>

          <div className="field">
            <label className="switch">
              <input type="checkbox" checked={edges} onChange={(e) => setEdges(e.target.checked)} />
              Also let the AI move the edge (tighter or wider outline)
            </label>
            <span className="field__help">
              Off by default. Measured on real diabetic photos against professional markings, the app's own edge was more exact (overlap 82.0%, against 79.2% when the
              AI moved it). With this off, the AI still removes areas that are not wound and finds wounds the app missed. With it on, the AI moves the edge only where the
              app is unsure of it.
            </span>
          </div>

          <div className="note">
            <p>
              <strong>What is sent:</strong> the photo made smaller (at most 1024 pixels), a copy with the numbered outlines, and the marking prompt. No file name, date,
              place or camera details. A dark or flat photo is brightened in the sent copy only, so the wound colours read right. The original stays as it is.
            </p>
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

      {s?.demo ? (
        <section className="panel" aria-labelledby="demo" style={{ marginTop: "var(--space-lg)" }}>
          <h2 className="panel__title" id="demo">
            Demo data
          </h2>
          <DemoNote ctx={ctx} />
        </section>
      ) : null}
    </main>
  );
}
