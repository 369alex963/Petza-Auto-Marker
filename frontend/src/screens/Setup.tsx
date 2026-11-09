import { useEffect, useRef, useState } from "react";
import type { AppCtx } from "../App";
import { api, type Prompts } from "../api";
import { Folder } from "../icons";

type Key = "images_dir" | "examples_dir" | "output_dir";

const FIELDS: { key: Key; label: string; help: string }[] = [
  { key: "examples_dir", label: "Marked examples", help: "Photos with their JSON markings, used to teach the app." },
  { key: "images_dir", label: "Photos to mark", help: "The folder of new wound photos. Subfolders are included." },
  { key: "output_dir", label: "Save JSON files to", help: "Leave empty to save each JSON next to its photo." },
];

export function Setup({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [form, setForm] = useState<Record<Key, string>>({ images_dir: "", examples_dir: "", output_dir: "" });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (s) setForm({ images_dir: s.settings.images_dir, examples_dir: s.settings.examples_dir, output_dir: s.settings.output_dir });
  }, [s?.settings.images_dir, s?.settings.examples_dir, s?.settings.output_dir]); // eslint-disable-line react-hooks/exhaustive-deps

  const browse = async (key: Key) => {
    try {
      const r = await api.browse("folder", form[key]);
      if (r.path) setForm((f) => ({ ...f, [key]: r.path }));
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await api.saveSettings(form);
      await ctx.refresh();
      ctx.toast("Folders saved");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const d = s?.device;
  return (
    <main className="page">
      <div className="page__head">
        <div>
          <h1 className="page__title">Folders</h1>
          <p className="page__sub">Where the examples and the photos live, and the marking instructions for the AI. Nothing leaves this computer unless the AI second opinion is on.</p>
        </div>
      </div>
      <div className="cols">
        <form className="panel" onSubmit={save} noValidate>
          {FIELDS.map((f) => (
            <div className="field" key={f.key}>
              <label className="field__label" htmlFor={f.key}>
                {f.label}
              </label>
              <div className="field__row">
                <input
                  id={f.key}
                  name={f.key}
                  className="input"
                  value={form[f.key]}
                  onChange={(e) => setForm((x) => ({ ...x, [f.key]: e.target.value }))}
                  placeholder="C:\path\to\folder"
                  autoComplete="off"
                  spellCheck={false}
                />
                <button type="button" className="btn" onClick={() => void browse(f.key)}>
                  <Folder /> Browse
                </button>
              </div>
              <span className="field__help">{f.help}</span>
            </div>
          ))}
          {error ? (
            <div className="note note--error" role="alert">
              <p>{error}</p>
            </div>
          ) : null}
          <div>
            <button type="submit" className="btn btn--primary" disabled={saving}>
              {saving ? "Saving" : "Save folders"}
            </button>
          </div>
        </form>

        <section className="panel" aria-labelledby="hw">
          <h2 className="panel__title" id="hw">
            This computer
          </h2>
          {d ? (
            <dl className="facts">
              <dt>Marking runs on</dt>
              <dd>{d.inference_label}</dd>
              <dt>Teaching runs on</dt>
              <dd>{d.train_device === "cuda" ? `${d.gpu_name} (${d.gpu_memory_gb} GB)` : `CPU, ${d.cpu_threads} threads`}</dd>
              <dt>Model</dt>
              <dd>{s?.model ? `${s.model.run_id}, learned from ${s.model.train_count} photos` : "Not taught yet"}</dd>
            </dl>
          ) : (
            <p className="muted">Checking the hardware</p>
          )}
          <hr className="rule" />
          <PromptEditor onChange={() => void ctx.refresh()} toast={ctx.toast} />
        </section>
      </div>
    </main>
  );
}

function PromptEditor({ onChange, toast }: { onChange: () => void; toast: (m: string) => void }) {
  const [store, setStore] = useState<Prompts | null>(null);
  const [text, setText] = useState("");
  const [state, setState] = useState<"saved" | "saving" | "edited">("saved");
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const apply = (p: Prompts) => {
    setStore(p);
    setText(p.prompts[p.active] ?? "");
    setState("saved");
  };

  useEffect(() => {
    api.prompts().then(apply).catch((e: Error) => setError(e.message));
    return () => window.clearTimeout(timer.current);
  }, []);

  const run = async (fn: () => Promise<Prompts>, msg?: string) => {
    window.clearTimeout(timer.current);
    setError(null);
    try {
      apply(await fn());
      onChange();
      if (msg) toast(msg);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  // Every edit saves itself a moment after typing stops.
  const edit = (value: string) => {
    if (!store) return;
    setText(value);
    setState("edited");
    window.clearTimeout(timer.current);
    const name = store.active;
    timer.current = window.setTimeout(async () => {
      setState("saving");
      try {
        setStore(await api.putPrompt(name, value));
        setState("saved");
      } catch (e) {
        setError((e as Error).message);
        setState("edited");
      }
    }, 700);
  };

  const saveAs = () => {
    const name = window.prompt("A name for this prompt:", "")?.trim();
    if (!name) return;
    if (store?.prompts[name] !== undefined && !window.confirm(`Replace the prompt “${name}”?`)) return;
    void run(() => api.putPrompt(name, text), "Prompt saved");
  };

  if (!store) return error ? <p className="muted">{error}</p> : <p className="muted">Loading the prompt</p>;
  const isDefault = store.active === store.default_name;

  return (
    <div className="prompt">
      <h3 className="panel__title">Marking prompt for the AI</h3>
      <p className="muted">
        The AI second opinion (Settings) reads this when it checks each marking. Claude and ChatGPT both read Hebrew well. The local model does not read text; it learns from the marked examples.
      </p>
      <div className="field__row">
        <label className="sr-only" htmlFor="prompt-pick">
          Prompt
        </label>
        <select id="prompt-pick" className="input" value={store.active} onChange={(e) => void run(() => api.selectPrompt(e.target.value))}>
          {Object.keys(store.prompts).map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
        <button type="button" className="btn" onClick={saveAs}>
          Save as new
        </button>
        {isDefault ? (
          <button
            type="button"
            className="btn btn--quiet"
            disabled={text === store.default_text}
            onClick={() => window.confirm("Put back the original default text?") && void run(() => api.putPrompt(store.default_name, store.default_text), "Default restored")}
          >
            Restore
          </button>
        ) : (
          <button
            type="button"
            className="btn btn--quiet"
            onClick={() => window.confirm(`Delete the prompt “${store.active}”?`) && void run(() => api.deletePrompt(store.active), "Prompt deleted")}
          >
            Delete
          </button>
        )}
      </div>
      <label className="sr-only" htmlFor="prompt-text">
        Prompt text
      </label>
      <textarea id="prompt-text" className="input prompt__text" dir="auto" rows={18} value={text} onChange={(e) => edit(e.target.value)} spellCheck={false} />
      <span className="field__help" role="status">
        {error ? error : state === "saved" ? "Saved in the app." : state === "saving" ? "Saving" : "Editing"}
      </span>
    </div>
  );
}
