import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api } from "../api";
import { FileIcon, Folder } from "../icons";

type Key = "images_dir" | "rules_path" | "examples_dir" | "output_dir";

const FIELDS: { key: Key; label: string; help: string; kind: "folder" | "file" }[] = [
  { key: "images_dir", label: "Photos to mark", help: "The folder of wound photos. Subfolders are included.", kind: "folder" },
  { key: "rules_path", label: "Marking rules", help: "A .md or .txt file with how to mark. Optional.", kind: "file" },
  { key: "examples_dir", label: "Marked examples", help: "Photos with their JSON markings, used to teach the app.", kind: "folder" },
  { key: "output_dir", label: "Save JSON files to", help: "Leave empty to save each JSON next to its photo.", kind: "folder" },
];

export function Setup({ ctx }: { ctx: AppCtx }) {
  const s = ctx.status;
  const [form, setForm] = useState<Record<Key, string>>({ images_dir: "", rules_path: "", examples_dir: "", output_dir: "" });
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (s) setForm({ images_dir: s.settings.images_dir, rules_path: s.settings.rules_path, examples_dir: s.settings.examples_dir, output_dir: s.settings.output_dir });
  }, [s?.settings.images_dir, s?.settings.rules_path, s?.settings.examples_dir, s?.settings.output_dir]); // eslint-disable-line react-hooks/exhaustive-deps

  const browse = async (key: Key, kind: "folder" | "file") => {
    try {
      const r = await api.browse(kind, form[key]);
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
          <p className="page__sub">Where the photos, the rules and the examples live. Nothing leaves this computer.</p>
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
                  placeholder={f.kind === "file" ? "C:\\path\\to\\rules.md" : "C:\\path\\to\\folder"}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button type="button" className="btn" onClick={() => void browse(f.key, f.kind)}>
                  {f.kind === "file" ? <FileIcon /> : <Folder />} Browse
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
              <dt>Rules</dt>
              <dd>{s?.rules.loaded ? `Loaded${s.rules.labels.length ? `, labels: ${s.rules.labels.join(", ")}` : ""}` : "No rules file"}</dd>
            </dl>
          ) : (
            <p className="muted">Checking the hardware</p>
          )}
          {s?.rules.text ? (
            <>
              <hr className="rule" />
              <h3 className="panel__title">The rules</h3>
              <pre className="rules-text">{s.rules.text}</pre>
            </>
          ) : null}
        </section>
      </div>
    </main>
  );
}
