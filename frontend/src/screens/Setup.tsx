import { useEffect, useState } from "react";
import type { AppCtx } from "../App";
import { api } from "../api";
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
          <p className="page__sub">Where the marked examples and the photos live. Nothing leaves this computer.</p>
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
        </section>
      </div>
    </main>
  );
}
