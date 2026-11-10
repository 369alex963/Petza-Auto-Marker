import { useEffect, useRef, useState } from "react";
import { api, type Prompts } from "../api";

export function PromptEditor({ onChange, toast }: { onChange: () => void; toast: (m: string) => void }) {
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
        The AI second opinion reads this when it checks each marking. Claude, ChatGPT and Groq all read Hebrew. The local model does not read text; it learns from the marked examples.
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
