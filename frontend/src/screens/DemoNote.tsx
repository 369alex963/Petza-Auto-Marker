import { useState } from "react";
import type { AppCtx } from "../App";
import { api } from "../api";

/** Shown while the app still uses the demo photos: what they are, and how to start clean. */
export function DemoNote({ ctx }: { ctx: AppCtx }) {
  const [busy, setBusy] = useState(false);
  if (!ctx.status?.demo) return null;

  const remove = async () => {
    if (
      !window.confirm(
        "Remove all demo data and start clean?\n\nThis deletes the demo folder (its photos, examples and JSON files, and anything you added there), the demo examples in the training pool and every trained model, because all of them learned from the demo. Your prompts and AI settings stay.\n\nAfterwards, choose your own folders and press Teach.",
      )
    )
      return;
    setBusy(true);
    try {
      await api.removeDemo();
      await ctx.refresh();
      ctx.toast("Demo data removed. The app is clean.");
    } catch (e) {
      ctx.toast((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="note note--demo" role="note">
      <p>
        <strong>These are demo photos</strong>, so you can try everything right away. The demo model learned from only 42 example photos, with no wound
        data before them. Once it learns from your own marked photos, bulk marking becomes far more accurate.
      </p>
      <button type="button" className="btn" disabled={busy || ctx.job?.status === "running"} onClick={() => void remove()}>
        Remove demo data
      </button>
    </div>
  );
}
