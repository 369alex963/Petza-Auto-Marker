type P = { size?: number; title?: string };

const svg = (size: number, title: string | undefined, children: React.ReactNode, fill = false) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill={fill ? "currentColor" : "none"}
    stroke={fill ? "none" : "currentColor"}
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden={title ? undefined : true}
    role={title ? "img" : undefined}
  >
    {title ? <title>{title}</title> : null}
    {children}
  </svg>
);

export const Heart = ({ size = 18, title }: P) =>
  svg(size, title, <path d="M12 20.5S3 15.1 3 9.2A4.7 4.7 0 0 1 12 6.6a4.7 4.7 0 0 1 9 2.6c0 5.9-9 11.3-9 11.3z" />, true);

export const HeartOutline = ({ size = 18, title }: P) =>
  svg(size, title, <path d="M12 20.5S3 15.1 3 9.2A4.7 4.7 0 0 1 12 6.6a4.7 4.7 0 0 1 9 2.6c0 5.9-9 11.3-9 11.3z" />);

/** The smiley reads the model's confidence: happy, unsure, or worried. */
export function Smiley({ confidence, size = 20, threshold = 0.75 }: { confidence: number | null; size?: number; threshold?: number }) {
  const mood = confidence == null ? "none" : confidence >= threshold ? "happy" : confidence >= threshold * 0.75 ? "unsure" : "worried";
  const mouth =
    mood === "happy" ? "M8 14.5c1.1 1.5 2.4 2.2 4 2.2s2.9-.7 4-2.2" : mood === "unsure" ? "M8.5 15.5h7" : "M8 16.8c1.1-1.4 2.4-2 4-2s2.9.6 4 2";
  const label =
    confidence == null ? "Not marked yet" : `${Math.round(confidence * 100)}% sure${mood === "happy" ? "" : mood === "unsure" ? ", worth a look" : ", please check"}`;
  return svg(
    size,
    label,
    <>
      <circle cx="12" cy="12" r="9.2" />
      <circle cx="9" cy="10" r="0.6" fill="currentColor" />
      <circle cx="15" cy="10" r="0.6" fill="currentColor" />
      {mood === "none" ? null : <path d={mouth} />}
    </>,
  );
}

export const Folder = ({ size = 18 }: P) =>
  svg(size, undefined, <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6H9l2 2.2h8.5A1.5 1.5 0 0 1 21 9.7v8.8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z" />);
export const FileIcon = ({ size = 18 }: P) =>
  svg(size, undefined, <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5" /></>);
export const Lasso = ({ size = 18 }: P) =>
  svg(size, undefined, <><path d="M4 9.5 10 4l9 4-2 9-8 2-5-5z" /><circle cx="4" cy="9.5" r="1.6" fill="currentColor" /><circle cx="19" cy="8" r="1.6" fill="currentColor" /><circle cx="9" cy="19" r="1.6" fill="currentColor" /></>);
export const Cursor = ({ size = 18 }: P) => svg(size, undefined, <path d="M5 3l14 7-6 2-2 6z" />);
export const Undo = ({ size = 18 }: P) => svg(size, undefined, <><path d="M9 14 4 9l5-5" /><path d="M4 9h10a6 6 0 0 1 0 12h-3" /></>);
export const Redo = ({ size = 18 }: P) => svg(size, undefined, <><path d="m15 14 5-5-5-5" /><path d="M20 9H10a6 6 0 0 0 0 12h3" /></>);
export const Trash = ({ size = 18 }: P) =>
  svg(size, undefined, <><path d="M4 7h16" /><path d="M10 11v6M14 11v6" /><path d="M6 7l1 13h10l1-13" /><path d="M9 7V4h6v3" /></>);
export const Fit = ({ size = 18 }: P) =>
  svg(size, undefined, <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />);
export const Eye = ({ size = 18 }: P) =>
  svg(size, undefined, <><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></>);
export const ChevronLeft = ({ size = 18 }: P) => svg(size, undefined, <path d="m15 18-6-6 6-6" />);
export const ChevronRight = ({ size = 18 }: P) => svg(size, undefined, <path d="m9 18 6-6-6-6" />);
export const Sparkle = ({ size = 18 }: P) =>
  svg(size, undefined, <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" />);
export const Stop = ({ size = 16 }: P) => svg(size, undefined, <rect x="6" y="6" width="12" height="12" rx="2" />);
export const Check = ({ size = 16 }: P) => svg(size, undefined, <path d="m5 12.5 4.5 4.5L19 7.5" />);
