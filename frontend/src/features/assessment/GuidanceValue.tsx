import { humanize } from "@/utils/format";

/**
 * Renders a single value from a free-form LLM guidance object. The agents are
 * not guaranteed to return a fixed shape, so we render defensively by runtime
 * type: strings become paragraphs, arrays become bullet lists, nested objects
 * recurse. This keeps the UI resilient to schema drift.
 */
export function GuidanceValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return null;

  if (typeof value === "string") {
    const text = value.trim();
    if (!text) return null;
    return (
      <div className="prose-paw text-sm">
        {text.split(/\n+/).map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    );
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return <span className="text-sm text-slate-300">{String(value)}</span>;
  }

  if (Array.isArray(value)) {
    const items = value.filter((v) => v !== null && v !== undefined && v !== "");
    if (items.length === 0) return null;
    return (
      <ul className="space-y-1.5">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2 text-sm text-slate-300">
            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-brand-400" />
            <span className="min-w-0">
              {typeof item === "object" ? (
                <GuidanceObject value={item as Record<string, unknown>} />
              ) : (
                String(item)
              )}
            </span>
          </li>
        ))}
      </ul>
    );
  }

  if (typeof value === "object") {
    return <GuidanceObject value={value as Record<string, unknown>} />;
  }

  return null;
}

/** Renders an object as labelled key/value rows. */
function GuidanceObject({ value }: { value: Record<string, unknown> }) {
  const entries = Object.entries(value).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );
  if (entries.length === 0) return null;

  return (
    <div className="space-y-1">
      {entries.map(([k, v]) => (
        <div key={k} className="text-sm">
          <span className="font-medium text-slate-400">{humanize(k)}: </span>
          {typeof v === "object" ? (
            <div className="mt-1 pl-3">
              <GuidanceValue value={v} />
            </div>
          ) : (
            <span className="text-slate-300">{String(v)}</span>
          )}
        </div>
      ))}
    </div>
  );
}
