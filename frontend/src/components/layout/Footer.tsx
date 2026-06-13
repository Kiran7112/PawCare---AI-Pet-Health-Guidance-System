export function Footer() {
  return (
    <footer className="mt-16 border-t border-slate-800 bg-slate-900/60 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 py-6 text-xs text-slate-400 sm:flex-row sm:px-6">
        <span className="flex items-center gap-1.5 font-semibold text-slate-300">
          <span aria-hidden>🐾</span> PawCare<span className="gradient-text">+</span>
        </span>
        <span className="text-center">
          ⚠️ Guidance only — not a substitute for professional veterinary care.
        </span>
        <span className="rounded-full bg-slate-800 px-2 py-0.5 font-medium text-slate-300">
          v1.0.0
        </span>
      </div>
    </footer>
  );
}
