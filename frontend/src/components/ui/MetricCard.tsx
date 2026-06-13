import type { ReactNode } from "react";
import { cn } from "@/utils/cn";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  icon?: ReactNode;
  hint?: ReactNode;
  accent?: string; // e.g. "text-critical"
  /** Tint for the icon chip background, e.g. "bg-red-50 text-critical". */
  iconClass?: string;
  className?: string;
}

export function MetricCard({
  label,
  value,
  icon,
  hint,
  accent,
  iconClass,
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        "group rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-card transition-all duration-300 hover:-translate-y-0.5 hover:border-slate-700 hover:shadow-card-hover",
        className,
      )}
    >
      <div className="flex items-center gap-2.5">
        {icon && (
          <span
            className={cn(
              "flex h-9 w-9 items-center justify-center rounded-xl ring-1 ring-inset ring-white/10 transition-transform group-hover:scale-105",
              iconClass ?? "bg-brand-500/15 text-brand-300",
            )}
          >
            {icon}
          </span>
        )}
        <span className="text-sm font-medium text-slate-400">{label}</span>
      </div>
      <div className={cn("mt-3 text-3xl font-extrabold tracking-tight text-white", accent)}>
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
    </div>
  );
}
