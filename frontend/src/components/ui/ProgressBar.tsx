import { cn } from "@/utils/cn";

interface ProgressBarProps {
  /** 0..100 */
  value: number;
  barClass?: string;
  className?: string;
  label?: string;
}

export function ProgressBar({ value, barClass, className, label }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value));
  return (
    <div
      className={cn("h-2.5 w-full overflow-hidden rounded-full bg-slate-800", className)}
      role="progressbar"
      aria-valuenow={Math.round(clamped)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div
        className={cn("h-full rounded-full transition-all", barClass ?? "bg-brand-500")}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
}
