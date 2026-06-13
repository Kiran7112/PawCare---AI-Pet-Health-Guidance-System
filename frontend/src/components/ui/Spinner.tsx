import { Loader2 } from "lucide-react";
import { cn } from "@/utils/cn";

export function Spinner({ className }: { className?: string }) {
  return (
    <Loader2
      className={cn("h-5 w-5 animate-spin text-brand-400", className)}
      aria-hidden
    />
  );
}

interface LoadingStateProps {
  title?: string;
  subtitle?: string;
}

export function LoadingState({
  title = "Loading…",
  subtitle,
}: LoadingStateProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex flex-col items-center justify-center gap-3 py-16 text-center"
    >
      <Spinner className="h-8 w-8" />
      <p className="font-medium text-slate-200">{title}</p>
      {subtitle && <p className="max-w-sm text-sm text-slate-400">{subtitle}</p>}
    </div>
  );
}
