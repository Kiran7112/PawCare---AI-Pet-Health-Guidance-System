import { NavLink } from "react-router-dom";
import { useAssessmentStore } from "@/features/assessment/AssessmentContext";
import { useHealthCheck } from "@/api/assessment";
import { getPathMeta } from "@/utils/format";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/utils/cn";

function navClass({ isActive }: { isActive: boolean }) {
  return cn(
    "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
    isActive
      ? "bg-brand-500/15 text-brand-300"
      : "text-slate-300 hover:bg-slate-800 hover:text-white",
  );
}

export function Header() {
  const { assessment } = useAssessmentStore();
  const health = useHealthCheck();
  const pathMeta = assessment?.path_taken ? getPathMeta(assessment.path_taken) : null;

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800 glass">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <NavLink to="/" className="group flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-gradient text-lg shadow-glow transition-transform group-hover:scale-105">
            🐾
          </span>
          <div className="leading-tight">
            <span className="block text-lg font-extrabold tracking-tight text-white">
              PawCare<span className="gradient-text">+</span>
            </span>
            <span className="hidden text-xs text-slate-400 sm:block">
              Pet Health &amp; Care Guidance
            </span>
          </div>
        </NavLink>

        <div className="flex items-center gap-2 sm:gap-3">
          {pathMeta && (
            <Badge tone="neutral" className="hidden sm:inline-flex">
              {pathMeta.icon} {pathMeta.label}
            </Badge>
          )}
          <nav className="flex items-center gap-1">
            <NavLink to="/" className={navClass} end>
              Assessment
            </NavLink>
            <NavLink to="/about" className={navClass}>
              About
            </NavLink>
          </nav>
          <span
            className="flex items-center gap-1.5 text-xs text-slate-400"
            title={
              health.data?.llm_configured
                ? "Backend online · LLM configured"
                : "Backend status"
            }
          >
            <span
              className={cn(
                "h-2.5 w-2.5 rounded-full ring-2 ring-slate-900",
                health.isSuccess
                  ? health.data?.llm_configured
                    ? "bg-green-500 animate-pulse"
                    : "bg-amber-500"
                  : health.isError
                    ? "bg-red-500"
                    : "bg-slate-300",
              )}
            />
          </span>
        </div>
      </div>
    </header>
  );
}
