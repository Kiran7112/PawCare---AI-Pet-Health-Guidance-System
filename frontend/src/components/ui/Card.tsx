import type { ReactNode } from "react";
import { cn } from "@/utils/cn";

interface CardProps {
  children: ReactNode;
  className?: string;
  /** Optional left accent bar color (e.g. "border-l-critical"). */
  accentBorder?: string;
  /** Add a subtle hover lift (for clickable / highlight cards). */
  interactive?: boolean;
}

export function Card({ children, className, accentBorder, interactive }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-slate-800 bg-slate-900 shadow-card transition-all duration-300",
        interactive && "hover:-translate-y-0.5 hover:border-slate-700 hover:shadow-card-hover",
        accentBorder && `border-l-4 ${accentBorder}`,
        className,
      )}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: ReactNode;
  icon?: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}

export function CardHeader({
  title,
  icon,
  description,
  action,
  className,
}: CardHeaderProps) {
  return (
    <div
      className={cn(
        "flex items-start justify-between gap-3 border-b border-slate-800 px-5 py-4",
        className,
      )}
    >
      <div className="flex items-start gap-3">
        {icon && (
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-brand-500/15 text-lg leading-none ring-1 ring-brand-500/25">
            {icon}
          </span>
        )}
        <div>
          <h3 className="text-base font-semibold text-slate-100">{title}</h3>
          {description && (
            <p className="mt-0.5 text-sm text-slate-400">{description}</p>
          )}
        </div>
      </div>
      {action}
    </div>
  );
}

export function CardBody({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return <div className={cn("px-5 py-4", className)}>{children}</div>;
}
