import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
}

export function EmptyState({ icon, title, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-slate-700 py-10 text-center">
      {icon && <div className="text-3xl opacity-80">{icon}</div>}
      <p className="font-medium text-slate-200">{title}</p>
      {description && (
        <p className="max-w-sm text-sm text-slate-400">{description}</p>
      )}
    </div>
  );
}
