import type { ReactNode } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  XCircle,
} from "lucide-react";
import { cn } from "@/utils/cn";

type Variant = "info" | "success" | "warning" | "error";

const STYLES: Record<Variant, { wrap: string; icon: ReactNode }> = {
  info: {
    wrap: "bg-brand-500/10 border-brand-500/30 text-brand-200",
    icon: <Info className="h-5 w-5 text-brand-400" aria-hidden />,
  },
  success: {
    wrap: "bg-green-500/10 border-green-500/30 text-green-200",
    icon: <CheckCircle2 className="h-5 w-5 text-green-400" aria-hidden />,
  },
  warning: {
    wrap: "bg-amber-500/10 border-amber-500/30 text-amber-200",
    icon: <AlertTriangle className="h-5 w-5 text-amber-400" aria-hidden />,
  },
  error: {
    wrap: "bg-red-500/10 border-red-500/30 text-red-200",
    icon: <XCircle className="h-5 w-5 text-red-400" aria-hidden />,
  },
};

interface AlertProps {
  variant?: Variant;
  title?: ReactNode;
  children?: ReactNode;
  className?: string;
}

export function Alert({ variant = "info", title, children, className }: AlertProps) {
  const style = STYLES[variant];
  return (
    <div
      role={variant === "error" ? "alert" : "status"}
      className={cn(
        "flex items-start gap-3 rounded-lg border px-4 py-3 text-sm",
        style.wrap,
        className,
      )}
    >
      <span className="mt-0.5 shrink-0">{style.icon}</span>
      <div>
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cn(!!title && "mt-0.5")}>{children}</div>}
      </div>
    </div>
  );
}
