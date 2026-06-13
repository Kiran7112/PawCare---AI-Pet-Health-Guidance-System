import { forwardRef, type ButtonHTMLAttributes } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/utils/cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  fullWidth?: boolean;
}

const VARIANTS: Record<Variant, string> = {
  primary:
    "!bg-brand-gradient bg-brand-600 text-white shadow-glow hover:brightness-110 hover:shadow-card-hover active:brightness-95 disabled:opacity-50 disabled:shadow-none",
  secondary:
    "bg-slate-800 text-slate-100 border border-slate-700 shadow-sm hover:bg-slate-700 hover:border-slate-600 active:bg-slate-600 disabled:opacity-50",
  ghost:
    "bg-brand-500/15 text-brand-200 border border-brand-500/30 hover:bg-brand-500/25 active:bg-brand-500/30 disabled:opacity-50",
  danger:
    "bg-critical text-white shadow-sm hover:bg-red-700 active:bg-red-800 disabled:opacity-50",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 px-3 text-sm",
  md: "h-10 px-4 text-sm",
  lg: "h-12 px-6 text-base",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "primary",
    size = "md",
    loading = false,
    fullWidth = false,
    className,
    children,
    disabled,
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-xl font-semibold transition-all duration-200 disabled:cursor-not-allowed",
        VARIANTS[variant],
        SIZES[size],
        fullWidth && "w-full",
        className,
      )}
      {...rest}
    >
      {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
});
