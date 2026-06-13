import { forwardRef, useId, type TextareaHTMLAttributes, type ReactNode } from "react";
import { cn } from "@/utils/cn";

interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label: ReactNode;
  hint?: ReactNode;
  error?: string;
}

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(
  function TextArea({ label, hint, error, className, id, ...rest }, ref) {
    const generatedId = useId();
    const fieldId = id ?? generatedId;
    const describedBy = error
      ? `${fieldId}-error`
      : hint
        ? `${fieldId}-hint`
        : undefined;

    return (
      <div className="flex flex-col">
        <label htmlFor={fieldId} className="mb-1.5 text-sm font-semibold text-slate-200">
          {label}
        </label>
        <textarea
          id={fieldId}
          ref={ref}
          aria-invalid={!!error}
          aria-describedby={describedBy}
          className={cn(
            "min-h-[120px] w-full resize-y rounded-lg border bg-slate-950/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500",
            "transition-colors focus:border-brand-400",
            error ? "border-red-500/60" : "border-slate-700",
            className,
          )}
          {...rest}
        />
        {error ? (
          <p id={`${fieldId}-error`} className="mt-1 text-xs text-red-400">
            {error}
          </p>
        ) : hint ? (
          <p id={`${fieldId}-hint`} className="mt-1 text-xs text-slate-400">
            {hint}
          </p>
        ) : null}
      </div>
    );
  },
);
