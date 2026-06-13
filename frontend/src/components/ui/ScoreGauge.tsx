import { cn } from "@/utils/cn";

interface ScoreGaugeProps {
  /** 0..100 fill percentage. */
  value: number;
  /** Big centered label (e.g. "39%" or "62"). */
  display: string;
  /** Small caption under the value. */
  caption?: string;
  /** Stroke color class via CSS var — pass a hex. */
  color?: string;
  size?: number;
  className?: string;
}

/** Lightweight SVG donut gauge — no charting dependency. */
export function ScoreGauge({
  value,
  display,
  caption,
  color = "#2554eb",
  size = 132,
  className,
}: ScoreGaugeProps) {
  const stroke = 11;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, value));
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div
      className={cn("relative inline-flex items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={stroke}
          className="text-slate-800"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.22,1,0.36,1)" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-extrabold text-white">{display}</span>
        {caption && (
          <span className="mt-0.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">
            {caption}
          </span>
        )}
      </div>
    </div>
  );
}
