import { useEffect, useState } from "react";
import { Spinner } from "@/components/ui/Spinner";
import { Card, CardBody } from "@/components/ui/Card";
import { cn } from "@/utils/cn";

const STAGES = [
  "Validating your input",
  "Extracting pet profile",
  "Scoring health risk & care capability",
  "Routing to the right care path",
  "Generating personalized guidance",
];

/**
 * Friendly progress indicator. The backend call is a single request, so we
 * advance through expected stages on a timer to communicate liveness (typical
 * assessment takes ~5–15s depending on the path).
 */
export function AssessmentProgress() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setStage((s) => Math.min(s + 1, STAGES.length - 1));
    }, 2500);
    return () => clearInterval(id);
  }, []);

  const progress = ((stage + 1) / STAGES.length) * 100;

  return (
    <Card className="overflow-hidden animate-scale-in">
      <div className="h-1.5 w-full bg-brand-gradient" />
      <CardBody className="py-10">
        <div className="mx-auto flex max-w-md flex-col items-center gap-5 text-center">
          <div className="relative flex h-20 w-20 items-center justify-center">
            <span className="absolute inset-0 rounded-full bg-brand-500/20" />
            <span className="absolute inset-0 animate-ping rounded-full bg-brand-500/30 opacity-60" />
            <span className="relative text-4xl animate-float" aria-hidden>
              🐾
            </span>
          </div>
          <div>
            <p className="text-lg font-bold text-white">
              Analyzing your pet&apos;s health profile…
            </p>
            <p className="mt-1 text-sm text-slate-400">
              Our AI agents are working through the assessment.
            </p>
          </div>

          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-brand-gradient transition-all duration-700 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>

          <ul className="w-full space-y-2.5 text-left">
            {STAGES.map((label, i) => {
              const done = i < stage;
              const active = i === stage;
              return (
                <li
                  key={label}
                  className={cn(
                    "flex items-center gap-2.5 rounded-lg px-2 py-1 text-sm transition-colors",
                    active && "bg-brand-500/15 font-semibold text-brand-200",
                    done && "text-slate-500",
                    !done && !active && "text-slate-500",
                  )}
                >
                  {active ? (
                    <Spinner className="h-4 w-4" />
                  ) : done ? (
                    <span className="flex h-4 w-4 items-center justify-center rounded-full bg-wellness text-[10px] text-white">
                      ✓
                    </span>
                  ) : (
                    <span className="h-2 w-2 rounded-full bg-slate-300" />
                  )}
                  {label}
                </li>
              );
            })}
          </ul>
        </div>
      </CardBody>
    </Card>
  );
}
