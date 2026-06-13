import type { AssessmentResponse } from "@/types/assessment";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { ScoreGauge } from "@/components/ui/ScoreGauge";
import {
  careSeverity,
  formatPercent,
  formatScore,
  getPathMeta,
  riskSeverity,
} from "@/utils/format";

const RISK_LABEL: Record<string, string> = {
  high: "High Risk",
  moderate: "Moderate Risk",
  low: "Low Risk",
};
const CARE_LABEL: Record<string, string> = {
  high: "High Capability",
  moderate: "Moderate Capability",
  low: "Low Capability",
};

const RISK_COLOR: Record<string, string> = {
  high: "#dc2626",
  moderate: "#f59e0b",
  low: "#16a34a",
};

const BANNER_GRADIENT: Record<string, string> = {
  CRITICAL_CARE_PATH: "from-rose-500 to-red-600",
  PREVENTIVE_CARE_PATH: "from-amber-400 to-orange-500",
  WELLNESS_PATH: "from-emerald-500 to-green-600",
};

export function OverviewTab({ data }: { data: AssessmentResponse }) {
  const { summary } = data;
  const risk = summary.health_assessment.health_risk_score;
  const care = summary.health_assessment.care_capability_score;
  const pathMeta = getPathMeta(summary.path_analysis.path_taken);
  const riskSev = riskSeverity(risk);
  const careSev = careSeverity(care);
  const gradient = BANNER_GRADIENT[pathMeta.key] ?? "from-slate-500 to-slate-600";

  return (
    <div className="space-y-6">
      {/* Path banner */}
      <div
        className={`relative overflow-hidden rounded-2xl bg-gradient-to-br ${gradient} p-6 text-white shadow-card`}
      >
        <span
          className="pointer-events-none absolute -right-4 -top-6 text-[7rem] opacity-20"
          aria-hidden
        >
          {pathMeta.icon}
        </span>
        <div className="relative">
          <span className="inline-flex items-center rounded-full bg-white/20 px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide">
            Care Path
          </span>
          <h2 className="mt-2 text-2xl font-extrabold">{pathMeta.label}</h2>
          <p className="mt-1 max-w-xl text-sm text-white/90">
            {pathMeta.description}
          </p>
          <p className="mt-3 inline-flex items-center gap-2 rounded-lg bg-white/15 px-3 py-1.5 text-sm font-medium">
            ⏱ {pathMeta.urgency}
          </p>
        </div>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        {/* Scores card with gauges */}
        <Card>
          <CardHeader title="Health &amp; Care Scores" icon="📊" />
          <CardBody>
            <div className="flex flex-wrap items-center justify-around gap-4">
              <div className="flex flex-col items-center gap-2">
                <ScoreGauge
                  value={(risk ?? 0) * 100}
                  display={formatPercent(risk)}
                  caption="Risk"
                  color={riskSev ? RISK_COLOR[riskSev] : "#94a3b8"}
                />
                {riskSev && (
                  <Badge
                    tone={
                      riskSev === "high"
                        ? "danger"
                        : riskSev === "moderate"
                          ? "warning"
                          : "success"
                    }
                  >
                    {RISK_LABEL[riskSev]}
                  </Badge>
                )}
              </div>
              <div className="flex flex-col items-center gap-2">
                <ScoreGauge
                  value={care ?? 0}
                  display={formatScore(care, "")}
                  caption="Care / 100"
                  color={careSev === "low" ? "#f59e0b" : "#16a34a"}
                />
                {careSev && (
                  <Badge tone={careSev === "low" ? "warning" : "success"}>
                    {CARE_LABEL[careSev]}
                  </Badge>
                )}
              </div>
            </div>
          </CardBody>
        </Card>

        {/* Details card */}
        <Card>
          <CardHeader title="What this means" icon="🩺" />
          <CardBody className="space-y-4">
            <RiskFactors factors={summary.health_assessment.health_risk_factors} />
            {summary.path_analysis.path_decision_rationale && (
              <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-3 text-sm text-slate-300">
                <span className="font-semibold text-slate-200">
                  Path decision:{" "}
                </span>
                {summary.path_analysis.path_decision_rationale}
              </div>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function RiskFactors({ factors }: { factors: Record<string, number> }) {
  const top = Object.entries(factors ?? {}).slice(0, 4);
  if (top.length === 0) {
    return (
      <p className="text-sm text-slate-400">
        No specific risk factors were highlighted for this assessment.
      </p>
    );
  }
  const max = Math.max(...top.map(([, v]) => (typeof v === "number" ? v : 0)), 1);
  return (
    <div>
      <p className="mb-2 text-sm font-semibold text-slate-200">Key risk factors</p>
      <ul className="space-y-2.5">
        {top.map(([factor, importance]) => {
          const val = typeof importance === "number" ? importance : 0;
          return (
            <li key={factor}>
              <div className="mb-1 flex justify-between gap-3 text-sm">
                <span className="text-slate-300">{factor}</span>
                <span className="tabular-nums text-slate-500">
                  {val.toFixed(2)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-brand-gradient"
                  style={{ width: `${Math.max(6, (val / max) * 100)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
