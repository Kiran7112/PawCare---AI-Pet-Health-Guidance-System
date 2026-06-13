import type { ReactNode } from "react";
import type { AssessmentResponse } from "@/types/assessment";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import {
  careSeverity,
  formatPercent,
  formatScore,
  riskSeverity,
} from "@/utils/format";

function Field({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="mt-0.5 text-sm text-slate-100">{value || "—"}</dd>
    </div>
  );
}

export function AnalysisTab({ data }: { data: AssessmentResponse }) {
  const { summary, result } = data;
  const profile = summary.pet_profile;
  const risk = summary.health_assessment.health_risk_score;
  const care = summary.health_assessment.care_capability_score;
  const riskSev = riskSeverity(risk);
  const careSev = careSeverity(care);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title="Pet Profile" icon="📋" />
        <CardBody>
          <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            <Field label="Species" value={profile.species} />
            <Field label="Breed" value={profile.breed} />
            <Field label="Age" value={`${profile.age_years} yrs`} />
            <Field label="Weight status" value={profile.weight_status} />
            <Field label="Sex" value={profile.sex} />
            <Field
              label="Living situation"
              value={(result.living_situation as string) ?? "unknown"}
            />
            <Field
              label="Exercise level"
              value={(result.exercise_level as string) ?? "unknown"}
            />
            <Field
              label="Diet type"
              value={(result.diet_type as string) ?? "unknown"}
            />
            <Field
              label="Known conditions"
              value={
                profile.known_conditions.length
                  ? profile.known_conditions.join(", ")
                  : "None reported"
              }
            />
          </dl>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="Health Assessment Metrics" icon="📊" />
        <CardBody>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-4">
              <p className="text-sm text-slate-400">Health Risk Score</p>
              <p className="mt-1 text-xl font-bold text-white">
                {risk !== null ? risk.toFixed(3) : "N/A"}{" "}
                <span className="text-sm font-normal text-slate-400">
                  ({formatPercent(risk)})
                </span>
              </p>
              {riskSev && (
                <Badge
                  className="mt-2"
                  tone={
                    riskSev === "high"
                      ? "danger"
                      : riskSev === "moderate"
                        ? "warning"
                        : "success"
                  }
                >
                  {riskSev.toUpperCase()}
                </Badge>
              )}
            </div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-4">
              <p className="text-sm text-slate-400">Care Capability</p>
              <p className="mt-1 text-xl font-bold text-white">
                {formatScore(care)}
              </p>
              {careSev && (
                <Badge
                  className="mt-2"
                  tone={careSev === "low" ? "warning" : "success"}
                >
                  {careSev.toUpperCase()}
                </Badge>
              )}
            </div>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
