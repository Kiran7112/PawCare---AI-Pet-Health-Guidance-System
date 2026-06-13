import type { AssessmentResponse } from "@/types/assessment";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { MetricCard } from "@/components/ui/MetricCard";
import {
  formatDate,
  formatPercent,
  formatScore,
  getPathMeta,
  shortId,
} from "@/utils/format";

export function SummaryTab({ data }: { data: AssessmentResponse }) {
  const { summary } = data;
  const profile = summary.pet_profile;
  const pathMeta = getPathMeta(summary.path_analysis.path_taken);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader title="Pet Profile" icon="🐾" />
        <CardBody>
          <div className="grid gap-3 sm:grid-cols-2 text-slate-300">
            <p className="text-sm">
              <span className="font-medium text-slate-400">Species:</span>{" "}
              {profile.species} · {profile.breed}
            </p>
            <p className="text-sm">
              <span className="font-medium text-slate-400">Age:</span>{" "}
              {profile.age_years} years
            </p>
            <p className="text-sm">
              <span className="font-medium text-slate-400">Weight:</span>{" "}
              {profile.weight_status}
            </p>
            <p className="text-sm">
              <span className="font-medium text-slate-400">Conditions:</span>{" "}
              {profile.known_conditions.length
                ? profile.known_conditions.join(", ")
                : "None reported"}
            </p>
          </div>
        </CardBody>
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        <MetricCard
          label="Health Risk"
          value={formatPercent(summary.health_assessment.health_risk_score)}
        />
        <MetricCard
          label="Care Capability"
          value={formatScore(summary.health_assessment.care_capability_score)}
        />
        <MetricCard
          label="Care Path"
          value={
            <span className={pathMeta.accent}>
              {pathMeta.icon} {pathMeta.label}
            </span>
          }
        />
      </div>

      <Card>
        <CardHeader title="Assessment Information" icon="ℹ️" />
        <CardBody>
          <dl className="grid gap-3 sm:grid-cols-2">
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">
                Urgency
              </dt>
              <dd className="text-sm text-slate-100">{pathMeta.urgency}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">
                Request ID
              </dt>
              <dd className="font-mono text-sm text-slate-100">
                {shortId(summary.system.request_id)}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">
                Date
              </dt>
              <dd className="text-sm text-slate-100">
                {formatDate(summary.system.analysis_timestamp)}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-400">
                Status
              </dt>
              <dd className="text-sm text-slate-100">
                {summary.system.processing_complete ? "Complete" : "Incomplete"}
              </dd>
            </div>
          </dl>
        </CardBody>
      </Card>
    </div>
  );
}
