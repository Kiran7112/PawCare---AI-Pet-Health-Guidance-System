import type { GuidanceContent } from "@/types/assessment";
import { Card, CardBody, CardHeader } from "@/components/ui/Card";
import { Alert } from "@/components/ui/Alert";
import { humanize } from "@/utils/format";
import { GuidanceValue } from "./GuidanceValue";
import type { SectionConfig } from "./guidanceConfig";

interface GuidanceSectionProps {
  config: SectionConfig;
  content: GuidanceContent;
}

/**
 * Renders one guidance card. Emphasis fields (e.g. urgency) are pulled out into
 * a callout; everything else is rendered as a two-column grid of labelled blocks.
 */
export function GuidanceSection({ config, content }: GuidanceSectionProps) {
  const emphasis = new Set(config.emphasis ?? []);

  const entries = Object.entries(content).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );

  const emphasized = entries.filter(([k]) => emphasis.has(k));
  const regular = entries.filter(([k]) => !emphasis.has(k));

  if (entries.length === 0) return null;

  return (
    <Card interactive accentBorder="border-l-brand-500" className="animate-fade-in">
      <CardHeader title={config.title} icon={config.icon} />
      <CardBody className="space-y-4">
        {emphasized.map(([k, v]) => (
          <Alert key={k} variant="warning" title={humanize(k)}>
            <GuidanceValue value={v} />
          </Alert>
        ))}

        <div className="grid gap-3 sm:grid-cols-2">
          {regular.map(([k, v]) => (
            <div
              key={k}
              className="min-w-0 rounded-xl border border-slate-800 bg-slate-950/40 p-3.5"
            >
              <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-100">
                <span className="h-1.5 w-1.5 rounded-full bg-brand-400" />
                {humanize(k)}
              </h4>
              <GuidanceValue value={v} />
            </div>
          ))}
        </div>
      </CardBody>
    </Card>
  );
}
