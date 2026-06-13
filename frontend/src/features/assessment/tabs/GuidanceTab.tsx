import type { AssessmentResponse, GuidanceContent } from "@/types/assessment";
import { EmptyState } from "@/components/ui/EmptyState";
import { getPathMeta } from "@/utils/format";
import { getSectionsForPath } from "../guidanceConfig";
import { GuidanceSection } from "../GuidanceSection";

export function GuidanceTab({ data }: { data: AssessmentResponse }) {
  const path = data.result.path_taken ?? data.path_taken;
  const pathMeta = getPathMeta(path);
  const sections = getSectionsForPath(path);

  const rendered = sections
    .map((config) => {
      const content = data.result[config.field] as GuidanceContent | null | undefined;
      if (!content || typeof content !== "object" || Object.keys(content).length === 0) {
        return null;
      }
      return <GuidanceSection key={config.field} config={config} content={content} />;
    })
    .filter(Boolean);

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-white">
          {pathMeta.icon} Personalized Health Guidance
        </h2>
        <p className="text-sm text-slate-400">{pathMeta.urgency}</p>
      </div>

      {rendered.length > 0 ? (
        rendered
      ) : (
        <EmptyState
          icon="📭"
          title="No detailed guidance available"
          description="The guidance agents did not return content for this assessment. This can happen if the LLM call failed — check the warnings panel above."
        />
      )}
    </div>
  );
}
