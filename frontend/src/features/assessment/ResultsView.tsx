import { Download, RotateCcw } from "lucide-react";
import type { AssessmentResponse } from "@/types/assessment";
import { Button } from "@/components/ui/Button";
import { Alert } from "@/components/ui/Alert";
import { Tabs, type TabItem } from "@/components/ui/Tabs";
import { downloadAssessmentJson } from "@/utils/export";
import { OverviewTab } from "./tabs/OverviewTab";
import { GuidanceTab } from "./tabs/GuidanceTab";
import { AnalysisTab } from "./tabs/AnalysisTab";
import { SummaryTab } from "./tabs/SummaryTab";

interface ResultsViewProps {
  data: AssessmentResponse;
  onNewAssessment: () => void;
}

export function ResultsView({ data, onNewAssessment }: ResultsViewProps) {
  const errors = data.error_messages ?? [];
  const hasErrors = data.error_occurred && errors.length > 0;

  const tabs: TabItem[] = [
    { id: "overview", label: "Overview", icon: "📋", content: <OverviewTab data={data} /> },
    { id: "guidance", label: "Health Guidance", icon: "🎯", content: <GuidanceTab data={data} /> },
    { id: "analysis", label: "Detailed Analysis", icon: "🔬", content: <AnalysisTab data={data} /> },
    { id: "summary", label: "Summary", icon: "📑", content: <SummaryTab data={data} /> },
  ];

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-brand-400">
            PawCare+ Report
          </p>
          <h1 className="text-2xl font-extrabold tracking-tight text-white">
            Assessment Results
          </h1>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={onNewAssessment}>
            <RotateCcw className="h-4 w-4" />
            New assessment
          </Button>
          <Button onClick={() => downloadAssessmentJson(data)}>
            <Download className="h-4 w-4" />
            Download JSON
          </Button>
        </div>
      </div>

      {hasErrors && (
        <Alert variant="warning" title="Assessment completed with warnings">
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {errors.slice(0, 5).map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </Alert>
      )}

      <Tabs tabs={tabs} initialId="overview" />
    </div>
  );
}
