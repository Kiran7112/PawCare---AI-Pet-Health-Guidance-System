import type { AssessmentResponse } from "@/types/assessment";

/** Build the downloadable JSON payload for an assessment. */
export function buildExportPayload(response: AssessmentResponse) {
  return {
    metadata: {
      request_id: response.request_id,
      timestamp: response.result.analysis_timestamp ?? null,
      exported_at: new Date().toISOString(),
      version: "1.0.0",
    },
    assessment: response.result,
    summary: response.summary,
  };
}

export function downloadAssessmentJson(response: AssessmentResponse): void {
  const payload = buildExportPayload(response);
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pawcare_assessment_${response.request_id.slice(0, 8)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
