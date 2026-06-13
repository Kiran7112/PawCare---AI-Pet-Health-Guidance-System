import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";
import type {
  AssessmentRequest,
  AssessmentResponse,
  HealthResponse,
} from "@/types/assessment";

export const assessmentKeys = {
  health: ["health"] as const,
};

/** Backend liveness + whether an LLM key is configured. */
export function useHealthCheck() {
  return useQuery({
    queryKey: assessmentKeys.health,
    queryFn: ({ signal }) => apiClient.get<HealthResponse>("/api/health", signal),
    staleTime: 60_000,
    retry: 1,
  });
}

/** Run a pet assessment. This is a long-running, side-effecting call -> mutation. */
export function useAssessment() {
  return useMutation<AssessmentResponse, Error, AssessmentRequest>({
    mutationFn: (payload) =>
      apiClient.post<AssessmentResponse>("/api/assess", payload),
  });
}
