/**
 * Types mirroring the PawCare+ API contract (see backend/api/schemas.py and
 * graph.get_pet_health_summary). The LLM/aggregator outputs are free-form, so
 * the deep guidance content is typed loosely on purpose.
 */

export const CARE_PATHS = {
  CRITICAL: "CRITICAL_CARE_PATH",
  PREVENTIVE: "PREVENTIVE_CARE_PATH",
  WELLNESS: "WELLNESS_PATH",
} as const;

export type CarePath = (typeof CARE_PATHS)[keyof typeof CARE_PATHS] | string;

export interface AssessmentRequest {
  about_pet: string;
  daily_routine: string;
  health_concerns: string;
}

/** Free-form content emitted by an LLM guidance agent. */
export type GuidanceContent = Record<string, unknown>;

export interface PetProfileSummary {
  name: string;
  species: string;
  breed: string;
  age_years: number;
  weight_status: string;
  sex: string;
  known_conditions: string[];
  medications_current: string[];
  allergies_known: string[];
  behavioral_issues: string[];
}

export interface HealthAssessmentSummary {
  health_risk_score: number | null;
  care_capability_score: number | null;
  health_risk_factors: Record<string, number>;
  care_capability_factors: Record<string, number>;
  extraction_confidence: number;
}

export interface PathAnalysisSummary {
  path_taken: CarePath;
  urgency: string;
  path_decision_rationale: string;
  path_thresholds_used: Record<string, unknown>;
}

export interface SystemSummary {
  request_id: string;
  analysis_timestamp: string;
  processing_complete: boolean;
  processing_stage: string;
  error_occurred: boolean;
  error_count: number;
  errors: string[];
}

export interface AssessmentSummary {
  pet_profile: PetProfileSummary;
  health_assessment: HealthAssessmentSummary;
  path_analysis: PathAnalysisSummary;
  outputs: Record<string, GuidanceContent>;
  system: SystemSummary;
}

/** The full workflow state (50+ fields). Indexable for path-specific outputs. */
export interface AssessmentResult {
  request_id?: string;
  analysis_timestamp?: string;
  path_taken?: CarePath;
  health_risk_score?: number | null;
  care_capability_score?: number | null;
  error_occurred?: boolean;
  error_messages?: string[];

  living_situation?: string;
  exercise_level?: string;
  diet_type?: string;

  // Critical-path outputs
  health_risk_analysis_output?: GuidanceContent | null;
  emergency_prep_output?: GuidanceContent | null;
  nutrition_critical_output?: GuidanceContent | null;
  behavioral_coaching_output?: GuidanceContent | null;
  wellness_monitoring_output?: GuidanceContent | null;
  // Preventive-path outputs
  health_assessment_output?: GuidanceContent | null;
  nutrition_preventive_output?: GuidanceContent | null;
  wellness_tracking_output?: GuidanceContent | null;
  // Wellness-path outputs
  wellness_optimization_output?: GuidanceContent | null;
  nutrition_wellness_output?: GuidanceContent | null;
  lifestyle_enrichment_output?: GuidanceContent | null;

  [key: string]: unknown;
}

export interface AssessmentResponse {
  request_id: string;
  path_taken: CarePath | null;
  health_risk_score: number | null;
  care_capability_score: number | null;
  error_occurred: boolean;
  error_messages: string[];
  result: AssessmentResult;
  summary: AssessmentSummary;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  llm_configured: boolean;
}
