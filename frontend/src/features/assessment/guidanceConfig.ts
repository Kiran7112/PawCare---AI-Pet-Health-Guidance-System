import { CARE_PATHS, type CarePath } from "@/types/assessment";

export interface SectionConfig {
  /** Key on the AssessmentResult holding this section's content dict. */
  field: string;
  title: string;
  icon: string;
  /** Fields surfaced as a highlighted callout (e.g. urgency). */
  emphasis?: string[];
}

const CRITICAL_SECTIONS: SectionConfig[] = [
  {
    field: "health_risk_analysis_output",
    title: "Health Risk Analysis",
    icon: "🔴",
    emphasis: ["urgency_timeline"],
  },
  {
    field: "emergency_prep_output",
    title: "Emergency Preparedness",
    icon: "🚨",
    emphasis: ["when_to_call_vet"],
  },
  { field: "nutrition_critical_output", title: "Critical Nutrition Plan", icon: "🥗" },
  { field: "behavioral_coaching_output", title: "Behavioral Coaching", icon: "🐾" },
  { field: "wellness_monitoring_output", title: "Wellness Monitoring", icon: "📊" },
];

const PREVENTIVE_SECTIONS: SectionConfig[] = [
  {
    field: "health_assessment_output",
    title: "Preventive Health Assessment",
    icon: "🩺",
    emphasis: ["recommended_checkups"],
  },
  { field: "nutrition_preventive_output", title: "Preventive Nutrition Guide", icon: "🥗" },
  { field: "wellness_tracking_output", title: "Wellness Tracking Plan", icon: "📋" },
];

const WELLNESS_SECTIONS: SectionConfig[] = [
  { field: "wellness_optimization_output", title: "Wellness Optimization", icon: "🌟" },
  { field: "nutrition_wellness_output", title: "Nutrition Enhancement", icon: "🥕" },
  { field: "lifestyle_enrichment_output", title: "Lifestyle Enrichment", icon: "🎾" },
];

const ALL_SECTIONS = [
  ...CRITICAL_SECTIONS,
  ...PREVENTIVE_SECTIONS,
  ...WELLNESS_SECTIONS,
];

export function getSectionsForPath(path: CarePath | null | undefined): SectionConfig[] {
  switch (path) {
    case CARE_PATHS.CRITICAL:
      return CRITICAL_SECTIONS;
    case CARE_PATHS.PREVENTIVE:
      return PREVENTIVE_SECTIONS;
    case CARE_PATHS.WELLNESS:
      return WELLNESS_SECTIONS;
    default:
      return ALL_SECTIONS;
  }
}
