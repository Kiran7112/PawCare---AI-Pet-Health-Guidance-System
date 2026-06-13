import { CARE_PATHS, type CarePath } from "@/types/assessment";

export interface PathMeta {
  key: CarePath;
  label: string;
  icon: string;
  /** Tailwind text/border/bg accent classes. */
  accent: string;
  accentBg: string;
  accentBorder: string;
  description: string;
  urgency: string;
}

const UNKNOWN_PATH: PathMeta = {
  key: "UNKNOWN",
  label: "Unknown",
  icon: "❓",
  accent: "text-slate-300",
  accentBg: "bg-slate-800",
  accentBorder: "border-slate-600",
  description: "Care path could not be determined.",
  urgency: "Unknown",
};

const PATH_META: Record<string, PathMeta> = {
  [CARE_PATHS.CRITICAL]: {
    key: CARE_PATHS.CRITICAL,
    label: "Critical Care",
    icon: "🔴",
    accent: "text-critical",
    accentBg: "bg-red-50",
    accentBorder: "border-red-300",
    description:
      "High-risk pet requiring prompt veterinary attention and urgent care guidance.",
    urgency: "IMMEDIATE — veterinary attention within 24 hours",
  },
  [CARE_PATHS.PREVENTIVE]: {
    key: CARE_PATHS.PREVENTIVE,
    label: "Preventive Care",
    icon: "🟡",
    accent: "text-amber-600",
    accentBg: "bg-amber-50",
    accentBorder: "border-amber-300",
    description:
      "Moderate-risk pet needing proactive management and preventive measures.",
    urgency: "SCHEDULED — vet visit within 1–2 weeks",
  },
  [CARE_PATHS.WELLNESS]: {
    key: CARE_PATHS.WELLNESS,
    label: "Wellness",
    icon: "🟢",
    accent: "text-wellness",
    accentBg: "bg-green-50",
    accentBorder: "border-green-300",
    description:
      "Low-risk, healthy pet — focus on optimization, enrichment and longevity.",
    urgency: "ROUTINE — regular wellness maintenance",
  },
};

export function getPathMeta(path: CarePath | null | undefined): PathMeta {
  if (!path) return UNKNOWN_PATH;
  return PATH_META[path] ?? UNKNOWN_PATH;
}

export type Severity = "low" | "moderate" | "high";

/** Health risk score is 0..1. Mirrors thresholds in workflow.py (0.3 / 0.6). */
export function riskSeverity(score: number | null | undefined): Severity | null {
  if (score === null || score === undefined || Number.isNaN(score)) return null;
  if (score > 0.6) return "high";
  if (score > 0.3) return "moderate";
  return "low";
}

/** Care capability is 0..100; higher is better. */
export function careSeverity(score: number | null | undefined): Severity | null {
  if (score === null || score === undefined || Number.isNaN(score)) return null;
  if (score >= 80) return "high"; // high capability (good)
  if (score >= 50) return "moderate";
  return "low";
}

export function formatPercent(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return "N/A";
  return `${(score * 100).toFixed(1)}%`;
}

export function formatScore(
  score: number | null | undefined,
  suffix = "/100",
): string {
  if (score === null || score === undefined || Number.isNaN(score)) return "N/A";
  return `${Math.round(score)}${suffix}`;
}

/** Turn snake_case / camelCase field keys into Title Case labels. */
export function humanize(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
}

export function formatDate(iso: string | undefined): string {
  if (!iso) return "Unknown";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso.slice(0, 10) : d.toLocaleDateString();
}

export function shortId(id: string | undefined): string {
  if (!id) return "unknown";
  return id.length > 8 ? `${id.slice(0, 8)}…` : id;
}
