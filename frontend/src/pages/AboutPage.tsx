import { Card, CardBody, CardHeader } from "@/components/ui/Card";

const PATHS = [
  {
    icon: "🔴",
    title: "Critical Care",
    desc: "High-risk pets (risk > 0.6). Five agents cover risk analysis, emergency prep, critical nutrition, behavioral coaching and monitoring.",
  },
  {
    icon: "🟡",
    title: "Preventive Care",
    desc: "Moderate risk (0.3–0.6). Three agents cover preventive assessment, nutrition and wellness tracking.",
  },
  {
    icon: "🟢",
    title: "Wellness",
    desc: "Low risk (≤ 0.3). Three agents focus on optimization, nutrition enhancement and lifestyle enrichment.",
  },
];

const PIPELINE = [
  "Input validation",
  "LLM profile extraction (17 structured fields)",
  "ML scoring — health risk & owner care capability (parallel)",
  "Risk-based routing",
  "Path-specific guidance agents",
  "Output aggregation",
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-extrabold text-white">About PawCare+</h1>
        <p className="mt-2 text-slate-300">
          PawCare+ is a multi-agent system built on LangGraph. It combines two
          machine-learning models with eleven LLM agents to turn a plain-language
          description of your pet into a structured, personalized care plan.
        </p>
      </div>

      <Card>
        <CardHeader title="How the assessment works" icon="⚙️" />
        <CardBody>
          <ol className="space-y-2">
            {PIPELINE.map((step, i) => (
              <li key={step} className="flex gap-3 text-sm text-slate-300">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-brand-500/20 text-xs font-semibold text-brand-300">
                  {i + 1}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </CardBody>
      </Card>

      <div className="grid gap-4 sm:grid-cols-3">
        {PATHS.map((p) => (
          <Card key={p.title} interactive>
            <CardBody>
              <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-800 text-2xl ring-1 ring-slate-700">
                {p.icon}
              </div>
              <h3 className="mt-3 font-semibold text-white">{p.title}</h3>
              <p className="mt-1 text-sm text-slate-300">{p.desc}</p>
            </CardBody>
          </Card>
        ))}
      </div>

      <p className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
        ⚠️ PawCare+ provides educational guidance only and is not a substitute for
        professional veterinary diagnosis or treatment.
      </p>
    </div>
  );
}
