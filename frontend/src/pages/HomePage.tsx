import { useAssessment, useHealthCheck } from "@/api/assessment";
import { useAssessmentStore } from "@/features/assessment/AssessmentContext";
import { AssessmentForm } from "@/features/assessment/AssessmentForm";
import { AssessmentProgress } from "@/features/assessment/AssessmentProgress";
import { ResultsView } from "@/features/assessment/ResultsView";
import { Alert } from "@/components/ui/Alert";
import type { AssessmentFormValues } from "@/features/assessment/schema";

export default function HomePage() {
  const { assessment, setAssessment, reset } = useAssessmentStore();
  const mutation = useAssessment();
  const health = useHealthCheck();

  const handleSubmit = (values: AssessmentFormValues) => {
    mutation.mutate(values, {
      onSuccess: (data) => setAssessment(data),
    });
  };

  const handleNew = () => {
    reset();
    mutation.reset();
  };

  if (mutation.isPending) {
    return (
      <div className="mx-auto max-w-2xl">
        <AssessmentProgress />
      </div>
    );
  }

  if (assessment) {
    return <ResultsView data={assessment} onNewAssessment={handleNew} />;
  }

  return (
    <div className="space-y-7">
      <section className="relative overflow-hidden rounded-3xl border border-slate-800 bg-hero-radial bg-slate-900/70 px-6 py-10 shadow-card backdrop-blur sm:px-10 sm:py-12">
        <span
          className="pointer-events-none absolute -right-10 -top-10 hidden text-[9rem] opacity-20 sm:block animate-float"
          aria-hidden
        >
          🐾
        </span>
        <div className="relative max-w-2xl">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-brand-500/30 bg-brand-500/15 px-3 py-1 text-xs font-semibold text-brand-300">
            <span className="h-1.5 w-1.5 rounded-full bg-brand-400" />
            ML + 11 AI agents
          </span>
          <h1 className="mt-4 text-3xl font-extrabold leading-tight tracking-tight text-white sm:text-4xl">
            AI-powered pet health &amp;{" "}
            <span className="gradient-text">care guidance</span>
          </h1>
          <p className="mt-3 text-base text-slate-300 sm:text-lg">
            Describe your pet in plain language. We assess health risk, evaluate
            your care capability with machine-learning models, and generate a
            personalized plan — emergency, preventive, or wellness.
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            {[
              { icon: "🔴", label: "Critical Care" },
              { icon: "🟡", label: "Preventive" },
              { icon: "🟢", label: "Wellness" },
            ].map((p) => (
              <span
                key={p.label}
                className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-800/80 px-3 py-1 text-sm font-medium text-slate-300 shadow-sm"
              >
                {p.icon} {p.label}
              </span>
            ))}
          </div>
        </div>
      </section>

      {health.isSuccess && !health.data.llm_configured && (
        <Alert variant="warning" title="LLM key not configured on the server">
          The backend is running but no OpenAI API key was found. The assessment
          will still run, but guidance sections may be empty. Set{" "}
          <code className="font-mono">OPENAI_API_KEY</code> in the server&apos;s{" "}
          <code className="font-mono">.env</code>.
        </Alert>
      )}

      <AssessmentForm
        onSubmit={handleSubmit}
        isSubmitting={mutation.isPending}
        serverError={mutation.isError ? mutation.error.message : null}
      />
    </div>
  );
}
