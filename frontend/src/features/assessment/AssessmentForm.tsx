import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { Search, Eraser, Sparkles } from "lucide-react";
import { TextArea } from "@/components/ui/TextArea";
import { Button } from "@/components/ui/Button";
import { Card, CardBody } from "@/components/ui/Card";
import { Alert } from "@/components/ui/Alert";
import {
  assessmentSchema,
  EMPTY_FORM,
  type AssessmentFormValues,
} from "./schema";

interface AssessmentFormProps {
  onSubmit: (values: AssessmentFormValues) => void;
  isSubmitting: boolean;
  serverError?: string | null;
}

const SAMPLE: AssessmentFormValues = {
  about_pet:
    "Max is a 7-year-old neutered male Labrador Retriever, slightly overweight. No major past illnesses, up to date on vaccines.",
  daily_routine:
    "Lives indoors with a fenced yard. Two 20-minute walks daily, eats premium kibble twice a day. Experienced owner, vet within 10 minutes.",
  health_concerns:
    "Drinking noticeably more water for the last 10 days, occasional lethargy and reduced appetite in the evenings.",
};

export function AssessmentForm({
  onSubmit,
  isSubmitting,
  serverError,
}: AssessmentFormProps) {
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<AssessmentFormValues>({
    resolver: zodResolver(assessmentSchema),
    defaultValues: EMPTY_FORM,
    mode: "onBlur",
  });

  return (
    <Card className="overflow-hidden">
      <div className="h-1.5 w-full bg-brand-gradient" />
      <CardBody className="space-y-5 p-6 sm:p-7">
        <div className="flex items-start gap-3">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-brand-500/15 text-2xl ring-1 ring-brand-500/25">
            📝
          </span>
          <div>
            <h2 className="text-lg font-bold text-white">
              Pet Health Assessment
            </h2>
            <p className="mt-0.5 text-sm text-slate-400">
              Tell us about your pet in plain language. Our AI agents analyze the
              health risk, your care capability, and generate a personalized plan.
            </p>
          </div>
        </div>

        {serverError && (
          <Alert variant="error" title="Assessment failed">
            {serverError}
          </Alert>
        )}

        <form
          onSubmit={handleSubmit(onSubmit)}
          className="space-y-5"
          noValidate
        >
          <div className="grid gap-5 md:grid-cols-2">
            <TextArea
              label="🐕 About your pet"
              hint="Species, breed, age, sex, weight, known conditions, personality."
              placeholder="e.g. Max is a 7-year-old male Labrador, slightly overweight…"
              error={errors.about_pet?.message}
              {...register("about_pet")}
            />
            <TextArea
              label="⏰ Daily routine & living situation"
              hint="Exercise, diet, environment, owner experience, vet access."
              placeholder="e.g. Two walks a day, premium kibble, lives indoors…"
              error={errors.daily_routine?.message}
              {...register("daily_routine")}
            />
          </div>

          <TextArea
            label="🏥 Current health concerns"
            hint="Symptoms, behavioral issues, medications, recent vet visits."
            placeholder="e.g. Increased thirst and lethargy over the past week…"
            error={errors.health_concerns?.message}
            {...register("health_concerns")}
          />

          <div className="flex flex-col gap-3 border-t border-slate-800 pt-4 sm:flex-row sm:items-center">
            <Button
              type="submit"
              size="lg"
              loading={isSubmitting}
              className="sm:flex-1"
            >
              <Search className="h-4 w-4" />
              {isSubmitting ? "Analyzing…" : "Generate assessment"}
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="lg"
              disabled={isSubmitting}
              onClick={() => reset(EMPTY_FORM)}
            >
              <Eraser className="h-4 w-4" />
              Clear
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="lg"
              disabled={isSubmitting}
              onClick={() => reset(SAMPLE)}
            >
              <Sparkles className="h-4 w-4" />
              Use sample
            </Button>
          </div>
        </form>
      </CardBody>
    </Card>
  );
}
