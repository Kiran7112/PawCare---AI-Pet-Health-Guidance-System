import { z } from "zod";

const field = (label: string) =>
  z
    .string()
    .trim()
    .min(10, `Please add a bit more detail to "${label}" (at least 10 characters).`)
    .max(5000, `"${label}" is too long (max 5000 characters).`);

export const assessmentSchema = z.object({
  about_pet: field("About your pet"),
  daily_routine: field("Daily routine & living situation"),
  health_concerns: field("Current health concerns"),
});

export type AssessmentFormValues = z.infer<typeof assessmentSchema>;

export const EMPTY_FORM: AssessmentFormValues = {
  about_pet: "",
  daily_routine: "",
  health_concerns: "",
};
