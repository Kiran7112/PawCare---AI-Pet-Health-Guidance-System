import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { AssessmentResponse } from "@/types/assessment";

interface AssessmentContextValue {
  assessment: AssessmentResponse | null;
  setAssessment: (a: AssessmentResponse | null) => void;
  reset: () => void;
}

const AssessmentContext = createContext<AssessmentContextValue | null>(null);

/**
 * Holds the latest assessment so it survives client-side navigation (e.g. the
 * user visits "About" and returns). React Query owns the request lifecycle;
 * this owns the *selected* result.
 */
export function AssessmentProvider({ children }: { children: ReactNode }) {
  const [assessment, setAssessment] = useState<AssessmentResponse | null>(null);

  const value = useMemo<AssessmentContextValue>(
    () => ({
      assessment,
      setAssessment,
      reset: () => setAssessment(null),
    }),
    [assessment],
  );

  return (
    <AssessmentContext.Provider value={value}>
      {children}
    </AssessmentContext.Provider>
  );
}

export function useAssessmentStore(): AssessmentContextValue {
  const ctx = useContext(AssessmentContext);
  if (!ctx) {
    throw new Error("useAssessmentStore must be used within AssessmentProvider");
  }
  return ctx;
}
