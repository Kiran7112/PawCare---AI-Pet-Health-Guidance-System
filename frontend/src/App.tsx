import { lazy, Suspense } from "react";
import {
  createBrowserRouter,
  RouterProvider,
  type RouteObject,
} from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";
import { AssessmentProvider } from "@/features/assessment/AssessmentContext";
import { LoadingState } from "@/components/ui/Spinner";

// Code-split routes: the home page pulls in the form + results bundle lazily.
const HomePage = lazy(() => import("@/pages/HomePage"));
const AboutPage = lazy(() => import("@/pages/AboutPage"));
const NotFoundPage = lazy(() => import("@/pages/NotFoundPage"));

const routes: RouteObject[] = [
  {
    element: (
      <AssessmentProvider>
        <AppLayout />
      </AssessmentProvider>
    ),
    children: [
      { index: true, element: <HomePage /> },
      { path: "about", element: <AboutPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
];

const router = createBrowserRouter(routes);

export default function App() {
  return (
    <Suspense fallback={<LoadingState title="Loading PawCare+…" />}>
      <RouterProvider router={router} />
    </Suspense>
  );
}
