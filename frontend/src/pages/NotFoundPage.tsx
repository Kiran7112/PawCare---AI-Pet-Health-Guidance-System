import { Link } from "react-router-dom";
import { Button } from "@/components/ui/Button";

export default function NotFoundPage() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-24 text-center">
      <div className="text-5xl">🐾</div>
      <h1 className="text-2xl font-bold text-white">Page not found</h1>
      <p className="max-w-sm text-slate-400">
        The page you&apos;re looking for wandered off. Let&apos;s get you back to
        the assessment.
      </p>
      <Link to="/">
        <Button>Back to assessment</Button>
      </Link>
    </div>
  );
}
