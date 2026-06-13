import { Component, type ErrorInfo, type ReactNode } from "react";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";

interface Props {
  children: ReactNode;
}
interface State {
  error: Error | null;
}

/** Catches render-time crashes so a single bad payload can't blank the app. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Render error caught by ErrorBoundary:", error, info);
  }

  handleReset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-2xl px-4 py-16">
          <Alert variant="error" title="Something went wrong">
            <p className="mb-3">
              The page hit an unexpected error while rendering. You can try again.
            </p>
            <p className="mb-3 font-mono text-xs opacity-80">
              {this.state.error.message}
            </p>
            <Button variant="secondary" size="sm" onClick={this.handleReset}>
              Try again
            </Button>
          </Alert>
        </div>
      );
    }
    return this.props.children;
  }
}
