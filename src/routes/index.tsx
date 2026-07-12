import { createFileRoute } from "@tanstack/react-router";
import { lazy, Suspense, useEffect, useState } from "react";

const App = lazy(() => import("../App"));

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  // Client-only mount: the ported Busulla app owns its own routing
  // (react-router-dom) and relies on browser-only APIs.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;
  return (
    <Suspense fallback={null}>
      <App />
    </Suspense>
  );
}
