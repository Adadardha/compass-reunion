import { createFileRoute } from "@tanstack/react-router";
import { lazy, Suspense, useEffect, useState } from "react";

const App = lazy(() => import("../App"));

export const Route = createFileRoute("/$")({
  component: CatchAll,
});

function CatchAll() {
  // The ported Busulla app uses react-router-dom + browser-only APIs
  // (localStorage, speech recognition), so it renders on the client only.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;
  return (
    <Suspense fallback={null}>
      <App />
    </Suspense>
  );
}
