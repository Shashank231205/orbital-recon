import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import "leaflet/dist/leaflet.css";
import "@/styles.css";
import { App } from "@/App";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Detection results are immutable once a job completes, so refetching on
      // window focus would only add traffic.
      refetchOnWindowFocus: false,
      staleTime: 60_000,
      retry: 1,
    },
  },
});

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element is missing from the document");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
