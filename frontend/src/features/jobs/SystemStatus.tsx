/** Backend health, so the operator can see what is available before uploading. */

import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { StatusDot } from "@/components/Indicators";

export function SystemStatus() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: 1,
  });

  if (isError || !data) {
    return (
      <span className="row mono dim">
        <StatusDot state={isError ? "bad" : "idle"} />
        {isError ? "backend unreachable" : "connecting"}
      </span>
    );
  }

  const usable = data.llm_providers.filter((provider) => provider.configured);
  const healthy = usable.filter((provider) => provider.healthy);

  return (
    <span className="row mono dim" style={{ gap: 14 }}>
      <span className="row">
        <StatusDot state="ok" />
        {data.device}
      </span>
      <span className="row">
        <StatusDot
          state={usable.length === 0 ? "idle" : healthy.length > 0 ? "ok" : "bad"}
        />
        {usable.length === 0
          ? "no model keys"
          : `${healthy.length}/${usable.length} providers`}
      </span>
      <span>v{data.version}</span>
    </span>
  );
}
