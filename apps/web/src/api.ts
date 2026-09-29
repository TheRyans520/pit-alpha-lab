import type { EquityResponse, MarketContextEvidence, ModelComparison, RunItem, RunSummary } from "./types";

const API_ROOT = (import.meta.env.VITE_API_ROOT ?? "").replace(/\/$/, "");

async function readJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, { signal });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${detail || response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export async function getRuns(signal?: AbortSignal): Promise<RunItem[]> {
  const payload = await readJson<{ runs: RunItem[] }>("/api/runs", signal);
  return payload.runs;
}

export async function getComparison(signal?: AbortSignal): Promise<ModelComparison[]> {
  const payload = await readJson<{ cost_bps: number; models: ModelComparison[] }>(
    "/api/case-study/model-comparison",
    signal,
  );
  return payload.models;
}

export function getSummary(runId: string, signal?: AbortSignal): Promise<RunSummary> {
  return readJson(`/api/runs/${encodeURIComponent(runId)}/summary`, signal);
}

export function getEquity(
  runId: string,
  strategy: string,
  costBps: number,
  signal?: AbortSignal,
): Promise<EquityResponse> {
  const params = new URLSearchParams({ strategy, cost_bps: String(costBps) });
  return readJson(`/api/runs/${encodeURIComponent(runId)}/equity?${params}`, signal);
}

export function getMarketContextEvidence(signal?: AbortSignal): Promise<MarketContextEvidence> {
  return readJson("/api/case-study/market-context", signal);
}
