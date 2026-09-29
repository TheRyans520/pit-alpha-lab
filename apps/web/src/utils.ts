import type { EquityPoint } from "./types";

export const MODEL_LABELS = {
  ridge: "Ridge",
  lightgbm: "LightGBM",
  mlp: "MLP",
} as const;

export function pct(value: number, digits = 1): string {
  return new Intl.NumberFormat("en-US", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function decimal(value: number, digits = 3): string {
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function compact(value: number): string {
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

export function strategyForModel(model: string, buffered = false): string {
  return `${model}_top_k${buffered ? "_buffered" : ""}`;
}

export function seriesMetrics(points: EquityPoint[]) {
  if (!points.length) {
    return { annualizedReturn: 0, sharpe: 0, drawdown: 0, turnover: 0, cumulative: 0 };
  }
  const returns = points.map((point) => point.net_return);
  const years = returns.length / 252;
  const finalWealth = points.at(-1)?.wealth ?? 1;
  const annualizedReturn = years > 0 && finalWealth > 0 ? finalWealth ** (1 / years) - 1 : -1;
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance =
    returns.length > 1
      ? returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1)
      : 0;
  const sharpe = variance > 0 ? (mean / Math.sqrt(variance)) * Math.sqrt(252) : 0;
  let peak = 1;
  let drawdown = 0;
  for (const point of points) {
    peak = Math.max(peak, point.wealth);
    drawdown = Math.min(drawdown, point.wealth / peak - 1);
  }
  return {
    annualizedReturn,
    sharpe,
    drawdown,
    turnover: points.reduce((sum, point) => sum + point.turnover, 0) / points.length,
    cumulative: finalWealth - 1,
  };
}
