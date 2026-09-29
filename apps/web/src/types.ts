export type ModelName = "ridge" | "lightgbm" | "mlp";

export interface RunItem {
  run_id: string;
  created_at_utc: string;
  completed_at_utc: string | null;
  snapshot_id: string;
  data_source: string;
  model: ModelName;
  primary_cost_bps: number;
  quality_status: string;
  content_id: string;
}

export interface ModelComparison {
  model: ModelName;
  ic: number;
  rank_ic: number;
  ic_ci_low: number;
  ic_ci_high: number;
  rank_ic_ci_low: number;
  rank_ic_ci_high: number;
  annualized_return: number;
  sharpe: number;
  annualized_volatility: number;
  maximum_drawdown: number;
  mean_turnover: number;
  breakeven_one_way_cost_bps: number;
  preferred: boolean;
}

export interface PortfolioMetric {
  strategy: string;
  window: string;
  cost_bps: number;
  annualized_return: number;
  annualized_volatility: number;
  cumulative_return: number;
  maximum_drawdown: number;
  mean_turnover: number;
  sharpe: number;
  historical_var_95_loss: number;
  historical_cvar_95_loss: number;
  mean_effective_positions: number;
  days: number;
}

export interface RunSummary {
  run_id: string;
  snapshot_id: string;
  model: ModelName;
  primary_cost_bps: number;
  artifacts_verified: boolean;
  data: {
    rows: number;
    dates: number;
    instruments: number;
    daily_members_min: number;
    daily_members_max: number;
  };
  data_quality: {
    status: string;
    checks: number;
    passed: number;
    warnings: number;
    failed_errors: number;
    label_coverage: number;
    realized_return_coverage: number;
    minimum_feature_coverage: number;
  };
  prediction: {
    days: number;
    ic: number;
    ic_std: number;
    icir: number;
    rank_ic: number;
    rank_ic_std: number;
    rank_icir: number;
  } | null;
  bootstrap_uncertainty: Array<{
    metric: string;
    point_estimate: number;
    ci_low: number;
    ci_high: number;
    confidence: number;
  }>;
  signal_decay: Array<{
    horizon_days: number;
    mean_rank_ic: number;
    rank_ic_std: number;
    days: number;
  }>;
  portfolio: PortfolioMetric[];
  cost_break_even: Array<{
    strategy: string;
    breakeven_one_way_cost_bps: number;
  }>;
}

export interface EquityPoint {
  date: string;
  net_return: number;
  wealth: number;
  turnover: number;
}

export interface EquityResponse {
  run_id: string;
  strategy: string;
  cost_bps: number;
  points: EquityPoint[];
}

export interface MarketContextEvidence {
  multiseed: {
    seed_count: number;
    seeds: number[];
    decision: string;
    fractions: {
      positive_rank_ic: number;
      positive_net_annualized_return: number;
      positive_buffered_net_annualized_return: number;
    };
    metrics: Record<string, {
      mean: number;
      sample_std: number;
      minimum: number;
      maximum: number;
    }>;
  };
  seeds: Array<{
    model_seed: number;
    rank_ic: number;
    annualized_return_10bps: number;
    buffered_annualized_return_10bps: number;
    artifacts_verified: boolean;
  }>;
  ablation: {
    decision: string;
    observations: {
      ranking_effect_on_rank_ic_without_context: number;
      context_effect_on_rank_ic_with_ranking: number;
      ridge_minus_best_neural_net_annualized_return: number;
    };
  };
  ablations: Array<{
    variant: string;
    rank_ic: number;
    annualized_return_10bps: number;
  }>;
}
