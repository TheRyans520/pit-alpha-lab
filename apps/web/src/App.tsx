import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { motion } from "motion/react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Database,
  ExternalLink,
  Fingerprint,
  Gauge,
  GitCompareArrows,
  BrainCircuit,
  FlaskConical,
  Layers3,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { getComparison, getEquity, getMarketContextEvidence, getRuns, getSummary } from "./api";
import type { EquityPoint, MarketContextEvidence, ModelComparison, ModelName, RunItem, RunSummary } from "./types";
import { compact, decimal, MODEL_LABELS, pct, seriesMetrics, strategyForModel } from "./utils";

const EquityChart = lazy(() => import("./components/EquityChart"));

const COSTS = [0, 5, 10, 20];
const SNAPSHOT_BENCHMARK_RETURN = 0.016267815;
const MODELS: ModelName[] = ["ridge", "lightgbm", "mlp"];
const MODEL_COLORS: Record<ModelName, string> = {
  ridge: "#d45132",
  lightgbm: "#277c70",
  mlp: "#7562ab",
};

const SNAPSHOT_COMPARISON: ModelComparison[] = [
  {
    model: "ridge",
    ic: 0.0278483642,
    rank_ic: 0.0253494428,
    ic_ci_low: 0.0148321848,
    ic_ci_high: 0.0403181374,
    rank_ic_ci_low: 0.0122054997,
    rank_ic_ci_high: 0.0381743035,
    annualized_return: 0.0848564284,
    sharpe: 0.4450670969,
    annualized_volatility: 0.2545957993,
    maximum_drawdown: -0.2560732289,
    mean_turnover: 0.3062631923,
    breakeven_one_way_cost_bps: 20.5166756491,
    preferred: true,
  },
  {
    model: "lightgbm",
    ic: 0.0240944591,
    rank_ic: 0.0248468377,
    ic_ci_low: 0.0123185921,
    ic_ci_high: 0.0359600231,
    rank_ic_ci_low: 0.0137842087,
    rank_ic_ci_high: 0.0370489018,
    annualized_return: 0.0535688606,
    sharpe: 0.3234731953,
    annualized_volatility: 0.282399849,
    maximum_drawdown: -0.4373133548,
    mean_turnover: 0.3069640621,
    breakeven_one_way_cost_bps: 16.718552633,
    preferred: false,
  },
  {
    model: "mlp",
    ic: 0.0070131578,
    rank_ic: 0.0074900349,
    ic_ci_low: -0.0070336745,
    ic_ci_high: 0.0215685019,
    rank_ic_ci_low: -0.009514717,
    rank_ic_ci_high: 0.0238331837,
    annualized_return: -0.1133410932,
    sharpe: -0.3625223844,
    annualized_volatility: 0.2483830416,
    maximum_drawdown: -0.6054936134,
    mean_turnover: 0.3465665971,
    breakeven_one_way_cost_bps: 0,
    preferred: false,
  },
];

const SNAPSHOT_CONTEXT: MarketContextEvidence = {
  multiseed: {
    seed_count: 5,
    seeds: [7, 17, 29, 41, 53],
    decision: "reject_for_promotion",
    fractions: {
      positive_rank_ic: 1,
      positive_net_annualized_return: 0,
      positive_buffered_net_annualized_return: 0.2,
    },
    metrics: {
      rank_ic: { mean: 0.0145686, sample_std: 0.0097837, minimum: 0.0055407, maximum: 0.026306 },
      annualized_return_10bps: { mean: -0.0517919, sample_std: 0.0203007, minimum: -0.0700954, maximum: -0.0209621 },
      buffered_annualized_return_10bps: { mean: -0.0385076, sample_std: 0.035091, minimum: -0.0692876, maximum: 0.0031327 },
    },
  },
  seeds: [
    { model_seed: 7, rank_ic: 0.026306, annualized_return_10bps: -0.0599, buffered_annualized_return_10bps: -0.0684, artifacts_verified: true },
    { model_seed: 17, rank_ic: 0.008707, annualized_return_10bps: -0.0701, buffered_annualized_return_10bps: -0.0531, artifacts_verified: true },
    { model_seed: 29, rank_ic: 0.005541, annualized_return_10bps: -0.021, buffered_annualized_return_10bps: 0.0031, artifacts_verified: true },
    { model_seed: 41, rank_ic: 0.00826, annualized_return_10bps: -0.066, buffered_annualized_return_10bps: -0.0693, artifacts_verified: true },
    { model_seed: 53, rank_ic: 0.02403, annualized_return_10bps: -0.042, buffered_annualized_return_10bps: -0.0048, artifacts_verified: true },
  ],
  ablation: {
    decision: "retain_ridge_reject_current_context_architecture",
    observations: {
      ranking_effect_on_rank_ic_without_context: 0.0165498,
      context_effect_on_rank_ic_with_ranking: -0.0153332,
      ridge_minus_best_neural_net_annualized_return: 0.1049341,
    },
  },
  ablations: [],
};

type Status = "connecting" | "live" | "snapshot";

function MetricCard({
  label,
  value,
  note,
  tone = "neutral",
}: {
  label: string;
  value: string;
  note: string;
  tone?: "neutral" | "positive" | "negative";
}) {
  return (
    <motion.article
      className={`metric-card ${tone}`}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </motion.article>
  );
}

function LoadingPanel() {
  return (
    <div className="loading-panel">
      <RefreshCw size={18} className="spin" />
      <span>Reading audited artifacts…</span>
    </div>
  );
}

function ModelComparisonPanel({ models }: { models: ModelComparison[] }) {
  const maxReturn = Math.max(...models.map((model) => Math.abs(model.annualized_return)), 0.12);
  return (
    <section className="panel comparison-panel" id="models">
      <div className="panel-header">
        <div>
          <div className="eyebrow">Out-of-sample · 10 bps</div>
          <h2>Complexity did not win</h2>
        </div>
        <div className="audit-pill">
          <ShieldCheck size={15} /> Frozen protocol
        </div>
      </div>
      <p className="panel-intro">
        All three models see the same point-in-time universe, features, folds and execution assumptions.
        The simple linear baseline remains the honest winner.
      </p>
      <div className="model-rows">
        {models.map((model, index) => {
          const width = `${Math.max(4, (Math.abs(model.annualized_return) / maxReturn) * 100)}%`;
          return (
            <motion.div
              className="model-row"
              key={model.model}
              initial={{ opacity: 0, x: -14 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.08 * index }}
            >
              <div className="model-row-head">
                <div className="model-name-wrap">
                  <span className="model-dot" style={{ background: MODEL_COLORS[model.model] }} />
                  <strong>{MODEL_LABELS[model.model]}</strong>
                  {model.preferred && <span className="preferred-tag">preferred</span>}
                </div>
                <span className={model.annualized_return >= 0 ? "positive-text" : "negative-text"}>
                  {pct(model.annualized_return)} p.a.
                </span>
              </div>
              <div className="return-track">
                <motion.div
                  className={`return-fill ${model.annualized_return < 0 ? "negative" : ""}`}
                  style={{ background: MODEL_COLORS[model.model] }}
                  initial={{ width: 0 }}
                  animate={{ width }}
                  transition={{ duration: 0.7, delay: 0.12 + index * 0.08 }}
                />
              </div>
              <div className="model-stats">
                <span>Rank IC <b>{decimal(model.rank_ic, 3)}</b></span>
                <span>Sharpe <b>{decimal(model.sharpe, 2)}</b></span>
                <span>Max DD <b>{pct(model.maximum_drawdown)}</b></span>
              </div>
            </motion.div>
          );
        })}
      </div>
      <div className="verdict">
        <div className="verdict-icon"><Sparkles size={19} /></div>
        <div>
          <strong>Research verdict</strong>
          <p>Ridge is retained. MLP’s confidence interval crosses zero, so its negative result stays visible.</p>
        </div>
      </div>
    </section>
  );
}

function App() {
  const [status, setStatus] = useState<Status>("connecting");
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [comparison, setComparison] = useState<ModelComparison[]>(SNAPSHOT_COMPARISON);
  const [contextEvidence, setContextEvidence] = useState<MarketContextEvidence>(SNAPSHOT_CONTEXT);
  const [selectedModel, setSelectedModel] = useState<ModelName>("ridge");
  const [costBps, setCostBps] = useState(10);
  const [buffered, setBuffered] = useState(false);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [strategyPoints, setStrategyPoints] = useState<EquityPoint[]>([]);
  const [benchmarkPoints, setBenchmarkPoints] = useState<EquityPoint[]>([]);
  const [loadingSeries, setLoadingSeries] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      getRuns(controller.signal),
      getComparison(controller.signal),
      getMarketContextEvidence(controller.signal),
    ])
      .then(([nextRuns, nextComparison, nextContextEvidence]) => {
        setRuns(nextRuns);
        setComparison(nextComparison);
        setContextEvidence(nextContextEvidence);
        setStatus("live");
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setStatus("snapshot");
        }
      });
    return () => controller.abort();
  }, []);

  const latestRuns = useMemo(() => {
    const byModel = new Map<ModelName, RunItem>();
    for (const run of runs) {
      if (
        run.data_source === "qlib" &&
        run.quality_status === "pass" &&
        MODELS.includes(run.model) &&
        !byModel.has(run.model)
      ) {
        byModel.set(run.model, run);
      }
    }
    return byModel;
  }, [runs]);

  const selectedRun = latestRuns.get(selectedModel);

  useEffect(() => {
    if (!selectedRun) {
      setSummary(null);
      setStrategyPoints([]);
      setBenchmarkPoints([]);
      return;
    }
    const controller = new AbortController();
    setLoadingSeries(true);
    const strategy = strategyForModel(selectedModel, buffered);
    Promise.all([
      getSummary(selectedRun.run_id, controller.signal),
      getEquity(selectedRun.run_id, strategy, costBps, controller.signal),
      getEquity(selectedRun.run_id, "eligible_equal_weight", costBps, controller.signal),
    ])
      .then(([nextSummary, strategyEquity, benchmarkEquity]) => {
        setSummary(nextSummary);
        setStrategyPoints(strategyEquity.points);
        setBenchmarkPoints(benchmarkEquity.points);
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setStrategyPoints([]);
          setBenchmarkPoints([]);
        }
      })
      .finally(() => setLoadingSeries(false));
    return () => controller.abort();
  }, [selectedRun, selectedModel, costBps, buffered]);

  const selectedComparison = comparison.find((row) => row.model === selectedModel);
  const liveMetrics = useMemo(() => seriesMetrics(strategyPoints), [strategyPoints]);
  const benchmarkMetrics = useMemo(() => seriesMetrics(benchmarkPoints), [benchmarkPoints]);
  const displayMetrics = strategyPoints.length
    ? liveMetrics
    : {
        annualizedReturn: selectedComparison?.annualized_return ?? 0,
        sharpe: selectedComparison?.sharpe ?? 0,
        drawdown: selectedComparison?.maximum_drawdown ?? 0,
        turnover: selectedComparison?.mean_turnover ?? 0,
        cumulative: 0,
      };
  const benchmarkAnnualReturn = benchmarkPoints.length
    ? benchmarkMetrics.annualizedReturn
    : SNAPSHOT_BENCHMARK_RETURN;

  const chartData = useMemo(() => {
    const benchmarkByDate = new Map(benchmarkPoints.map((point) => [point.date, point.wealth]));
    return strategyPoints.map((point) => ({
      date: point.date,
      strategy: point.wealth,
      benchmark: benchmarkByDate.get(point.date),
    }));
  }, [strategyPoints, benchmarkPoints]);

  const breakeven = selectedComparison?.breakeven_one_way_cost_bps ?? 0;
  const interval = summary?.bootstrap_uncertainty.find((row) => row.metric === "rank_ic");

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="PIT Alpha Lab home">
          <span className="brand-mark"><span /></span>
          <span>PIT / ALPHA</span>
        </a>
        <nav className="main-nav" aria-label="Primary navigation">
          <a href="#performance">Performance</a>
          <a href="#models">Models</a>
          <a href="#evidence">Evidence</a>
        </nav>
        <div className={`connection-status ${status}`}>
          <span className="status-dot" />
          {status === "live" ? "Live artifacts" : status === "snapshot" ? "Audited snapshot" : "Connecting"}
        </div>
      </header>

      <main id="top">
        <section className="hero">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65 }}
          >
            <div className="hero-kicker"><CircleDot size={14} /> CSI 300 · 2021–2025 OOS</div>
            <h1>Research that<br /><em>survives hindsight.</em></h1>
            <p>
              Point-in-time equity signals, tested under frozen universes, realistic turnover and
              explicit transaction costs. Every number traces back to a checksum-audited run.
            </p>
            <div className="hero-actions">
              <a className="primary-button" href="#performance">Explore results <ArrowDownRight size={17} /></a>
              <a className="text-link" href="#evidence">View methodology <ExternalLink size={15} /></a>
            </div>
          </motion.div>
          <motion.div
            className="hero-proof"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.15, duration: 0.55 }}
          >
            <div className="proof-topline">
              <span>Selected finding</span>
              <span className="proof-index">01 / 03</span>
            </div>
            <div className="proof-number">+{pct(SNAPSHOT_COMPARISON[0].annualized_return)}</div>
            <div className="proof-label">Ridge annualized return, net of 10 bps one-way costs</div>
            <div className="proof-rule" />
            <div className="proof-grid">
              <div><span>Rank IC</span><strong>{decimal(SNAPSHOT_COMPARISON[0].rank_ic, 3)}</strong></div>
              <div><span>Max drawdown</span><strong>{pct(SNAPSHOT_COMPARISON[0].maximum_drawdown)}</strong></div>
              <div><span>Break-even</span><strong>{decimal(SNAPSHOT_COMPARISON[0].breakeven_one_way_cost_bps, 1)} bps</strong></div>
            </div>
            <div className="proof-caption"><Check size={14} /> 14 / 14 semantic data checks passed</div>
          </motion.div>
        </section>

        <section className="control-strip" aria-label="Experiment controls">
          <div className="model-switcher">
            <span className="control-label">Model</span>
            <div className="segmented">
              {MODELS.map((model) => (
                <button
                  type="button"
                  key={model}
                  className={selectedModel === model ? "active" : ""}
                  onClick={() => setSelectedModel(model)}
                >
                  {MODEL_LABELS[model]}
                </button>
              ))}
            </div>
          </div>
          <div className="cost-switcher">
            <span className="control-label">One-way cost</span>
            <div className="cost-buttons">
              {COSTS.map((cost) => (
                <button
                  type="button"
                  key={cost}
                  className={costBps === cost ? "active" : ""}
                  onClick={() => setCostBps(cost)}
                  disabled={status !== "live"}
                  title={status === "live" ? `${cost} bps one-way cost` : "Start the API to change cost assumptions"}
                >
                  {cost} <small>bps</small>
                </button>
              ))}
            </div>
          </div>
          <label className="buffer-toggle">
            <input
              type="checkbox"
              checked={buffered}
              disabled={status !== "live"}
              onChange={(event) => setBuffered(event.target.checked)}
            />
            <span className="toggle-track"><span /></span>
            <span>Turnover buffer</span>
          </label>
          <div className="run-id">
            <Fingerprint size={15} />
            <span>{selectedRun?.content_id.slice(0, 12) ?? "snapshot-v1"}</span>
            <ChevronDown size={14} />
          </div>
        </section>

        <section className="metrics-grid" id="performance">
          <MetricCard
            label="Annualized return"
            value={pct(displayMetrics.annualizedReturn)}
            note={`${displayMetrics.annualizedReturn - benchmarkAnnualReturn >= 0 ? "+" : ""}${pct(displayMetrics.annualizedReturn - benchmarkAnnualReturn)} vs equal-weight`}
            tone={displayMetrics.annualizedReturn >= 0 ? "positive" : "negative"}
          />
          <MetricCard
            label="Sharpe ratio"
            value={decimal(displayMetrics.sharpe, 2)}
            note={status === "live" ? `${costBps} bps · ${buffered ? "buffered" : "standard"} rebalance` : "Audited snapshot · 10 bps"}
            tone={displayMetrics.sharpe >= 0 ? "positive" : "negative"}
          />
          <MetricCard
            label="Maximum drawdown"
            value={pct(displayMetrics.drawdown)}
            note="Peak-to-trough, net returns"
          />
          <MetricCard
            label="Break-even cost"
            value={`${decimal(breakeven, 1)} bps`}
            note="One-way cost to zero terminal return"
            tone={breakeven > 10 ? "positive" : "negative"}
          />
        </section>

        <section className="main-grid">
          <section className="panel chart-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Net performance</div>
                <h2>Out-of-sample wealth</h2>
              </div>
              <div className="chart-legend">
                <span><i style={{ background: MODEL_COLORS[selectedModel] }} />{MODEL_LABELS[selectedModel]}</span>
                <span><i className="benchmark-line" />Equal weight</span>
              </div>
            </div>
            <div className="chart-summary">
              <strong>{strategyPoints.length ? `${decimal(1 + displayMetrics.cumulative, 2)}×` : "—"}</strong>
              <span>Terminal wealth</span>
              {strategyPoints.length > 0 && (
                <span className={displayMetrics.cumulative >= 0 ? "delta positive-text" : "delta negative-text"}>
                  {displayMetrics.cumulative >= 0 ? <ArrowUpRight size={15} /> : <ArrowDownRight size={15} />}
                  {pct(displayMetrics.cumulative)} total
                </span>
              )}
            </div>
            <div className="chart-wrap">
              {loadingSeries ? (
                <LoadingPanel />
              ) : chartData.length ? (
                <Suspense fallback={<LoadingPanel />}>
                  <EquityChart
                    data={chartData}
                    color={MODEL_COLORS[selectedModel]}
                    label={MODEL_LABELS[selectedModel]}
                  />
                </Suspense>
              ) : (
                <div className="empty-chart">
                  <Activity size={26} />
                  <strong>Start the local API to explore the full curve</strong>
                  <span>The audited summary remains available in snapshot mode.</span>
                  <code>python -m pitalpha serve</code>
                </div>
              )}
            </div>
            <div className="chart-footnote">
              <span>Weekly Top-30 rebalance</span>
              <span>{compact(strategyPoints.length || 1212)} trading days</span>
              <span>Costs charged on one-way turnover</span>
            </div>
          </section>

          <aside className="panel signal-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Signal evidence</div>
                <h2>Weak, but measurable</h2>
              </div>
              <Gauge size={20} />
            </div>
            <div className="signal-ring-wrap">
              <div className="signal-ring" style={{ "--progress": `${Math.min(100, Math.abs((summary?.prediction?.rank_ic ?? selectedComparison?.rank_ic ?? 0) * 2000))}%` } as React.CSSProperties}>
                <div><strong>{decimal(summary?.prediction?.rank_ic ?? selectedComparison?.rank_ic ?? 0, 3)}</strong><span>Rank IC</span></div>
              </div>
            </div>
            <div className="confidence-block">
              <div className="confidence-title"><span>95% confidence interval</span><b>{interval ? `${decimal(interval.ci_low, 3)} — ${decimal(interval.ci_high, 3)}` : "snapshot"}</b></div>
              <div className="confidence-track">
                <span className="zero-mark" />
                <span
                  className="confidence-range"
                  style={{
                    left: `${Math.max(0, Math.min(100, ((interval?.ci_low ?? selectedComparison?.rank_ic_ci_low ?? 0) + 0.02) / 0.08 * 100))}%`,
                    right: `${Math.max(0, Math.min(100, 100 - (((interval?.ci_high ?? selectedComparison?.rank_ic_ci_high ?? 0) + 0.02) / 0.08 * 100)))}%`,
                    background: MODEL_COLORS[selectedModel],
                  }}
                />
              </div>
              <div className="confidence-scale"><span>−.02</span><span>0</span><span>.06</span></div>
            </div>
            <div className="signal-decay">
              <span>Signal decay</span>
              {(summary?.signal_decay ?? []).map((item) => (
                <div key={item.horizon_days}>
                  <small>{item.horizon_days}D</small>
                  <span><i style={{ width: `${Math.min(100, item.mean_rank_ic * 3000)}%`, background: MODEL_COLORS[selectedModel] }} /></span>
                  <b>{decimal(item.mean_rank_ic, 3)}</b>
                </div>
              ))}
              {!summary && <p>Live horizon diagnostics load with the local API.</p>}
            </div>
          </aside>
        </section>

        <section className="lower-grid">
          <ModelComparisonPanel models={comparison} />

          <section className="panel evidence-panel" id="evidence">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Data integrity</div>
                <h2>Evidence before alpha</h2>
              </div>
              <Database size={20} />
            </div>
            <div className="quality-score">
              <div className="quality-number">{summary?.data_quality.passed ?? 14}<span>/{summary?.data_quality.checks ?? 14}</span></div>
              <div><strong>Checks passed</strong><p>No blocking errors or silent repairs.</p></div>
              <CheckCircle2 size={25} />
            </div>
            <div className="evidence-list">
              <div><span><Layers3 size={16} /> Observations</span><strong>{compact(summary?.data.rows ?? 1_312_314)}</strong></div>
              <div><span><GitCompareArrows size={16} /> PIT members / day</span><strong>{summary ? `${summary.data.daily_members_min}–${summary.data.daily_members_max}` : "298–300"}</strong></div>
              <div><span><Activity size={16} /> Feature coverage</span><strong>{pct(summary?.data_quality.minimum_feature_coverage ?? 0.9607548)}</strong></div>
              <div><span><ShieldCheck size={16} /> Artifact hashes</span><strong>{summary?.artifacts_verified === false ? "Mismatch" : "Verified"}</strong></div>
            </div>
            <div className="provenance-note">
              <span>Snapshot</span>
              <code>{summary?.snapshot_id ?? "qlib-cn-2026-09-09-common-2025"}</code>
            </div>
          </section>
        </section>

        <section className="panel advanced-panel">
          <div className="panel-header advanced-header">
            <div>
              <div className="eyebrow">Neural stability audit · 5 seeds</div>
              <h2>Signal is not the same as a strategy</h2>
            </div>
            <div className="rejected-pill"><FlaskConical size={15} /> Not promoted</div>
          </div>
          <p className="panel-intro">
            The market-context network finds positive rank correlation in every seed, but none of
            the unbuffered portfolios survive 10 bps costs. Negative evidence stays first-class.
          </p>
          <div className="advanced-layout">
            <div className="seed-audit">
              <div className="advanced-stat-row">
                <div><span>Mean Rank IC</span><strong>{decimal(contextEvidence.multiseed.metrics.rank_ic.mean, 4)}</strong></div>
                <div><span>Mean net return</span><strong className="negative-text">{pct(contextEvidence.multiseed.metrics.annualized_return_10bps.mean)}</strong></div>
                <div><span>Positive net seeds</span><strong>0 / {contextEvidence.multiseed.seed_count}</strong></div>
              </div>
              <div className="seed-lines">
                {contextEvidence.seeds.map((seed) => (
                  <div className="seed-line" key={seed.model_seed}>
                    <span>Seed {seed.model_seed}</span>
                    <div className="seed-track">
                      <motion.i
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.max(4, Math.abs(seed.annualized_return_10bps) / 0.08 * 100)}%` }}
                        transition={{ duration: 0.55 }}
                      />
                    </div>
                    <b>{pct(seed.annualized_return_10bps)}</b>
                    <small>RIC {decimal(seed.rank_ic, 3)}</small>
                  </div>
                ))}
              </div>
            </div>
            <div className="ablation-card">
              <div className="ablation-icon"><BrainCircuit size={22} /></div>
              <div className="eyebrow">What the ablation says</div>
              <h3>Ranking helps the statistic.<br />Context does not—yet.</h3>
              <div className="ablation-facts">
                <div>
                  <span>Ranking loss effect</span>
                  <strong className="positive-text">+{decimal(contextEvidence.ablation.observations.ranking_effect_on_rank_ic_without_context, 3)} Rank IC</strong>
                </div>
                <div>
                  <span>Context gate effect</span>
                  <strong className="negative-text">{decimal(contextEvidence.ablation.observations.context_effect_on_rank_ic_with_ranking, 3)} Rank IC</strong>
                </div>
                <div>
                  <span>Ridge performance lead</span>
                  <strong>{pct(contextEvidence.ablation.observations.ridge_minus_best_neural_net_annualized_return)} p.a.</strong>
                </div>
              </div>
              <p>Next gate: true temporal sequences on a fresh holdout—not more tuning on 2021–2025.</p>
            </div>
          </div>
        </section>

        <section className="method-strip">
          <div className="eyebrow">The research contract</div>
          <div className="method-flow">
            {[
              ["01", "Freeze", "Checksummed market snapshot"],
              ["02", "Split", "Purged annual walk-forward"],
              ["03", "Score", "Fit on past information only"],
              ["04", "Trade", "Weekly Top-30 with costs"],
              ["05", "Audit", "Manifest every conclusion"],
            ].map(([index, title, note]) => (
              <div className="method-step" key={index}>
                <span>{index}</span><strong>{title}</strong><p>{note}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer>
        <div className="brand"><span className="brand-mark"><span /></span><span>PIT / ALPHA</span></div>
        <p>Research software, not investment advice. Historical results are not live performance.</p>
        <span>v0.1 · reproducible by construction</span>
      </footer>
    </div>
  );
}

export default App;
