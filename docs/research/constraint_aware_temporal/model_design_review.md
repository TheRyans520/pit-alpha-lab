# Model design review: literature before architecture

> Status: **engineering review, not a novelty or performance claim**
>
> Review date: 2026-09-29
>
> Evidence status: agent-opened primary sources are discovery evidence; every locator remains
> pending accountable human verification in `source_manifest.json`.

## Decision

The first PIT Alpha Lab reference model will be a compact **Constraint-Aware Temporal Mixer**
(working name), not a large generic Transformer. It will combine causal 60-session patches,
lightweight indicator/time mixing, an optional same-date market-context gate, multi-horizon heads
and uncertainty output. Its scores will pass through the framework-owned hard execution projector.

This is a synthesis to test, not a claim that the architecture is new or superior. The model earns
promotion only through the frozen protocol, unchanged baselines and component ablations.

## Evidence-screening gate

A candidate design enters implementation only when all of the following are recorded:

1. a primary paper or official proceedings record and, when available, the authors' code;
2. the original task, dataset, split and target rather than headline metrics alone;
3. the input representation, temporal and cross-sectional mechanism, objective and constraints;
4. reproducibility facts: license, dependencies, checkpoints and obvious implementation gaps;
5. the exact component proposed for reuse and a reason it fits point-in-time equity ranking;
6. an ablation that can falsify the component's value under the common portfolio ledger.

GitHub popularity is not evidence. A reported forecasting gain outside finance is not treated as an
alpha or portfolio-performance result, and a preprint is not treated as peer reviewed.

## Candidate landscape

| Candidate | Published mechanism | Relevance to this project | Decision for first model |
|---|---|---|---|
| Ridge / LightGBM | Tabular linear and tree controls | Strong frozen controls with known after-cost results | Keep unchanged as mandatory anchors |
| StockMixer, AAAI 2024 | Indicator, multiscale time and stock mixing with MLP blocks | Directly addresses small-data stock forecasting and offers a lightweight cross-sectional design | Borrow compact mixing pattern; reimplement against our causal tensor contract |
| MASTER, AAAI 2024 | Market-guided feature selection plus intra- and inter-stock aggregation | Motivates PIT market context and cross-stock information | Context remains optional because our earlier context gate failed |
| PatchTST, ICLR 2023 | Patched histories and channel-independent shared weights | Useful inductive bias for 60-session paths | Borrow causal patching; do not transfer its benchmark claims to stocks |
| TimeMixer, ICLR 2024 | Multiscale decomposition and mixing | A small multiscale branch may capture short/medium regimes | One ablation only; avoid assuming seasonal decomposition suits returns |
| iTransformer, ICLR 2024 | Whole series represented as variate tokens with cross-variate attention | Possible way to model feature interactions | Defer until the smaller mixer establishes a credible baseline |
| MOMENT, ICML 2024 | Patch-based masked pretraining across diverse public time series | Potential frozen-encoder or transfer baseline | Defer: domain transfer, size and reproducibility cost must first be justified |
| TimeBridge, ICML 2025 | Separate local non-stationarity and long-run cross-variate dependency paths | Recent financial-index evidence makes the mechanism worth tracking | Defer to a named non-stationarity ablation; its task is not stock ranking |
| ModernTCN Revisited, TMLR 2025 | Audit of data loading, validation and evaluation sensitivity | Direct support for protocol-first comparison | Adopt the evaluation lesson, not necessarily the backbone |
| Differentiable convex layer, NeurIPS 2019 | Backpropagation through a disciplined convex program | Could connect scores to constrained allocation | Later ablation only; never replaces the external hard projector |

StockMixer reports indicator, temporal and stock mixing in a lightweight MLP architecture.
[claim:C006] [evidence:E007] MASTER motivates market-guided selection and intra-/inter-stock
aggregation. [claim:C003] [evidence:E004] PatchTST motivates patching and channel-independent shared
weights. [claim:C004] [evidence:E005]

TimeMixer decomposes multiscale histories into seasonal and trend components and mixes information
across scales; this motivates, but does not validate, a small multiscale ablation for equity
returns. [claim:C009] [evidence:E010]

iTransformer represents whole series as variate tokens and applies attention across variables; its
published benchmarks do not establish cross-sectional stock-alpha performance. [claim:C010]
[evidence:E011]

MOMENT is an open family of pretrained time-series foundation models built around patching and
masked reconstruction across diverse public datasets; finance transfer remains an unanswered
project-specific question. [claim:C011] [evidence:E012]

TimeBridge separates short-term non-stationarity handling from long-term cross-variate dependency
modeling and reports experiments on CSI 500 and S&P 500 index forecasting; this task still differs
from cross-sectional stock ranking. [claim:C012] [evidence:E013]

ModernTCN Revisited reports that time-series performance conclusions can change under corrected
data loading, validation and evaluation setups; this reinforces the project's unified-protocol
requirement rather than motivating a new backbone. [claim:C013] [evidence:E014]

## Architecture selected for implementation

```text
60-session causal tensor [stock, time, feature]
        |
        +--> missing/stale/tradeability masks
        |
        v
training-only robust normalization
        |
        v
causal non-overlapping / trailing patches
        |
        v
shared indicator mixer + temporal mixer
        |
        +--> optional same-date market context gate
        |
        v
1d / 5d / 20d score heads + uncertainty head
        |
        v
framework plugin score contract
        |
        v
hard tradeability -> liquidity -> turnover -> cash projector
```

The initial capacity must stay deliberately small. Parameter count, peak memory and wall time will
be reported beside financial metrics. A larger Transformer, relation graph or foundation-model
encoder may be added only as a separately named ablation after this baseline passes leakage and
reproducibility tests.

## What is actually ours

The defensible contribution is the integrated research system, not ownership of known neural
blocks:

- a common plugin contract for built-in and user-supplied models;
- causal/PIT tensor construction and tests that deliberately attack leakage;
- model-independent hard market constraints with fail-closed missing execution state;
- one ledger for baselines, the reference model and external plugins;
- constraint, context, objective and uncertainty ablations on a locked fresh sample;
- visible negative results and evidence-linked artifacts.

No academic novelty claim will be made until a wider search, code-license review, human source
verification and empirical comparison are complete.

## Implementation order

1. Freeze the 60-session tensor schema and its timestamp/missingness contract.
2. Add boundary, causality and cross-sectional batch tests before model training.
3. Implement the smallest temporal mixer through `ModelAdapter`.
4. Reproduce it on synthetic data and development years only.
5. Add market context, multiscale, uncertainty and soft cost/risk terms one at a time.
6. Wire the hard projector only when audited PIT execution fields pass the data gate.
7. Freeze code, environment and snapshot before the single confirmatory 2026 evaluation.

## Primary records reviewed

- Fan, J., & Shen, Y. (2024). *StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock
  Price Forecasting*. AAAI 2024. <https://doi.org/10.1609/aaai.v38i8.28681>
- Li, T. et al. (2024). *MASTER: Market-Guided Stock Transformer for Stock Price Forecasting*.
  AAAI 2024. <https://ojs.aaai.org/index.php/AAAI/article/view/27767>
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers*. ICLR 2023. <https://github.com/yuqinie98/PatchTST>
- Wang, S. et al. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting*.
  ICLR 2024. <https://arxiv.org/abs/2405.14616>
- Liu, Y. et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series
  Forecasting*. ICLR 2024. <https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html>
- Goswami, M. et al. (2024). *MOMENT: A Family of Open Time-series Foundation Models*. ICML 2024.
  <https://proceedings.mlr.press/v235/goswami24a.html>
- Liu, P. et al. (2025). *TimeBridge: Non-Stationarity Matters for Long-term Time Series
  Forecasting*. ICML 2025. <https://openreview.net/forum?id=pyKO0ZZ5lz>
- Akacik, O., & Hoogendoorn, M. (2025). *ModernTCN Revisited: A Critical Look at the Experimental
  Setup in General Time Series Analysis*. TMLR. <https://openreview.net/forum?id=R20kKdWmVZ>
