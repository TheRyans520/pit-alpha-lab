# Cash-funded execution accounting: synthetic case study

This L0 fixture demonstrates cash and inventory accounting with execution constraints.
It supplies its own prices and side masks; it is not an alpha result or validated exchange model.

```bash
.venv/bin/python -m pitalpha execution-demo
```

The command produces an input snapshot, daily cash/NAV, per-instrument quantities and fills,
independent reconciliation, a report and SHA-256 manifest. The read-only model-results API
excludes this different artifact type, so synthetic accounting is not mixed into model rankings.

## Accounting contract

At an execution event the engine marks existing inventory, computes pre-trade NAV, and projects
desired weights using the shared constraint kernel. Sells execute before buys. With proportional
fee rate `f`, the maximum affordable purchase notional is `cash_after_sales / (1 + f)`.
All buy and sell fees are deducted from cash. The check is:

```text
closing_cash = opening_cash + sale_proceeds - purchases - all_fees
closing_NAV = closing_cash + marked_inventory = pre_trade_NAV - all_fees
```

This differs from the historical CSI300 ledger's gross-weight return series with a linear cost
overlay. In the new engine, fees affect affordable shares and therefore subsequent positions.
Target and participation caps use pre-trade NAV. Fees can change post-trade normalized weights;
post-fee concentration guarantees are not claimed.

## Five deterministic events

Initial capital is 100,000 units; one-way fees are 10 bps.

| Event | Scenario | Expected behavior |
|---|---|---|
| 1 | Buy B blocked; 50% desired allocation to A | Buy A; retain unused cash; pay entry fee |
| 2 | Sale of A blocked; full allocation desired in B | Keep A; buy B only with remaining cash after fees |
| 3 | A/B ADV limits bind | Sell at most 10,000 of A; buy at most 8,000 of B |
| 4 | Mark-only observation | Inventory changes value; no orders or fees |
| 5 | Liquidation | Sell remaining shares; charge exit fees; hold cash |

The independently checked result is recorded in `results/reconciliation.json`.
`results/execution_daily.csv` and `execution_positions.csv` are small, fully synthetic derived
outputs. Their price path was chosen to exercise accounting branches and is not a return forecast.

## What the tests establish

- An analytic one-stock entry/exit agrees with fee-inclusive share arithmetic.
- A blocked sale cannot finance a replacement buy.
- Missing held marks fail; missing side-mask entries block that side.
- Future-dated decisions/ADV and out-of-order events fail.
- Perturbing a later price cannot change earlier positions.
- Eight seeded 15-event paths satisfy nonnegative cash/inventory and turnover caps and reconcile
  independently. These are software stress tests, not model seed robustness evidence.
- Changing a reported cash balance causes reconciliation failure.

## Scope limits

Fractional shares, immediate settlement, zero cash interest and supplied marks are assumed.
No venue-specific lot sizes, queue positions, partial fills, slippage, market impact or corporate
cash/share actions are modeled. ADV clips are mathematical scenario constraints, not calibrated
capacity estimates. An already breached gross-exposure cap fails closed in the shared kernel.
Real mask construction and an all-held-security valuation feed are still required before connecting
this path to a real-market research study. The old CSI300 result files remain unchanged.
