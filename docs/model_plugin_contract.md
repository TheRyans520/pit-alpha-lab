# Model plugin contract

PIT Alpha Lab separates the research engine from model implementations. Built-in models and
third-party models receive the same training/test frames and produce the same audited score and
metadata contract. Portfolio construction, costs and market constraints remain owned by the engine.

## Security boundary

An installed plugin is executable Python code. The registry lists external entry-point names without
importing them, and experiment runs reject external models unless the user explicitly passes
`--allow-external-model` after reviewing the installed package. External models cannot shadow a
built-in model name.

The read-only API exposes built-in capabilities at `/api/models`; it never loads third-party code.

## Adapter interface

The entry-point factory must return `pitalpha.models.ModelAdapter`:

```python
from pitalpha.models import ModelAdapter, ModelDescriptor


def fit_predict(train, test, feature_columns, params):
    # Fit preprocessing and model state on train only.
    # Return one finite score per test row plus JSON-safe metadata.
    scores = ...
    metadata = {"training_rows": len(train), "method": "example"}
    return scores, metadata


def create_adapter():
    return ModelAdapter(
        descriptor=ModelDescriptor(
            name="my_model",
            provider="external",
            input_kind="tabular",
            description="Reviewed example model.",
        ),
        fit_predict=fit_predict,
    )
```

Register that factory in the plugin package:

```toml
[project.entry-points."pitalpha.models"]
my_model = "my_package.plugin:create_adapter"
```

Install the plugin into the same isolated project environment—never global Python—then use
`python -m pitalpha models list --include-external-names` to inspect its registered name. Selecting
the model in YAML does not grant execution permission; the run also requires
`--allow-external-model`.

## Mandatory output contract

- one-dimensional score array with exactly one finite value per test row;
- metadata containing finite JSON-serializable values only;
- deterministic output under the fixed config and seed;
- no dependence on test labels;
- no mutation of the provided training or test frames;
- no portfolio, cost or execution logic inside the adapter.

`audit_test_label_independence` repeats the adapter, perturbs only test labels and checks input-frame
integrity. Passing this audit is necessary but not sufficient: temporal models additionally require
causality and sequence-boundary tests specific to their input builder.

Temporal adapters receive `instrument` in addition to the common columns. Their training frame may
also contain embargo-session feature rows with a missing `label`; those rows are history context
only and must never enter fitting, validation selection or preprocessing state. This lets the first
test decision retain a complete causal lookback without weakening the label embargo.

## Ownership boundary

| Component | Model plugin | PIT Alpha Lab engine |
|---|---:|---:|
| Training-only preprocessing | Yes | Audits |
| Score/uncertainty generation | Yes | Validates shape and finiteness |
| Point-in-time splits | No | Yes |
| Tradability and liquidity | No | Yes |
| Portfolio weights and cash | No | Yes |
| Costs, P&L and risk | No | Yes |
| Artifact hashes and reports | No | Yes |

This boundary makes model comparison fair and prevents a plugin from reporting a private,
incompatible backtest as if it were an engine result.
