"""Command-line interface for environment and experiment-contract operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from pitalpha import __version__
from pitalpha.artifacts import build_run_manifest, write_json_atomic
from pitalpha.config import ConfigError, config_digest, load_config
from pitalpha.environment import environment_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitalpha", description="PIT Alpha Lab research tooling")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="Report interpreter and isolation status")
    execution_demo = commands.add_parser("execution-demo", help="Run the synthetic cash-funded execution case study")
    execution_demo.add_argument("--output-root", type=Path, default=None)
    audit_run = commands.add_parser("audit-run", help="Verify and diagnose a saved historical research run")
    audit_run.add_argument("run_directory", type=Path)
    audit_run.add_argument("--output", type=Path, default=None)

    config_parser = commands.add_parser("config", help="Configuration operations")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    validate = config_commands.add_parser("validate", help="Validate and identify a YAML config")
    validate.add_argument("path", type=Path)

    manifest_parser = commands.add_parser("manifest", help="Run-manifest operations")
    manifest_commands = manifest_parser.add_subparsers(dest="manifest_command", required=True)
    create = manifest_commands.add_parser("create", help="Create a planned run manifest")
    create.add_argument("config", type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--code-revision", default=None)

    models_parser = commands.add_parser("models", help="Inspect the model adapter registry")
    model_commands = models_parser.add_subparsers(dest="models_command", required=True)
    list_models = model_commands.add_parser("list", help="List built-in and installed model names")
    list_models.add_argument(
        "--include-external-names",
        action="store_true",
        help="List installed third-party entry-point names without loading their code",
    )

    run_parser = commands.add_parser("run", help="Execute a supported experiment config")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--output-root", type=Path, default=None)
    run_parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="Override model randomness while recording a distinct resolved config",
    )
    run_parser.add_argument(
        "--disable-market-context",
        action="store_true",
        help="Ablate same-date market context while keeping the date-batched objective",
    )
    run_parser.add_argument(
        "--ranking-weight",
        type=float,
        default=None,
        help="Override the within-date ranking-loss weight for an auditable ablation",
    )
    run_parser.add_argument(
        "--allow-external-model",
        action="store_true",
        help="Execute the selected third-party pitalpha.models entry point after manual review",
    )
    serve_parser = commands.add_parser("serve", help="Serve the read-only research API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-run":
            from pitalpha.legacy_audit import audit_saved_run
            result = audit_saved_run(args.run_directory)
            if args.output is not None:
                write_json_atomic(args.output, result)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "execution-demo":
            from pitalpha.execution_demo import run_execution_demo
            print(json.dumps(run_execution_demo(args.output_root), indent=2))
            return 0
        if args.command == "doctor":
            report = environment_report()
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0 if report["isolated_environment"] and report["expected_project_environment"] else 1

        if args.command == "config" and args.config_command == "validate":
            config = load_config(args.path)
            print(json.dumps({"valid": True, "path": str(args.path), "sha256": config_digest(config)}, indent=2))
            return 0

        if args.command == "manifest" and args.manifest_command == "create":
            config = load_config(args.config)
            manifest = build_run_manifest(config, args.config, code_revision=args.code_revision)
            output = write_json_atomic(args.output, manifest)
            print(json.dumps({"created": str(output), "run_id": manifest["run_id"]}, indent=2))
            return 0
        if args.command == "models" and args.models_command == "list":
            from pitalpha.models import ENTRY_POINT_GROUP, list_model_descriptors

            payload = {
                "entry_point_group": ENTRY_POINT_GROUP,
                "models": list_model_descriptors(include_external_names=args.include_external_names),
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        if args.command == "run":
            from pitalpha.pipeline import run_experiment

            result = run_experiment(
                args.config,
                output_root=args.output_root,
                model_seed_override=args.model_seed,
                disable_market_context=args.disable_market_context,
                ranking_weight_override=args.ranking_weight,
                allow_external_model=args.allow_external_model,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "serve":
            import uvicorn

            uvicorn.run("pitalpha.api:app", host=args.host, port=args.port, reload=False)
            return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error("unsupported command")
    return 2
