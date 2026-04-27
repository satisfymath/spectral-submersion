"""CLI entry point for spectral-submersion pipeline.

Usage:
    spectral-submersion --help
    spectral-submersion run-pipeline --config configs/base.yaml
    spectral-submersion validate --config configs/synthetic.yaml
    spectral-submersion report --output reports/final/
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    env = {"PYTHONPATH": "src", **{k: v for k, v in subprocess.os.environ.items() if k != "PYTHONPATH"}}
    result = subprocess.run(cmd, cwd=cwd or Path.cwd(), env=env)
    return result.returncode


def cmd_setup(args):
    """Install dependencies."""
    return _run([sys.executable, "-m", "venv", ".venv"]) or \
           _run([".venv/bin/pip", "install", "--upgrade", "pip"]) or \
           _run([".venv/bin/pip", "install", "-r", "requirements.txt"])


def cmd_test(args):
    """Run test suite."""
    return _run([".venv/bin/python", "-m", "pytest", "tests/", "-v"])


def cmd_run_pipeline(args):
    """Run full benchmark pipeline."""
    return _run(["make", "pipeline"])


def cmd_validate(args):
    """Run validation on a specific corpus."""
    print(f"Validating with config: {args.config}")
    # Load config for parameters
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = cfg.get("data", {}).get("lost_corpus_path", "data/raw/lost_language/corpus.csv")
    print(f"Corpus: {data_path}")

    # Run controls
    ret = _run([
        ".venv/bin/python", "scripts/run_negative_controls.py",
        "--input", data_path,
        "--output", "reports/tables/validation_controls.csv",
    ])
    if ret != 0:
        return ret

    # Run bootstrap
    ret = _run([
        ".venv/bin/python", "scripts/run_bootstrap.py",
        "--input", data_path,
        "--output", "reports/tables/validation_bootstrap.csv",
        "--n-bootstrap", "30",
    ])
    return ret


def cmd_report(args):
    """Generate integrated report."""
    return _run(["make", "integrated-report"])


def main():
    parser = argparse.ArgumentParser(
        prog="spectral-submersion",
        description="Spectral Submersion: framework for lost language decipherment",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup
    subparsers.add_parser("setup", help="Install dependencies")

    # test
    subparsers.add_parser("test", help="Run test suite")

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full benchmark pipeline")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a corpus against controls")
    p_val.add_argument("--config", default="configs/base.yaml", help="YAML config file")

    # report
    p_rep = subparsers.add_parser("report", help="Generate integrated report")
    p_rep.add_argument("--output", default="reports/final", help="Output directory")

    args = parser.parse_args()

    commands = {
        "setup": cmd_setup,
        "test": cmd_test,
        "pipeline": cmd_run_pipeline,
        "validate": cmd_validate,
        "report": cmd_report,
    }

    if args.command in commands:
        sys.exit(commands[args.command](args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
