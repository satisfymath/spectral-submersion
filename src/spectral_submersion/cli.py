"""CLI entry point for spectral-submersion pipeline."""
import argparse
import subprocess
import sys
from pathlib import Path


def cmd_setup(args):
    """Run make setup."""
    subprocess.run(["make", "setup"], cwd=Path.cwd())


def cmd_test(args):
    """Run test suite."""
    subprocess.run(["make", "test"], cwd=Path.cwd())


def cmd_pipeline(args):
    """Run full benchmark pipeline."""
    subprocess.run(["make", "pipeline"], cwd=Path.cwd())


def cmd_report(args):
    """Generate integrated report."""
    subprocess.run(["make", "integrated-report"], cwd=Path.cwd())


def main():
    parser = argparse.ArgumentParser(
        prog="spectral-submersion",
        description="Spectral Submersion: framework for lost language decipherment",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("setup", help="Install dependencies and create venv")
    subparsers.add_parser("test", help="Run test suite")
    subparsers.add_parser("pipeline", help="Run full benchmark pipeline")
    subparsers.add_parser("report", help="Generate integrated hypothesis report")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
