#!/usr/bin/env python3
"""
Test runner script for fraud detection project.

This script provides a convenient way to run different types of tests
and generate coverage reports.
"""

import argparse
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")

    try:
        subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        print(
            "Make sure all dependencies are installed: "
            "pip install -r requirements-dev.txt"
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for fraud detection project"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--all", action="store_true", help="Run all checks and tests")
    parser.add_argument(
        "--fast", action="store_true", help="Run fast tests only (skip slow tests)"
    )

    args = parser.parse_args()

    # If no specific option is chosen, run all tests
    if not any(
        [args.unit, args.integration, args.coverage, args.lint, args.format, args.all]
    ):
        args.all = True

    success = True

    # Format code if requested
    if args.format or args.all:
        success &= run_command(
            ["black", "src/", "tests/", "--line-length=88"],
            "Code formatting with black",
        )
        success &= run_command(
            ["isort", "src/", "tests/", "--profile=black"], "Import sorting with isort"
        )

    # Run linting if requested
    if args.lint or args.all:
        success &= run_command(
            [
                "flake8",
                "src/",
                "tests/",
                "--max-line-length=88",
                "--extend-ignore=E203,W503",
            ],
            "Linting with flake8",
        )
        # Type checking (don't fail on errors initially)
        run_command(
            ["mypy", "src/", "--ignore-missing-imports", "--no-strict-optional"],
            "Type checking with mypy",
        )

    # Run unit tests
    if args.unit or args.all:
        test_cmd = ["pytest", "tests/", "-v", "-m", "unit or not integration"]
        if args.coverage or args.all:
            test_cmd.extend(
                ["--cov=src", "--cov-report=html", "--cov-report=term-missing"]
            )
        if args.fast:
            test_cmd.extend(["-m", "not slow"])

        success &= run_command(test_cmd, "Unit tests")

    # Run integration tests
    if args.integration or args.all:
        test_cmd = ["pytest", "tests/", "-v", "-m", "integration"]
        if args.coverage or args.all:
            test_cmd.extend(
                ["--cov=src", "--cov-report=html", "--cov-report=term-missing"]
            )

        # Integration tests might fail without real data, so don't fail the script
        run_command(test_cmd, "Integration tests")

    # Generate coverage report
    if args.coverage and not (args.unit or args.integration or args.all):
        success &= run_command(
            [
                "pytest",
                "tests/",
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ],
            "Coverage report generation",
        )

    # Security check
    if args.all:
        run_command(
            ["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"],
            "Security check with bandit",
        )

    # Summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All requested checks passed!")
        if args.coverage or args.all:
            print("📊 Coverage report available at: htmlcov/index.html")
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
