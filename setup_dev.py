#!/usr/bin/env python3
"""
Development environment setup script for fraud detection project.

This script helps set up the development environment with all necessary
dependencies and tools.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, check=check, shell=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"⚠️  {description} completed with warnings")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def main():
    print("🚀 Setting up fraud detection development environment...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(
            f"❌ Python {python_version.major}.{python_version.minor} "
            "is not supported. Please use Python 3.8+"
        )
        sys.exit(1)

    print(
        f"✅ Python {python_version.major}.{python_version.minor}."
        f"{python_version.micro} detected"
    )

    # Install production dependencies
    success = run_command(
        "pip install -r requirements.txt", "Installing production dependencies"
    )

    if not success:
        print("❌ Failed to install production dependencies")
        sys.exit(1)

    # Install development dependencies
    success = run_command(
        "pip install -r requirements-dev.txt", "Installing development dependencies"
    )

    if not success:
        print("⚠️  Some development dependencies failed to install")
        print("You can continue, but some tools might not work properly")

    # Set up pre-commit hooks
    run_command("pre-commit install", "Setting up pre-commit hooks", check=False)

    # Create necessary directories
    directories = ["logs", "htmlcov", "experiments"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created/verified directory: {directory}")

    # Run initial tests to verify setup
    print("\n🧪 Running initial tests to verify setup...")
    test_result = run_command(
        "python -m pytest tests/ -v --tb=short -x",
        "Running verification tests",
        check=False,
    )

    # Summary
    print(f"\n{'='*60}")
    print("🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Run 'make test' to run all tests")
    print("2. Run 'make format' to format your code")
    print("3. Run 'make lint' to check code quality")
    print("4. Run 'python run_tests.py --help' for more testing options")

    if not test_result:
        print("\n⚠️  Note: Some tests failed during verification.")
        print("This might be normal if you don't have real data files yet.")
        print("The development environment is still ready to use.")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
