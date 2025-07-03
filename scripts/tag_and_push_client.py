#!/usr/bin/env python3
"""
Script to tag and push agent-memory-client releases.

This script:
1. Reads the current version from agent-memory-client/__init__.py
2. Creates a git tag in the format: client/v{version}
3. Pushes the tag to origin

Usage:
    python scripts/tag_and_push_client.py [--dry-run]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_client_version() -> str:
    """Read the version from agent-memory-client/__init__.py"""
    init_file = Path("agent-memory-client/agent_memory_client/__init__.py")

    if not init_file.exists():
        raise FileNotFoundError(f"Could not find {init_file}")

    content = init_file.read_text()

    # Look for __version__ = "x.y.z"
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)

    if not version_match:
        raise ValueError(f"Could not find __version__ in {init_file}")

    return version_match.group(1)


def run_command(cmd: list[str], dry_run: bool = False) -> subprocess.CompletedProcess:
    """Run a command, optionally in dry-run mode."""
    print(f"Running: {' '.join(cmd)}")

    if dry_run:
        print("  (dry-run mode - command not executed)")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr.strip()}")
        raise


def check_git_status():
    """Check if git working directory is clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print("Warning: Git working directory is not clean:")
            print(result.stdout)
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                sys.exit(1)
    except subprocess.CalledProcessError:
        print("Error: Could not check git status")
        sys.exit(1)


def tag_exists(tag_name: str) -> bool:
    """Check if a tag already exists."""
    try:
        subprocess.run(
            ["git", "rev-parse", f"refs/tags/{tag_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Tag and push agent-memory-client release"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force tag creation even if tag already exists",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Add '-test' suffix to tag for TestPyPI deployment",
    )

    args = parser.parse_args()

    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    try:
        original_cwd = Path.cwd()
        if project_root.resolve() != original_cwd.resolve():
            print(f"Changing to project root: {project_root}")
            import os

            os.chdir(project_root)
    except Exception as e:
        print(f"Warning: Could not change to project root: {e}")

    try:
        # Get the current version
        version = get_client_version()
        tag_suffix = "-test" if args.test else ""
        tag_name = f"client/v{version}{tag_suffix}"

        print(f"Current client version: {version}")
        print(f"Tag to create: {tag_name}")
        print(f"Deployment target: {'TestPyPI' if args.test else 'PyPI (Production)'}")

        if not args.dry_run:
            # Check git status
            check_git_status()

            # Check if tag already exists
            if tag_exists(tag_name):
                if args.force:
                    print(f"Tag {tag_name} already exists, but --force specified")
                    run_command(["git", "tag", "-d", tag_name], args.dry_run)
                else:
                    print(
                        f"Error: Tag {tag_name} already exists. Use --force to overwrite."
                    )
                    sys.exit(1)

        # Create the tag
        run_command(["git", "tag", tag_name], args.dry_run)

        # Push the tag
        push_cmd = ["git", "push", "origin", tag_name]
        if args.force:
            push_cmd.insert(2, "--force")

        run_command(push_cmd, args.dry_run)

        print(f"\n✅ Successfully tagged and pushed {tag_name}")

        if not args.dry_run:
            print("\nThis should trigger the GitHub Actions workflow for:")
            if args.test:
                print("  - TestPyPI publication (testing)")
            else:
                print("  - PyPI publication (production)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
