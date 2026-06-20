"""Tests for AnvilAgent setup scripts."""

from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
TESTS_DIR = Path(__file__).resolve().parent

REQUIRED_SCRIPTS = [
    "anvil_setup.sh",
    "download_models.sh",
]

REQUIRED_SERVICE = "anvil-inference.service"


# ── File existence & permissions ───────────────────────────────────────


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_script_exists_and_executable(script: str) -> None:
    path = SCRIPTS_DIR / script
    assert path.is_file(), f"Script not found: {path}"
    mode = path.stat().st_mode
    assert mode & stat.S_IXUSR, f"Script is not executable: {path}"
    assert mode & stat.S_IXGRP, f"Script is not group-executable: {path}"


def test_service_file_exists() -> None:
    path = SCRIPTS_DIR / REQUIRED_SERVICE
    assert path.is_file(), f"Service file not found: {path}"


# ── Bash syntax checks (shellcheck-like) ───────────────────────────────


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_bash_shebang(script: str) -> None:
    path = SCRIPTS_DIR / script
    content = path.read_text()
    assert content.startswith("#!/usr/bin/env bash"), f"Missing bash shebang in {script}"


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_set_euo_pipefail(script: str) -> None:
    path = SCRIPTS_DIR / script
    content = path.read_text()
    assert "set -euo pipefail" in content, f"Missing 'set -euo pipefail' in {script}"


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_no_taboo_patterns(script: str) -> None:
    """Reject patterns that indicate broken or insecure scripts."""
    path = SCRIPTS_DIR / script
    content = path.read_text()

    taboo = [
        "eval ",
        "`",
        "rm -rf /",
        "sudo rm -rf",
    ]
    for pattern in taboo:
        assert pattern not in content, f"Taboo pattern '{pattern}' found in {script}"


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_bash_syntax(script: str) -> None:
    """Run bash -n to verify syntax without executing."""
    path = SCRIPTS_DIR / script
    result = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Syntax error in {script}:\n{result.stderr}"
    )


# ── Content structure checks ───────────────────────────────────────────


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_has_main_function(script: str) -> None:
    path = SCRIPTS_DIR / script
    content = path.read_text()
    assert "main()" in content or "main() {" in content, (
        f"No main() function found in {script}"
    )


@pytest.mark.parametrize("script", REQUIRED_SCRIPTS)
def test_uses_bracket_test_syntax(script: str) -> None:
    """Ensure scripts use modern [[ ]] instead of legacy [ ] for tests."""
    path = SCRIPTS_DIR / script
    content = path.read_text()
    # Allow [ -f ] for single-argument file tests but flag anything else
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments and string literals
        if stripped.startswith("#"):
            continue
        if "[[" in stripped:
            continue
        if stripped.startswith("["):
            # Allow certain basic POSIX constructs
            if any(
                stripped.startswith(f"[{x}") or stripped.startswith(f"[ {x}")
                for x in ("$", "-z", "-n", "-f", "-d", "-x", "-L")
            ):
                continue
            # It might be a harmless single-test expression
            if stripped.count("[") == 1 and stripped.count("]") == 1:
                inner = stripped[stripped.index("[") + 1 : stripped.index("]")]
                inner = inner.strip().strip('"').strip("'")
                # Single word tests like [ -f file ] or [ "$var" ] are OK
                if inner.startswith(("-f ", "-d ", "-x ", "-L ", "-z ", "-n ")):
                    continue
                if inner.startswith("$"):
                    continue
                pytest.fail(
                    f"Line {i}: '{stripped}' uses legacy '[ ]'. Use '[[ ]]' instead."
                )


def test_service_has_required_keys() -> None:
    path = SCRIPTS_DIR / REQUIRED_SERVICE
    content = path.read_text()
    assert "[Unit]" in content
    assert "[Service]" in content
    assert "[Install]" in content
    assert "ExecStart=" in content
    assert "User=" in content
    assert "Group=" in content
    assert "Restart=always" in content
    assert "WantedBy=multi-user.target" in content
    assert "CPUSchedulingPolicy=rr" in content


# ── Model registry sanity check ────────────────────────────────────────


def test_model_registry_has_entries() -> None:
    """Parse the model registry from download_models.sh and validate entries."""
    path = SCRIPTS_DIR / "download_models.sh"
    content = path.read_text()

    model_keys_found = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("MODELS[") and stripped.endswith('"'):
            model_keys_found += 1

    assert model_keys_found >= 3, (
        f"Expected at least 3 model entries in MODELS, found {model_keys_found}"
    )


# ── Test script self-check ─────────────────────────────────────────────


def test_this_file_is_runnable() -> None:
    """Verify that the test suite itself can be discovered by pytest."""
    path = TESTS_DIR / "test_setup.py"
    assert path.is_file()
    assert path.suffix == ".py"
