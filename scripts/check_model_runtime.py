# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Consistency checks for the provider-neutral model runtime package boundary."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.1.0.dev0"

RUNTIME = REPO_ROOT / "commerce-model-runtime"
COMMON = REPO_ROOT / "commerce-common"
SHOPPING_RUNTIME = REPO_ROOT / "shopping-agent" / "runtime-messages-api"
MERCHANT_RUNTIME = REPO_ROOT / "merchant-agent" / "runtime-messages-api"

FORBIDDEN_WIRE_TOKENS = (
    "content_block_delta",
    "input_json_delta",
    "cache_control",
    "eager_input_streaming",
)


def project(path: Path) -> dict:
    return tomllib.loads((path / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def dependency(project_data: dict, prefix: str) -> str | None:
    return next((item for item in project_data.get("dependencies", []) if item.startswith(prefix)), None)


def python_files(root: Path):
    yield from root.rglob("*.py")


def main() -> int:
    problems: list[str] = []

    runtime_project = project(RUNTIME)
    common_project = project(COMMON)
    shopping_project = project(SHOPPING_RUNTIME)
    merchant_project = project(MERCHANT_RUNTIME)

    for label, data in (
        ("commerce-model-runtime", runtime_project),
        ("commerce-common", common_project),
        ("shopping-agent-runtime", shopping_project),
        ("merchant-agent-runtime", merchant_project),
    ):
        if data["version"] != VERSION:
            problems.append(f"{label}: expected version {VERSION}, got {data['version']}")

    if dependency(common_project, "commerce-model-runtime") != f"commerce-model-runtime=={VERSION}":
        problems.append("commerce-common must pin commerce-model-runtime exactly")
    if dependency(shopping_project, "commerce-model-runtime") != (
        f"commerce-model-runtime[anthropic]=={VERSION}"
    ):
        problems.append("shopping Messages runtime must pin commerce-model-runtime[anthropic]")
    if dependency(merchant_project, "commerce-model-runtime") != (
        f"commerce-model-runtime[anthropic]=={VERSION}"
    ):
        problems.append("merchant Messages runtime must pin commerce-model-runtime[anthropic]")

    common_dependencies = common_project.get("dependencies", [])
    if any(item.split("[", 1)[0].split("=", 1)[0].strip() == "anthropic" for item in common_dependencies):
        problems.append("commerce-common must not depend directly on anthropic")

    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    editables = [line.strip() for line in requirements if line.strip().startswith("-e ")]
    if not editables or editables[0] != "-e ./commerce-model-runtime[anthropic]":
        problems.append("requirements.txt must install commerce-model-runtime[anthropic] first")

    leakage_roots = (
        COMMON / "commerce_common",
        SHOPPING_RUNTIME / "shopping_agent_runtime",
        MERCHANT_RUNTIME / "merchant_agent_runtime",
    )
    for path in (file for root in leakage_roots for file in python_files(root)):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_WIRE_TOKENS:
            if token in text:
                problems.append(f"provider wire token {token!r} leaked into {path.relative_to(REPO_ROOT)}")

    if problems:
        for problem in problems:
            print(f"  ✗ {problem}")
        return 1

    print("  ✓ provider-neutral runtime versions and sibling pins are consistent")
    print("  ✓ commerce-common has no direct Anthropic dependency")
    print("  ✓ provider wire event/cache tokens are confined to provider adapters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
