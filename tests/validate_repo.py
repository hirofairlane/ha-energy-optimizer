#!/usr/bin/env python3
"""Static repo validation used by CI (and runnable locally).

Checks:
  * the add-on / repository YAML files parse,
  * config.yaml `version` matches the ADD_ON_VERSION constant in the *shipped*
    entrypoint (rootfs/usr/bin/energy_optimizer.py — this is the file the
    Dockerfile COPYs into the image; the root-level energy_optimizer.py is a
    stale pre-v5 snapshot kept only for diffing and is intentionally ignored),
  * every base image declared in build.yaml covers an arch listed in
    config.yaml (the failure mode that breaks installs on a missing arch).

Exit non-zero on any failure.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = ROOT / "rootfs" / "usr" / "bin" / "energy_optimizer.py"


def main() -> int:
    errors: list[str] = []

    for rel in ("config.yaml", "build.yaml", "repository.yaml"):
        try:
            yaml.safe_load((ROOT / rel).read_text())
            print(f"OK   yaml {rel}")
        except Exception as exc:  # noqa: BLE001 - report any parse error
            errors.append(f"YAML parse failed for {rel}: {exc}")

    # ── version sync: config.yaml <-> shipped entrypoint ──────────────────────
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg_version = config.get("version")
    src = ENTRYPOINT.read_text()
    m = re.search(r'ADD_ON_VERSION\s*=\s*"([^"]+)"', src)
    src_version = m.group(1) if m else None
    if cfg_version == src_version:
        print(f"OK   version in sync: {cfg_version}")
    else:
        errors.append(
            f"version mismatch: config.yaml={cfg_version} "
            f"energy_optimizer.py ADD_ON_VERSION={src_version}"
        )

    # ── arch coverage: every config.yaml arch has a build.yaml base image ──────
    # Mirrors genia-air-ha's per-arch guard: a missing base image silently
    # breaks the add-on build for that architecture at install time.
    build = yaml.safe_load((ROOT / "build.yaml").read_text())
    declared_arches = set(config.get("arch", []))
    build_from = set((build or {}).get("build_from", {}).keys())
    missing = declared_arches - build_from
    if missing:
        errors.append(
            f"build.yaml missing base image for arch(es): {sorted(missing)} "
            f"(config.yaml declares {sorted(declared_arches)}, "
            f"build.yaml provides {sorted(build_from)})"
        )
    else:
        print(f"OK   arch coverage: {sorted(declared_arches)}")

    if errors:
        print("\nFAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("\nAll static checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
