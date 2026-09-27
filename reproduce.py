#!/usr/bin/env python3
"""Run all checked-in experiment configurations with the local solver."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def command_for(config: dict[str, object]) -> list[str]:
    args = [
        sys.executable,
        "run.py",
        "--method", str(config["method"]),
        "--g", str(config["g"]),
        "--dt", str(config["dt"]),
        "--nx", str(config["nx"]),
        "--xmax", str(config["xmax"]),
        "--trap", str(config["trap"]),
    ]
    for key in ("quartic", "barrier"):
        if key in config:
            args.extend((f"--{key}", str(config[key])))
    return args


def main() -> None:
    experiments = sorted((ROOT / "experiments").glob("*.json"))
    if not experiments:
        raise SystemExit("No experiment JSON files found under experiments/.")

    for config_path in experiments:
        config = json.loads(config_path.read_text())
        result = subprocess.run(command_for(config), cwd=ROOT, check=False)
        if result.returncode:
            raise SystemExit(f"{config_path.name} failed with exit status {result.returncode}")

        destination = RESULTS / config_path.stem
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("summary.json", "density.png", "convergence.png"):
            shutil.copy2(RESULTS / filename, destination / filename)
        print(f"Saved {config_path.stem} outputs under {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
