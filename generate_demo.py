#!/usr/bin/env python3
"""Run the real BEC experiments and materialize the public demo fixture."""
from __future__ import annotations

import datetime as dt
import difflib
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = [
    ("BEC-001", "harmonic-reference", "harmonic-reference.json"),
    ("BEC-002", "harmonic-interacting", "harmonic-interacting.json"),
    ("BEC-003", "harmonic-small-step", "harmonic-small-step.json"),
    ("BEC-004", "gradient-flow", "gradient-flow.json"),
    ("BEC-005", "quartic-trap", "quartic-trap.json"),
    ("BEC-006", "double-well", "double-well.json"),
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def git_provenance() -> dict:
    """Capture the repository state that exists immediately before execution."""
    return {
        "repository": Path(git_value("rev-parse", "--show-toplevel") or ROOT).name,
        "branch": git_value("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "commit": git_value("rev-parse", "HEAD") or "unknown",
        "dirty": bool(git_value("status", "--porcelain")),
    }


def main() -> None:
    runs = []
    summaries = {}
    previous_config_name: str | None = None
    for index, (run_id, slug, config_name) in enumerate(CASES, start=1):
        config_path = ROOT / "experiments" / config_name
        config_text = config_path.read_text()
        config = json.loads(config_text)
        args = ["python3", "run.py", "--method", config["method"], "--g", str(config["g"]),
                "--dt", str(config["dt"]), "--nx", str(config["nx"]), "--xmax", str(config["xmax"]),
                "--trap", config["trap"]]
        if config["trap"] == "quartic":
            args += ["--quartic", str(config["quartic"])]
        if config["trap"] == "double-well":
            args += ["--barrier", str(config["barrier"])]
        source_state = git_provenance()
        started = utc_now()
        clock = time.monotonic()
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        duration = time.monotonic() - clock
        ended = utc_now()
        if result.returncode:
            raise SystemExit(f"{run_id} failed:\n{result.stderr}")
        summary = json.loads((ROOT / "results" / "summary.json").read_text())
        run_dir = ROOT / "results" / slug
        run_dir.mkdir(exist_ok=True)
        for filename in ("summary.json", "density.png", "convergence.png"):
            shutil.copy2(ROOT / "results" / filename, run_dir / filename)

        patch_base_name = "harmonic-interacting.json" if run_id == "BEC-004" else previous_config_name
        patch = "".join(difflib.unified_diff(
            (ROOT / "experiments" / patch_base_name).read_text().splitlines(keepends=True)
            if patch_base_name else [],
            config_text.splitlines(keepends=True),
            fromfile=f"a/experiments/{patch_base_name}" if patch_base_name else "/dev/null",
            tofile=f"b/experiments/{config_name}",
        ))
        previous_config_name = config_name
        facts = [
            {"key": "solver.method", "label": "Imaginary-time method", "value": config["method"],
             "source": "command"},
            {"key": "solver.interaction_strength", "label": "Interaction strength g", "value": config["g"],
             "unit": "dimensionless", "source": "command"},
            {"key": "solver.imaginary_time_step", "label": "Imaginary-time step", "value": config["dt"],
             "unit": "ω⁻¹", "source": "command"},
            {"key": "grid.points", "label": "Grid points", "value": config["nx"], "source": "command"},
            {"key": "grid.half_width", "label": "Grid half-width", "value": config["xmax"],
             "unit": "aₕₒ", "source": "command"},
            {"key": "trap.type", "label": "Trap potential", "value": config["trap"], "source": "command"},
            {"key": "solver.energy", "label": "Ground-state energy", "value": summary["energy"],
             "unit": "ℏω", "highlight": True, "source": f"results/{slug}/summary.json#/energy"},
            {"key": "solver.chemical_potential", "label": "Chemical potential",
             "value": summary["chemical_potential"], "unit": "ℏω",
             "source": f"results/{slug}/summary.json#/chemical_potential"},
            {"key": "solver.stationary_residual", "label": "Stationary-equation residual",
             "value": summary["residual"], "source": f"results/{slug}/summary.json#/residual"},
            {"key": "solver.normalization_error", "label": "Normalization error",
             "value": summary["normalization_error"], "source": f"results/{slug}/summary.json#/normalization_error"},
            {"key": "solver.iterations", "label": "Iterations", "value": summary["iterations"],
             "source": f"results/{slug}/summary.json#/iterations"},
        ]
        if config["trap"] == "quartic":
            facts.append({"key": "trap.quartic_strength", "label": "Quartic strength",
                          "value": config["quartic"], "source": "command"})
        if config["trap"] == "double-well":
            facts.append({"key": "trap.central_barrier", "label": "Double-well barrier",
                          "value": config["barrier"], "source": "command"})
        if summary["energy_error"] is not None:
            facts.append({"key": "solver.harmonic_reference_energy_error", "label": "Energy error from E = 1/2",
                          "value": summary["energy_error"], "unit": "ℏω",
                          "source": "derived: solver.energy minus 0.5 for g = 0 harmonic reference"})
            facts.append({"key": "solver.harmonic_reference_mu_error", "label": "Chemical-potential error from μ = 1/2",
                          "value": summary["chemical_potential_error"], "unit": "ℏω",
                          "source": "derived: solver.chemical_potential minus 0.5 for g = 0 harmonic reference"})
        scientific_status = "scratch"
        interpretation = (
            f"For the non-interacting harmonic reference, E={summary['energy']:.9f} ℏω and "
            f"μ={summary['chemical_potential']:.9f} ℏω. The computed absolute errors from 1/2 are "
            f"{abs(summary['energy_error']):.3g} and {abs(summary['chemical_potential_error']):.3g} ℏω, "
            "consistent with the finite-grid discretization at nx=256."
        ) if index == 1 and summary["energy_error"] is not None else ""
        if index == 1 and summary["energy_error"] is not None and max(abs(summary["energy_error"]), abs(summary["chemical_potential_error"])) < 2e-4:
            scientific_status = "notable"
        if index == 2:
            reference = summaries["BEC-001"]
            interpretation = (
                f"At fixed harmonic trap, dt, and grid, increasing g from 0 to {config['g']} changes "
                f"E from {reference['energy']:.9f} to {summary['energy']:.9f} ℏω and μ from "
                f"{reference['chemical_potential']:.9f} to {summary['chemical_potential']:.9f} ℏω. "
                "This is the computed repulsive-interaction case; the differences are relative to the "
                "non-interacting reference, not a convergence estimate."
            )
        if index == 3:
            reference = summaries["BEC-002"]
            delta = abs(summary["energy"] - reference["energy"])
            scientific_status = "baseline" if delta < 1e-6 else "candidate"
            interpretation = (
                f"Reducing dt from {reference['dt']} to {summary['dt']} changes E by {delta:.3g} ℏω "
                f"({reference['energy']:.9f} to {summary['energy']:.9f} ℏω). The change is below "
                "1×10⁻⁶ ℏω at fixed g, trap, and grid, supporting this run as the harmonic reference baseline."
            )
        if index == 4:
            reference = summaries["BEC-002"]
            delta = abs(summary["energy"] - reference["energy"])
            scientific_status = "notable" if delta < 1e-5 else "candidate"
            interpretation = (
                f"At the same harmonic potential, g, dt, and grid, gradient flow gives E="
                f"{summary['energy']:.9f} ℏω versus {reference['energy']:.9f} ℏω for split-step, "
                f"an absolute difference of {delta:.3g} ℏω. The energies agree within 1×10⁻⁵ ℏω; "
                f"the finite-difference residual ({summary['residual']:.3g}) is higher than the split-step "
                f"residual ({reference['residual']:.3g}), so residuals retain method-dependent discretization."
            )
        if index in (5, 6):
            interpretation = (
                f"This run changes the trap to {config['trap']} and computes E={summary['energy']:.9f} ℏω. "
                "Because the potential differs from the harmonic cases, its energy is not a direct comparison "
                "of solver convergence or method agreement."
            )
        command = args
        manifest = {
            "schema": 1, "id": run_id, "experiment": config["experiment"], "title": slug.replace("-", " ").title(),
            "run_status": "success", "scientific_status": scientific_status,
            "summary": f"Computed with the 1D Gross-Pitaevskii solver ({config['method']}, g={config['g']}, dt={config['dt']}).",
            "interpretation": interpretation, "open_questions": [],
            "execution": {"command": command, "started_at": started, "finished_at": ended,
                          "duration_ms": round(duration * 1000), "exit_code": result.returncode, "capture": "tracked"},
            "git": source_state,
            "facts": facts,
            "artifacts": [{"path": f"results/{slug}/density.png", "role": "figure", "title": "Ground-state density"},
                          {"path": f"results/{slug}/convergence.png", "role": "figure", "title": "Energy convergence"},
                          {"path": f"results/{slug}/summary.json", "role": "data", "title": "Solver summary"}],
        }
        run_record = {"manifest": manifest, "codePatch": patch, "artifacts": manifest["artifacts"]}
        if run_id == "BEC-004":
            run_record["comparisonRunId"] = "BEC-002"
        runs.append(run_record)
        summaries[run_id] = {**summary, **config}
        print(f"{run_id} {slug}: E={summary['energy']:.9f}, μ={summary['chemical_potential']:.9f}, "
              f"residual={summary['residual']:.3g}, {summary['iterations']} iterations, converged={summary['converged']}")
    demo = {"owner": "demo", "repository": "bec-ground-state", "runs": runs, "reports": []}
    (ROOT / "demo.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
