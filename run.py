#!/usr/bin/env python3
"""Small 1D Gross–Pitaevskii ground-state solver for the SRS demo."""
from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import factorized


def trap_potential(x: np.ndarray, kind: str, quartic: float, barrier: float) -> np.ndarray:
    harmonic = 0.5 * x**2
    if kind == "harmonic":
        return harmonic
    if kind == "quartic":
        return harmonic + quartic * x**4
    return harmonic + barrier * np.exp(-x**2 / 2)


def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    return psi / math.sqrt(float(np.sum(np.abs(psi) ** 2) * dx))


def energy_mu(psi: np.ndarray, x: np.ndarray, dx: float, g: float, potential: np.ndarray) -> tuple[float, float]:
    lap = (np.roll(psi, 1) - 2 * psi + np.roll(psi, -1)) / dx**2
    lap[[0, -1]] = 0.0  # boundary values are negligible on the chosen domain
    kinetic_density = -0.5 * np.conj(psi) * lap
    density = np.abs(psi) ** 2
    mu = float(np.sum((kinetic_density + (potential + g * density) * density) * dx).real)
    energy = float(np.sum((kinetic_density + (potential + 0.5 * g * density) * density) * dx).real)
    return energy, mu


def residual(psi: np.ndarray, dx: float, g: float, potential: np.ndarray, mu: float) -> float:
    lap = (np.roll(psi, 1) - 2 * psi + np.roll(psi, -1)) / dx**2
    lap[[0, -1]] = 0.0
    r = -0.5 * lap + (potential + g * np.abs(psi) ** 2 - mu) * psi
    return float(math.sqrt(float(np.sum(np.abs(r) ** 2) * dx)))


def solve(args: argparse.Namespace) -> dict:
    x = np.linspace(-args.xmax, args.xmax, args.nx)
    dx = float(x[1] - x[0])
    potential = trap_potential(x, args.trap, args.quartic, args.barrier)
    psi = normalize(np.exp(-x**2 / 2), dx).astype(complex)
    k = 2 * np.pi * np.fft.fftfreq(args.nx, d=dx)
    # The split operator is the Fourier kinetic propagator and a real-space
    # potential/nonlinearity propagator, with normalization after each step.
    kinetic_half = np.exp(-0.25 * args.dt * k**2)
    iterations = 0
    errors: list[float] = []
    previous_energy = math.inf
    if args.method == "gradient-flow":
        main = np.full(args.nx, 1 / dx**2)
        main[[0, -1]] = 1 / dx**2
        off = np.full(args.nx - 1, -0.5 / dx**2)
        lap_h = diags([off, main, off], [-1, 0, 1], format="csc")
        h = lap_h + diags(potential, format="csc")
        # Semi-implicit normalized gradient flow: solve the linear kinetic and
        # trap step, then iterate the local nonlinear term and renormalize.
        step = factorized((eye(args.nx, format="csc") + args.dt * h).tocsc())
    for iterations in range(1, 30001):
        if args.method == "split-step":
            psi = np.fft.ifft(kinetic_half * np.fft.fft(psi))
            psi *= np.exp(-args.dt * (potential + args.g * np.abs(psi) ** 2))
            psi = np.fft.ifft(kinetic_half * np.fft.fft(psi))
        else:
            for _ in range(3):
                rhs = psi - args.dt * args.g * np.abs(psi) ** 2 * psi
                psi = step(rhs.real) + 1j * step(rhs.imag)
        psi = normalize(psi, dx)
        energy, mu = energy_mu(psi, x, dx, args.g, potential)
        errors.append(abs(energy - previous_energy) if math.isfinite(previous_energy) else 1.0)
        # The reported residual uses a finite-difference diagnostic, so its
        # floor includes the second-order grid error even for Fourier evolution.
        if errors[-1] < args.tolerance and residual(psi, dx, args.g, potential, mu) < 2e-3:
            break
        previous_energy = energy
    energy, mu = energy_mu(psi, x, dx, args.g, potential)
    return {
        "method": args.method, "trap": args.trap, "g": args.g, "dt": args.dt,
        "nx": args.nx, "xmax": args.xmax, "quartic": args.quartic, "barrier": args.barrier,
        "energy": energy, "chemical_potential": mu,
        "normalization": float(np.sum(np.abs(psi) ** 2) * dx),
        "normalization_error": abs(float(np.sum(np.abs(psi) ** 2) * dx) - 1.0),
        "residual": residual(psi, dx, args.g, potential, mu),
        "iterations": iterations, "converged": iterations < 30000,
        "x": x.tolist(), "density": np.abs(psi).astype(float).__mul__(np.abs(psi)).tolist(),
        "energy_error": energy - 0.5 if args.g == 0 and args.trap == "harmonic" else None,
        "chemical_potential_error": mu - 0.5 if args.g == 0 and args.trap == "harmonic" else None,
        "convergence": errors,
    }


def png_line_chart(path: Path, series: list[tuple[list[float], tuple[int, int, int]]], title: str) -> None:
    """Write a tiny dependency-free RGB PNG line chart."""
    width, height = 900, 520
    pix = bytearray([255, 255, 255] * width * height)
    def point(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pix[i:i+3] = bytes(color)
    # axes and plot bounds
    left, right, top, bottom = 72, width - 24, 48, height - 58
    for xx in range(left, right + 1): point(xx, bottom, (50, 50, 50))
    for yy in range(top, bottom + 1): point(left, yy, (50, 50, 50))
    del title  # figure data and captions live in the JSON and demo page
    for values, color in series:
        if not values: continue
        finite = np.asarray(values, dtype=float)
        finite = np.maximum(finite, 1e-16)
        logged = np.log10(finite)
        lo, hi = float(logged.min()), float(logged.max())
        if hi == lo: hi += 1
        old = None
        for j, val in enumerate(logged):
            xx = left + int(j * (right-left) / max(1, len(logged)-1))
            yy = bottom - int((val-lo) * (bottom-top) / (hi-lo))
            if old:
                x0,y0 = old
                steps=max(abs(xx-x0),abs(yy-y0),1)
                for t in range(steps+1): point(x0+(xx-x0)*t//steps,y0+(yy-y0)*t//steps,color)
            old=(xx,yy)
    raw = b"".join(b"\0" + bytes(pix[y*width*3:(y+1)*width*3]) for y in range(height))
    def chunk(tag: bytes, data: bytes) -> bytes:
        body=tag+data
        return struct.pack(">I",len(data))+body+struct.pack(">I",zlib.crc32(body)&0xffffffff)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">2I5B",width,height,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["split-step", "gradient-flow"], default="split-step")
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--xmax", type=float, default=8.0)
    parser.add_argument("--trap", choices=["harmonic", "quartic", "double-well"], default="harmonic")
    parser.add_argument("--quartic", type=float, default=0.01)
    parser.add_argument("--barrier", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    if args.nx < 64 or args.xmax <= 0 or args.dt <= 0 or args.g < 0:
        parser.error("require nx >= 64, xmax > 0, dt > 0 and g >= 0")
    result = solve(args)
    output = Path("results")
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps({k:v for k,v in result.items() if k not in ("x", "density", "convergence")}, indent=2)+"\n")
    png_line_chart(output / "density.png", [(result["density"], (25, 91, 153))], "Ground-state density")
    png_line_chart(output / "convergence.png", [(result["convergence"], (185, 72, 52))], "Energy convergence")
    print(json.dumps({k:v for k,v in result.items() if k not in ("x", "density", "convergence")}, indent=2))


if __name__ == "__main__":
    main()
