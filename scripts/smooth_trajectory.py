#!/usr/bin/env python3
"""smooth_trajectory.py — suavizado temporal post-proceso de una trayectoria TUM.

DIAGNÓSTICO del jitter (campaña Hito 3): aplica un filtro Savitzky-Golay a las
posiciones (y opcionalmente a las orientaciones) de un trajectory.txt y escribe
otro TUM. NO toca el solver: separa cuánto del jitter es **ruido de alta
frecuencia en la salida** (que un filtro elimina sin mover el neto) vs **deriva
estructural** (que el filtro no puede arreglar). Si path/neto cae cerca de ~5
tras filtrar sin degradar el ATE, el jitter es de alta frecuencia.

Uso:
  .venv/bin/python scripts/smooth_trajectory.py <in.txt> [--window 9] [--poly 2]
      [--smooth-rot] [--out <out.txt>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


def smooth_tum(data: np.ndarray, window: int = 9, poly: int = 2,
               smooth_rot: bool = False) -> np.ndarray:
    """Suaviza un array TUM (N,8: t x y z qx qy qz qw) con Savitzky-Golay sobre
    las posiciones (y opcionalmente los cuaterniones). Devuelve un array TUM
    nuevo. Quita el jitter de ALTA FRECUENCIA sin mover el neto (camino coherente
    sin tocar escala/ATE). Costo ~1 ms para 2500 poses (CPU) → no afecta fps.
    """
    t, xyz, quat = data[:, 0], data[:, 1:4], data[:, 4:8]
    win = window if window % 2 else window + 1
    win = min(win, len(t) - (1 - len(t) % 2))     # impar y ≤ N
    if win <= poly:
        return data.copy()                         # trayectoria muy corta: no filtra
    xyz_s = np.column_stack([savgol_filter(xyz[:, k], win, poly) for k in range(3)])
    if smooth_rot:
        q = quat.copy()
        for i in range(1, len(q)):                 # alinear hemisferio (evita flip de signo)
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        q_s = np.column_stack([savgol_filter(q[:, k], win, poly) for k in range(4)])
        q_s /= np.linalg.norm(q_s, axis=1, keepdims=True)
    else:
        q_s = quat
    return np.column_stack([t, xyz_s, q_s])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", type=Path)
    ap.add_argument("--window", type=int, default=9, help="Ventana savgol (impar). Default 9.")
    ap.add_argument("--poly", type=int, default=2, help="Orden polinomial savgol. Default 2.")
    ap.add_argument("--smooth-rot", action="store_true",
                    help="También suaviza los cuaterniones (sign-fix + renorm).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = np.loadtxt(args.inp)
    if data.ndim != 2 or data.shape[1] < 8:
        print("ERROR: se esperaba TUM (t x y z qx qy qz qw).")
        return 2
    xyz = data[:, 1:4]
    out_data = smooth_tum(data, args.window, args.poly, args.smooth_rot)
    xyz_s = out_data[:, 1:4]
    win = args.window if args.window % 2 else args.window + 1

    out = args.out or args.inp.with_name(args.inp.stem + f"_smooth_w{win}.txt")
    np.savetxt(out, out_data, fmt=["%.6f"] + ["%.9f"] * 7)

    def plen(p):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    net = float(np.linalg.norm(xyz[-1] - xyz[0]))
    print(f"in:  path={plen(xyz):.2f} m  neto={net:.2f} m  path/neto={plen(xyz)/net:.2f}")
    print(f"out: path={plen(xyz_s):.2f} m  neto={net:.2f} m  path/neto={plen(xyz_s)/net:.2f}"
          f"  (ventana={win}, poly={args.poly}, rot={'sí' if args.smooth_rot else 'no'})")
    print(f"escrito: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
