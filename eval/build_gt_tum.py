#!/usr/bin/env python3
"""build_gt_tum.py — Genera gt.txt en formato TUM desde los timecodes de
cintas naranjas reportados por el usuario (2026-06-10).

Fuente canónica: configs/gt/tape_timecodes.yaml
Finding: docs/findings/2026-06-10-gt-tape-timecodes-user.md

Modelo de GT:
  - Las cintas están cada 1 m → posición de la cinta N = (N-1) m.
  - El recorrido es una recta a lo largo de la malla → GT 1-D sobre el eje X
    (la alineación Umeyama de `evo` resuelve la orientación del frame est).
  - Posición interpolada LINEALMENTE entre cruces consecutivos (marcha ~cte
    entre cintas). No se extrapola fuera de [primer cruce, último cruce].
  - video_4 (loop): ida cintas 1→9, vértice de retorno a +1 m del último
    cruce (punto medio temporal entre cinta 9 ida y cinta 9 vuelta), vuelta
    9→1.
  - timestamp = t_segundos × fps = FRAME-INDEX absoluto del SVO — la misma
    convención de `trajectory.txt` (DPVO) y `trajectory_tum_mp4idx.txt`
    (MAC-VO). El skip del run no afecta: `evo` sincroniza por timestamp.
  - Orientación: identidad (el ATE traslacional no la usa).

Uso:
    .venv/bin/python scripts/build_gt_tum.py                # los 4 videos
    .venv/bin/python scripts/build_gt_tum.py --video gym_video_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
GT_YAML = REPO / "configs" / "gt" / "tape_timecodes.yaml"
OUT_DIR = REPO / "results" / "gt"


def crossings(entry: dict) -> list[tuple[float, float]]:
    """Lista [(t_s, pos_m)] ordenada por tiempo para un video del YAML."""
    pts: list[tuple[float, float]] = []
    if "tapes" in entry:
        for tape, t in entry["tapes"].items():
            pts.append((float(t), float(tape) - 1.0))
    else:
        for tape, t in entry["tapes_out"].items():
            pts.append((float(t), float(tape) - 1.0))
        out_last_t, out_last_pos = max(pts)
        ret = sorted((float(t), float(tape) - 1.0)
                     for tape, t in entry["tapes_return"].items())
        # vértice del retorno: +turnaround_extra_m, punto medio temporal
        extra = float(entry.get("turnaround_extra_m", 0.0))
        apex_t = (out_last_t + ret[0][0]) / 2.0
        pts.append((apex_t, out_last_pos + extra))
        pts.extend(ret)
    pts.sort()
    return pts


def build(name: str, entry: dict, out_dir: Path) -> Path:
    fps = float(entry["fps"])
    pts = crossings(entry)
    ts = np.array([t for t, _ in pts])
    xs = np.array([x for _, x in pts])
    f0, f1 = int(np.ceil(ts[0] * fps)), int(np.floor(ts[-1] * fps))
    frames = np.arange(f0, f1 + 1)
    x = np.interp(frames / fps, ts, xs)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}_gt_tum.txt"
    with out.open("w") as fh:
        fh.write(f"# GT TUM {name} — interpolado de configs/gt/tape_timecodes.yaml\n")
        fh.write("# timestamp = frame-index absoluto del SVO (t_s × fps)\n")
        for f, xi in zip(frames, x):
            fh.write(f"{float(f):.1f} {xi:.6f} 0.0 0.0 0.0 0.0 0.0 1.0\n")
    span = xs.max() - xs.min()
    print(f"OK {out.name}: frames [{f0},{f1}] ({len(frames)}), "
          f"recorrido GT {span:.1f} m, cruces {len(pts)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None,
                    help="Nombre del video en el YAML (default: todos).")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    cfg = yaml.safe_load(GT_YAML.read_text())
    names = [args.video] if args.video else list(cfg.keys())
    for name in names:
        build(name, cfg[name], args.out_dir)


if __name__ == "__main__":
    main()
