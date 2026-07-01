#!/usr/bin/env python3
"""Escala absoluta de los runs on/off usando el GT de cintas (brazos rectos).

Para cada video v1/v2/v3 y cada lado ON/OFF:
  - carga trajectory.txt (col0=frame, col1-3=xyz)
  - busca la pose en el frame de cada cinta (= round(t_s*60))
  - distancia euclídea entre cintas COLINEALES del mismo brazo recto vs GT
    (cinta1-cinta3 = 2 m, cinta6-cinta8 = 2 m, cinta6-cinta7 = 1 m)
  - factor de escala = dist_traj / dist_GT  (≈1 => métrico)
  - además: path length y neto en la ventana recortada
"""
import glob, yaml, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT = yaml.safe_load((ROOT / "configs/gt/gym_2026-06-30_giro_timecodes.yaml").read_text())
FPS = GT["meta"]["fps"]

def load_traj(path):
    a = np.loadtxt(path)
    return a[:, 0].astype(int), a[:, 1:4]   # frames, xyz

def pos_at(frames, xyz, fr):
    i = int(np.argmin(np.abs(frames - fr)))
    return xyz[i], int(frames[i])

def run_dir(v, side):
    g = glob.glob(str(ROOT / f"results/2026-06-30__gym_girodemarcado_v{v}_imu_{side}__*"))
    return g[0] if g else None

# pares (cinta_a, cinta_b, dist_GT, etiqueta, fase) — fase: 'ida' o 'vuelta'
ANCHORS = [
    (1, 3, 2.0, "A ida  c1-c3", "ida"),
    (6, 8, 2.0, "B ida  c6-c8", "ida"),
    (6, 7, 1.0, "B vta  c6-c7", "vuelta"),
    (1, 3, 2.0, "A vta  c1-c3", "vuelta"),
]

for v in (1, 2, 3):
    key = f"gym_girodemarcado_v{v}"
    spec = GT[key]
    f0, f1 = spec["trim_frames"]
    print(f"\n{'='*64}\n{key}   recorte=[{f0},{f1}]  ({f0/FPS:.1f}-{f1/FPS:.1f} s)")
    for side in ("ON", "OFF"):
        d = run_dir(v, side)
        if not d:
            print(f"  {side}: (sin run)"); continue
        frames, xyz = load_traj(Path(d) / "trajectory.txt")
        # ventana recortada
        m = (frames >= f0) & (frames <= f1)
        seg = xyz[m]
        path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
        neto = float(np.linalg.norm(seg[-1] - seg[0]))
        ests = []
        rows = []
        for ca, cb, gt_d, lab, fase in ANCHORS:
            ta = spec[fase].get(ca); tb = spec[fase].get(cb)
            if ta is None or tb is None:
                continue
            fa, fb = round(ta * FPS), round(tb * FPS)
            pa, fan = pos_at(frames, xyz, fa)
            pb, fbn = pos_at(frames, xyz, fb)
            dist = float(np.linalg.norm(pb - pa))
            s = dist / gt_d
            ests.append(s)
            rows.append(f"      {lab}: {dist:5.2f} m / {gt_d:.0f} m  -> s={s:.3f}")
        s_med = float(np.median(ests)) if ests else float("nan")
        print(f"  {side:3s}  path={path:6.2f} m  neto={neto:5.2f} m  path/neto={path/max(neto,1e-9):4.1f}"
              f"   escala_med={s_med:.3f}")
        for r in rows:
            print(r)
