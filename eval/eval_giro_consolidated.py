#!/usr/bin/env python3
"""Benchmark consolidado del demo IMU on/off sobre Aire-giros (gym girodemarcado).

Reproduce los Cuadros VII/VIII/IX del informe Hito 3: para cada video v1/v2/v3
y cada config canónica (OFF = VO pura métrica, ON = fusión tight con
--imu-strength 10 --imu-sig-a 15) calcula, dentro de la ventana útil del GT
(trim_frames del yaml):

  - N poses, camino integrado (sum ||dP||) y neto ||fin-inicio|| (= cierre;
    el recorrido real es ~16 m y vuelve a la cinta 1, neto GT ~ 0)
  - escala mediana de brazos est/GT + spread (convención de
    eval_imu_siga_sweep.py: dist estimada entre cintas colineales / dist real)
  - heading pico vs giroscopio (|Δ| de magnitudes)
  - fps efectivos y ms/frame p50 del stats.json

Convenciones idénticas a eval_imu_siga_sweep.py / eval_scale_arms_gt.py.
GT: configs/gt/gym_2026-06-30_giro_timecodes.yaml. Runs esperados:
results/*__gym_girodemarcado_v{1,2,3}_imu_{OFF,siga15}__* (el más reciente).
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
GT = yaml.safe_load((ROOT / "configs/gt/gym_2026-06-30_giro_timecodes.yaml").read_text())
FPS = GT["meta"]["fps"]
ANCHORS = [(1, 3, 2.0, "ida"), (6, 8, 2.0, "ida"), (6, 7, 1.0, "vuelta"), (1, 3, 2.0, "vuelta")]
# OFF = VO pura métrica; siga15 = punto de operación (strength 10, sig_a 15)
RUNS = [("OFF", "imu_OFF"), ("ON s15", "imu_siga15")]


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def plane_normal(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    return Vt[2]


def run_dir(v, suffix):
    g = glob.glob(str(ROOT / f"results/*__gym_girodemarcado_v{v}_{suffix}__*"))
    return max(g, key=os.path.getmtime) if g else None


def gyro_peak(v, f0, f1):
    ft = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/frame_times.csv",
                    delimiter=",", skiprows=1)
    ftimes = {int(r[0]): int(r[1]) for r in ft}
    imu = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/imu.csv",
                     delimiter=",", skiprows=1)
    ts = imu[:, 0]*1e-9; acc = imu[:, 1:4]; gyr = imu[:, 4:7]
    w = max(1, int(0.5*406)); k = np.ones(w)/w
    g = np.stack([np.convolve(acc[:, i], k, "same") for i in range(3)], 1)
    g /= np.linalg.norm(g, axis=1, keepdims=True)+1e-9
    yaw = np.degrees(np.cumsum(np.sum(gyr*g, axis=1)*np.gradient(ts)))
    yaw -= np.interp(ftimes[f0]*1e-9, ts, yaw)
    m = (ts >= ftimes[f0]*1e-9) & (ts <= ftimes[f1]*1e-9)
    yw = yaw[m]
    return yw[np.argmax(np.abs(yw))]


def traj_head_peak(a, f0, f1):
    fr = a[:, 0].astype(int); m = (fr >= f0) & (fr <= f1)
    P = a[m, 1:4]; Q = a[m, 4:8]
    n = plane_normal(P)
    z0 = quat_to_R(Q[0]) @ np.array([0, 0, 1.0])
    e1 = z0-(z0@n)*n; e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    h = []
    for q in Q:
        zc = quat_to_R(q) @ np.array([0, 0, 1.0]); zp = zc-(zc@n)*n
        h.append(np.degrees(np.arctan2(zp@e2, zp@e1)))
    h = np.degrees(np.unwrap(np.radians(h))); h -= h[0]
    return h[np.argmax(np.abs(h))]


def scale_arms(a, spec, f0, f1):
    fr = a[:, 0].astype(int); xyz = a[:, 1:4]
    ss = []
    for ca, cb, gt_d, fase in ANCHORS:
        ta = spec[fase].get(ca); tb = spec[fase].get(cb)
        if ta is None or tb is None:
            continue
        ia = int(np.argmin(np.abs(fr-round(ta*FPS))))
        ib = int(np.argmin(np.abs(fr-round(tb*FPS))))
        ss.append(float(np.linalg.norm(xyz[ib]-xyz[ia])/gt_d))
    return float(np.median(ss)), float(max(ss)-min(ss))


def main():
    gt_path_m = 16.0  # recorrido real ~16 m (meta.gt_total_path_m [15,17])
    hdr = (f"{'v':>2} {'config':>7} | {'N':>5} {'camino':>7} {'neto':>6} {'deriva%':>7} | "
           f"{'esc.med':>7} {'spread':>6} {'|Δ|°':>5} | {'fps':>6} {'p50ms':>6} {'frames':>6}")
    print(hdr)
    print("-" * len(hdr))
    for v in (1, 2, 3):
        spec = GT[f"gym_girodemarcado_v{v}"]; f0, f1 = spec["trim_frames"]
        gpk = gyro_peak(v, f0, f1)
        for label, suffix in RUNS:
            d = run_dir(v, suffix)
            if not d:
                print(f"{v:>2} {label:>7} | (sin run results/*_v{v}_{suffix}__*)")
                continue
            a = np.loadtxt(Path(d) / "trajectory.txt")
            fr = a[:, 0].astype(int); m = (fr >= f0) & (fr <= f1)
            seg = a[m, 1:4]
            path = float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
            neto = float(np.linalg.norm(seg[-1] - seg[0]))
            smed, spread = scale_arms(a, spec, f0, f1)
            hpk = traj_head_peak(a, f0, f1)
            st = json.load(open(Path(d) / "stats.json"))
            print(f"{v:>2} {label:>7} | {int(m.sum()):>5} {path:7.2f} {neto:6.2f} "
                  f"{100*neto/gt_path_m:7.1f} | {smed:7.2f} {spread:6.2f} "
                  f"{abs(abs(hpk)-abs(gpk)):5.0f} | {st['fps_effective']:6.2f} "
                  f"{st['ms_per_frame_p50']:6.1f} {st['n_frames_processed']:>6}")
        print("-" * len(hdr))


if __name__ == "__main__":
    main()
