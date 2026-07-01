#!/usr/bin/env python3
"""Hipótesis A (desacoplar heading↔escala down-pesando la traslación de la IMU):
compara OFF / ON(sig_a 0.2) / siga2 / siga10 / siga1000 en v1/v2/v3 por
escala-mediana-de-brazos (vs GT cintas) y pico de heading vs gyro.

¿Subir --imu-sig-a mantiene el heading (viene de r_R/gyro) y deja de inflar la
escala (r_v/r_p down-pesados → el depth-prior manda)? Date-agnóstico: toma el run
más reciente por (v, suffix). Reutiliza la métrica de eval_imu_s5_check.py.
"""
import glob, os, yaml, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT = yaml.safe_load((ROOT / "configs/gt/gym_2026-06-30_giro_timecodes.yaml").read_text())
FPS = GT["meta"]["fps"]
# ancla OFF, ancla ON (=strength 10, sig_a default 0.2), y el barrido de sig_a
RUNS = [("OFF", "imu_OFF"), ("ON(sa.2)", "imu_ON"),
        ("siga2", "imu_siga2"), ("siga10", "imu_siga10"), ("siga1k", "imu_siga1000")]
ANCHORS = [(1, 3, 2.0, "ida"), (6, 8, 2.0, "ida"), (6, 7, 1.0, "vuelta"), (1, 3, 2.0, "vuelta")]


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def plane_normal(P):
    c = P.mean(0); _, _, Vt = np.linalg.svd(P - c); return Vt[2]


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
    yw = yaw[m]; return yw[np.argmax(np.abs(yw))], ftimes


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
        ia = int(np.argmin(np.abs(fr-round(ta*FPS)))); ib = int(np.argmin(np.abs(fr-round(tb*FPS))))
        ss.append(float(np.linalg.norm(xyz[ib]-xyz[ia])/gt_d))
    return float(np.median(ss)), float(max(ss)-min(ss)), ss


print(f"{'v':>2} {'run':>9} | {'esc.med':>7} {'spread':>6} | {'head pk':>7} {'gyro':>6} {'|Δ|':>4}")
print("-"*56)
for v in (1, 2, 3):
    spec = GT[f"gym_girodemarcado_v{v}"]; f0, f1 = spec["trim_frames"]
    gpk, _ = gyro_peak(v, f0, f1)
    for label, suffix in RUNS:
        d = run_dir(v, suffix)
        if not d:
            print(f"{v:>2} {label:>9} | (sin run)"); continue
        a = np.loadtxt(Path(d) / "trajectory.txt")
        smed, spread, _ = scale_arms(a, spec, f0, f1)
        hpk = traj_head_peak(a, f0, f1)
        print(f"{v:>2} {label:>9} | {smed:7.2f} {spread:6.2f} | {hpk:7.0f} {gpk:6.0f} "
              f"{abs(abs(hpk)-abs(gpk)):4.0f}")
    print("-"*56)
