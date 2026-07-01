#!/usr/bin/env python3
"""Heading-vs-gyro recortado, con los timestamps de giro del GT superpuestos.

- gyro (verdad física): yaw acumulado = integral de (omega . g_hat) dt,
  con g_hat = direccion de gravedad (accel pasa-bajos) por muestra. Robusto a
  inclinacion: proyecta el giro sobre la vertical medida instante a instante.
- trayectoria ON/OFF: azimut del eje optico (+Z) de la camara sobre el plano
  del piso (normal = mejor plano ajustado a las posiciones recortadas).
- Todo relativo al frame 'inicio' del GT (heading=0). Comparten reloj via
  frame_times.csv (grab_index -> ts_ns).
"""
import glob, json, yaml, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
GT = yaml.safe_load((ROOT / "configs/gt/gym_2026-06-30_giro_timecodes.yaml").read_text())
FPS = GT["meta"]["fps"]

def quat_to_R(q):  # q = [qx,qy,qz,qw]
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def run_dir(v, side):
    g = glob.glob(str(ROOT / f"results/2026-06-30__gym_girodemarcado_v{v}_imu_{side}__*"))
    return g[0] if g else None

def plane_normal(P):
    c = P.mean(0); U, S, Vt = np.linalg.svd(P - c)
    return Vt[2]  # menor varianza = normal del plano

def gyro_heading(imu, t0_ns):
    ts = imu[:, 0] * 1e-9
    acc = imu[:, 1:4]; gyr = imu[:, 4:7]
    # gravedad por pasa-bajos (ventana ~0.5 s) y normalizada
    w = max(1, int(0.5 * 406))
    k = np.ones(w) / w
    g = np.stack([np.convolve(acc[:, i], k, "same") for i in range(3)], 1)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    yaw_rate = np.sum(gyr * g, axis=1)         # omega . vertical
    dt = np.gradient(ts)
    yaw = np.cumsum(yaw_rate * dt)
    yaw = np.degrees(yaw)
    yaw -= np.interp(t0_ns * 1e-9, ts, yaw)    # cero en 'inicio'
    return ts, yaw

def traj_heading(traj_xyzq, ftimes, t0_ns, f0, f1):
    fr = traj_xyzq[:, 0].astype(int)
    m = (fr >= f0) & (fr <= f1)
    fr = fr[m]; P = traj_xyzq[m, 1:4]; Q = traj_xyzq[m, 4:8]
    n = plane_normal(P)
    # eje horizontal de referencia (proyeccion del +Z de la 1a camara)
    z0 = quat_to_R(Q[0]) @ np.array([0, 0, 1.0])
    e1 = z0 - (z0 @ n) * n; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    head = []
    for q in Q:
        zc = quat_to_R(q) @ np.array([0, 0, 1.0])
        zp = zc - (zc @ n) * n
        head.append(np.degrees(np.arctan2(zp @ e2, zp @ e1)))
    head = np.unwrap(np.radians(head)); head = np.degrees(head)
    head -= head[0]
    # timeline en ns via frame_times
    ts_fr = np.array([ftimes[i] for i in fr]) * 1e-9
    return ts_fr, head

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
summary = []
for ax, v in zip(axes, (1, 2, 3)):
    key = f"gym_girodemarcado_v{v}"
    spec = GT[key]; f0, f1 = spec["trim_frames"]
    ft = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/frame_times.csv",
                    delimiter=",", skiprows=1)
    ftimes = {int(r[0]): int(r[1]) for r in ft}
    t0_ns = ftimes[f0]                      # ns del frame 'inicio'
    imu = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/imu.csv",
                     delimiter=",", skiprows=1)
    tg, yg = gyro_heading(imu, t0_ns)
    mg = (tg >= ftimes[f0]*1e-9) & (tg <= ftimes[f1]*1e-9)
    # alinear signo del gyro con ON (handedness de frames)
    th_on = np.loadtxt(run_dir(v, "ON") + "/trajectory.txt")
    ton, hon = traj_heading(th_on, ftimes, t0_ns, f0, f1)
    if np.sign(yg[mg][np.argmax(np.abs(yg[mg]))]) != np.sign(hon[np.argmax(np.abs(hon))]):
        yg = -yg
    th_off = np.loadtxt(run_dir(v, "OFF") + "/trajectory.txt")
    toff, hoff = traj_heading(th_off, ftimes, t0_ns, f0, f1)

    t00 = t0_ns * 1e-9
    # meseta de heading sostenido (entre el giro de ida y el de vuelta): la
    # cámara mantiene rumbo girado mientras recorre el brazo B + el about-face
    gi = spec["ida"].get("giro"); gv = spec["vuelta"].get("giro")
    if gi and gv:
        ax.axvspan(gi - f0/FPS, gv - f0/FPS, color="tab:olive", alpha=.06, zorder=0)
    ax.plot(tg[mg]-t00, yg[mg], "k-", lw=2, label="gyro (física)")
    ax.plot(ton-t00, hon, "C0-", lw=1.6, label="ON (IMU tight)")
    ax.plot(toff-t00, hoff, "C3-", lw=1.6, label="OFF (VO pura)")
    ax.axhline(0, color="gray", lw=.6, alpha=.5)
    # marcar giros y comienzo de vuelta del GT
    for fase, c in (("ida", "tab:orange"), ("vuelta", "tab:purple")):
        tg_s = spec[fase].get("giro")
        if tg_s: ax.axvline(tg_s - f0/FPS, color=c, ls="--", alpha=.7)
    cv = spec.get("comienzo_vuelta")
    if cv: ax.axvline(cv - f0/FPS, color="gray", ls=":", alpha=.8)

    def peak_final(t, h, lo, hi):
        mm = (t-t00 >= lo) & (t-t00 <= hi)
        hh = h[mm] if mm.any() else h
        tt = (t-t00)[mm] if mm.any() else (t-t00)
        ip = np.argmax(np.abs(hh))
        return hh[ip], tt[ip], hh[-1]
    span = (0, (f1-f0)/FPS)
    pg, ptg, fg = peak_final(tg[mg], yg[mg], *span)
    po, pto, fo = peak_final(ton, hon, *span)
    pf, ptf, ff = peak_final(toff, hoff, *span)
    summary.append((v, pg, fg, po, fo, pf, ff))
    # anotar el acuerdo ON≈gyro en el pico del giro y la sobre-rotación de OFF
    ax.annotate(f"ON {po:.0f}° ≈ gyro {pg:.0f}°\n|Δ| = {abs(po-pg):.0f}°",
                xy=(ptg, pg), xytext=(0.30, 0.62 if pg > 0 else 0.30),
                textcoords="axes fraction", fontsize=8.5, color="C0",
                ha="center", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="C0", alpha=.9),
                arrowprops=dict(arrowstyle="->", color="C0", lw=1.2))
    ax.annotate(f"OFF {pf:.0f}°\n(sobre-rota)",
                xy=(ptf, pf), xytext=(0.72, 0.18 if pf < 0 else 0.84),
                textcoords="axes fraction", fontsize=8.5, color="C3",
                ha="center", va="center",
                bbox=dict(boxstyle="round", fc="white", ec="C3", alpha=.9),
                arrowprops=dict(arrowstyle="->", color="C3", lw=1.2))
    ax.set_title(f"v{v}  recorte {f0/FPS:.0f}-{f1/FPS:.0f} s")
    ax.set_xlabel("t desde inicio [s]"); ax.grid(alpha=.3)
axes[0].set_ylabel("heading [°]  (0 en 'inicio')")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Rumbo de la cámara vs giroscopio (Aire-giros, N=3)", fontsize=12)
fig.tight_layout()
out = ROOT / "results/figures/heading_gt_girodemarcado_2026-06-30.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"figura: {out}\n")
print(f"{'v':>2} | {'gyro pk/fin':>14} | {'ON pk/fin':>14} | {'OFF pk/fin':>14}")
for v, pg, fg, po, fo, pf, ff in summary:
    print(f"{v:>2} | {pg:6.0f}/{fg:6.0f}   | {po:6.0f}/{fo:6.0f}   | {pf:6.0f}/{ff:6.0f}")
