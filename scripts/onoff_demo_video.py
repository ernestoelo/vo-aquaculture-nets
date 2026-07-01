#!/usr/bin/env python3
"""onoff_demo_video.py — .mp4 del demo on/off del Hito 3 (paso 2 del handoff
2026-07-01-hito3-rrd-demo-mp4-fernanda-repo): animación matplotlib + ffmpeg,
offline (sin Rerun viewer — en aarch64 no rinde headless y el export de video
del viewer no es robusto).

Mismo contenido y misma matemática que el .rrd (importa la geometría de
scripts/onoff_to_rerun.py — no la re-implementa), en 3 paneles sincronizados:

  [ video original ] [ planta cenital ON/OFF vs GT ] [ rumbo vs giroscopio ]

- video: results/mp4_gt/girodemarcado_v{v}.mp4 (frame#/segundos quemados =
  trazabilidad; posición MP4 == grab_index, SVO contiguo verificado).
- planta: trayectorias crecen en el tiempo, cursor por lado, GT triángulo/V
  con cintas; anclada a (0,0) (§7 CLAUDE.md).
- rumbo: gyro (verdad física) + ON + OFF crecen en el tiempo; eventos del GT
  (giros ida/vuelta, comienzo de vuelta) como líneas verticales.

Uso:
  .venv/bin/python scripts/onoff_demo_video.py                # v1,v2,v3
  .venv/bin/python scripts/onoff_demo_video.py --video 1 --speed 2
Salida: results/video_demo/onoff_girodemarcado_v{v}.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onoff_to_rerun import (  # noqa: E402 — misma geometría que el .rrd
    GT, FPS, ROOT, build_gt, gyro_heading, load_side,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_ON, C_OFF, C_GYRO, C_GT, C_TAPE = "#3c78f0", "#dc3c3c", "black", "0.45", "tab:orange"


def render_video(v, on_suffix, off_suffix, mp4_dir, out_dir, out_fps, speed, dpi):
    import cv2
    spec = GT[f"gym_girodemarcado_v{v}"]
    f0, f1 = spec["trim_frames"]
    gt_poly, gt_tapes = build_gt()

    ft = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/frame_times.csv",
                    delimiter=",", skiprows=1)
    ftimes = {int(r[0]): int(r[1]) for r in ft}
    t0_ns = ftimes[f0]
    imu = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/imu.csv",
                     delimiter=",", skiprows=1)

    fr_on, t_on, x_on, y_on, h_on, s_on, _ = load_side(
        v, on_suffix, spec, gt_tapes, ftimes, t0_ns)
    _, t_off, x_off, y_off, h_off, s_off, _ = load_side(
        v, off_suffix, spec, gt_tapes, ftimes, t0_ns)

    tg, yg = gyro_heading(imu, t0_ns)
    mg = (tg >= ftimes[f0] * 1e-9) & (tg <= ftimes[f1] * 1e-9)
    tg = tg[mg] - t0_ns * 1e-9
    yg = yg[mg]
    if np.sign(yg[np.argmax(np.abs(yg))]) != np.sign(h_on[np.argmax(np.abs(h_on))]):
        yg = -yg

    T = float(min(t_on[-1], t_off[-1]))
    pk = lambda h: h[int(np.argmax(np.abs(h)))]
    g_pk, on_pk, off_pk = pk(yg), pk(h_on), pk(h_off)

    # ---------- figura estática (artistas una vez, update por frame) ----------
    fig = plt.figure(figsize=(17.0, 5.2), dpi=dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.35],
                          left=0.03, right=0.985, top=0.86, bottom=0.11,
                          wspace=0.22)
    ax_v = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    ax_h = fig.add_subplot(gs[2])

    fig.suptitle(
        f"Demo Hito 3 — fusión inercial ON vs OFF   ·   Aire-giros v{v}   ·   "
        f"giro: gyro {g_pk:.0f}°, ON {on_pk:.0f}° (|Δ|={abs(on_pk-g_pk):.0f}°), "
        f"OFF {off_pk:.0f}° (sobre-rota)", fontsize=13)

    # video
    ax_v.set_axis_off()
    ax_v.set_title("Cámara izquierda (frame# quemado)", fontsize=10)
    cap = cv2.VideoCapture(str(mp4_dir / f"girodemarcado_v{v}.mp4"))
    ok, fr0 = cap.read()
    if not ok:
        raise SystemExit(f"v{v}: no pude leer {mp4_dir}/girodemarcado_v{v}.mp4")
    im = ax_v.imshow(fr0[..., ::-1], aspect="equal")
    cap_pos = 1  # próximo índice que devolverá cap.read()

    # planta
    ax_p.set_title(f"Planta cenital [m] (cualitativa; ON÷{s_on:.2f} OFF÷{s_off:.2f})",
                   fontsize=10)
    ax_p.plot(gt_poly[:, 0], gt_poly[:, 1], "--", color=C_GT, lw=1.5,
              label="GT recorrido (cintas 1 m)")
    for t, (txp, typ) in gt_tapes.items():
        ax_p.plot(txp, typ, "o", mfc="none", mec=C_TAPE, ms=8, mew=1.4)
        ax_p.annotate(str(t), (txp, typ), fontsize=6.5, ha="center", va="center",
                      color=C_TAPE)
    ln_pon, = ax_p.plot([], [], "-", color=C_ON, lw=1.6, label="ON (IMU tight)")
    ln_poff, = ax_p.plot([], [], "-", color=C_OFF, lw=1.6, label="OFF (VO pura)")
    cur_on, = ax_p.plot([], [], "o", color=C_ON, ms=8)
    cur_off, = ax_p.plot([], [], "o", color=C_OFF, ms=8)
    ax_p.plot(0, 0, "o", color="tab:green", ms=9, label="inicio (0,0)")
    allx = np.concatenate([gt_poly[:, 0], x_on, x_off])
    ally = np.concatenate([gt_poly[:, 1], y_on, y_off])
    mx, my = 0.08 * (np.ptp(allx) + 1e-9), 0.08 * (np.ptp(ally) + 1e-9)
    ax_p.set_xlim(allx.min() - mx, allx.max() + mx)
    ax_p.set_ylim(ally.min() - my, ally.max() + my)
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.grid(alpha=.3)
    ax_p.legend(loc="upper left", fontsize=7, framealpha=.9)
    ax_p.set_xlabel("X [m]", fontsize=8)
    ax_p.set_ylabel("Y [m]", fontsize=8)

    # rumbo
    ax_h.set_title("Rumbo cámara vs giroscopio [°] — métrica líder", fontsize=10)
    ln_hg, = ax_h.plot([], [], "-", color=C_GYRO, lw=2.2, label="gyro (física)")
    ln_hon, = ax_h.plot([], [], "-", color=C_ON, lw=1.7, label="ON (IMU tight)")
    ln_hoff, = ax_h.plot([], [], "-", color=C_OFF, lw=1.7, label="OFF (VO pura)")
    ax_h.axhline(0, color="gray", lw=.6, alpha=.5)
    for fase, c in (("ida", "tab:orange"), ("vuelta", "tab:purple")):
        tgiro = spec[fase].get("giro")
        if tgiro:
            ax_h.axvline(tgiro - f0 / FPS, color=c, ls="--", alpha=.6, lw=1)
    cv_t = spec.get("comienzo_vuelta")
    if cv_t:
        ax_h.axvline(cv_t - f0 / FPS, color="gray", ls=":", alpha=.7, lw=1)
    allh = np.concatenate([yg, h_on, h_off])
    mh = 0.08 * np.ptp(allh)
    ax_h.set_xlim(0, T)
    ax_h.set_ylim(allh.min() - mh, allh.max() + mh)
    ax_h.grid(alpha=.3)
    ax_h.legend(loc="best", fontsize=7, framealpha=.9)
    ax_h.set_xlabel("t desde inicio [s]", fontsize=8)
    ax_h.set_ylabel("heading [°]", fontsize=8)

    # canvas -> dimensiones pares para yuv420p
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    w -= w % 2
    h -= h % 2

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"onoff_girodemarcado_v{v}.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{out_fps}", "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)

    # ---------- loop de animación ----------
    times = np.arange(0.0, T, speed / out_fps)
    t_wall = time.time()
    for k, tk in enumerate(times):
        # video: frame fuente más cercano en el tiempo (decodificación secuencial)
        i_kf = min(int(np.searchsorted(t_on, tk)), len(fr_on) - 1)
        want = int(fr_on[i_kf])
        while cap_pos <= want:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            cap_pos += 1
        if ok:
            im.set_data(frame_bgr[..., ::-1])

        m_on = t_on <= tk
        m_off = t_off <= tk
        ln_pon.set_data(x_on[m_on], y_on[m_on])
        ln_poff.set_data(x_off[m_off], y_off[m_off])
        if m_on.any():
            cur_on.set_data([x_on[m_on][-1]], [y_on[m_on][-1]])
        if m_off.any():
            cur_off.set_data([x_off[m_off][-1]], [y_off[m_off][-1]])

        m_g = tg <= tk
        ln_hg.set_data(tg[m_g], yg[m_g])
        ln_hon.set_data(t_on[m_on], h_on[m_on])
        ln_hoff.set_data(t_off[m_off], h_off[m_off])

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:h, :w, :3]
        ff.stdin.write(np.ascontiguousarray(buf).tobytes())
        if (k + 1) % 200 == 0:
            print(f"  [v{v}] {k+1}/{len(times)} frames "
                  f"({(k+1)/(time.time()-t_wall):.1f} fps render)", flush=True)

    ff.stdin.close()
    ff.wait()
    cap.release()
    plt.close(fig)
    mb = out.stat().st_size / 1e6
    dur = len(times) / out_fps
    print(f"[v{v}] {out}  ({mb:.1f} MB, {dur:.0f} s @ {out_fps} fps, speed {speed}x, "
          f"{w}x{h})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="all", help="1|2|3|all (default all)")
    p.add_argument("--on", default="imu_siga15",
                   help="sufijo del run ON (default imu_siga15, near-metric)")
    p.add_argument("--off", default="imu_OFF", help="sufijo del run OFF")
    p.add_argument("--mp4-dir", type=Path, default=ROOT / "results/mp4_gt")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/video_demo")
    p.add_argument("--out-fps", type=int, default=20)
    p.add_argument("--speed", type=float, default=1.0,
                   help="factor de velocidad de reproducción (default 1.0 = tiempo real)")
    p.add_argument("--dpi", type=int, default=100)
    args = p.parse_args()

    videos = [1, 2, 3] if args.video == "all" else [int(args.video)]
    made = [render_video(v, args.on, args.off, args.mp4_dir, args.out_dir,
                         args.out_fps, args.speed, args.dpi) for v in videos]
    print("\nListo para enviar:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
