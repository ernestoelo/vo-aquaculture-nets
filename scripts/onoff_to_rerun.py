#!/usr/bin/env python3
"""onoff_to_rerun.py — .rrd del demo on/off del Hito 3 (fusión inercial).

Visualiza, por video del dataset `gym_girodemarcado` (Aire-giros N=3), las tres
piezas que sostienen el demo:

  1. VIDEO original (cámara izquierda): el mp4_gt con frame#/segundos quemados
     (results/mp4_gt/girodemarcado_v{v}.mp4, el mismo que el usuario anotó para
     el GT) sincronizado al timeline — feedback visual de qué ve DPVO en cada
     instante. Posición en el MP4 == grab_index (verificado 2026-07-01: los 3
     SVO son contiguos 0..N-1; el frames_total del yaml sobre-contaba).
  2. PLANTA (vista cenital) anclada al origen: ON (IMU tight, sig_a 15 near-metric)
     vs OFF (VO pura) sobre el GT del recorrido (triángulo / "V" invertida,
     base y altura ~3 m, cintas a 1 m real). Misma proyección/normalización/
     orientación rígida que scripts/make_planta_reframed_girodemarcado.py.
  3. RUMBO vs GIROSCOPIO en el tiempo (la MÉTRICA LÍDER, calificable): el yaw
     acumulado del giroscopio (verdad física) vs el azimut del eje óptico de la
     cámara para ON y OFF. Misma matemática que scripts/eval_heading_vs_gyro_gt.py.
     ON sigue al gyro (~1° en el giro); OFF sobre-rota y deriva.

La planta es cualitativa (la escala ON es near-metric pero no perfecta); la figura
calificable es el rumbo. Se dice explícito en los nombres de las vistas.

Salida: results/rerun_onoff/onoff_girodemarcado_v{v}.rrd (uno por video).

Uso:
  .venv/bin/python scripts/onoff_to_rerun.py                 # v1,v2,v3
  .venv/bin/python scripts/onoff_to_rerun.py --video 1       # solo v1
  .venv/bin/python scripts/onoff_to_rerun.py --on imu_siga15 --off imu_OFF
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import yaml

try:
    import rerun as rr
except ModuleNotFoundError:
    raise SystemExit(
        "Falta 'rerun-sdk' (import rerun) en este venv.\n"
        "Usar el venv del repo:  .venv/bin/python scripts/onoff_to_rerun.py\n"
        "o instalar:             pip install 'rerun-sdk==0.21.0'"
    )

ROOT = Path(__file__).resolve().parent.parent
GT = yaml.safe_load((ROOT / "configs/gt/gym_2026-06-30_giro_timecodes.yaml").read_text())
FPS = GT["meta"]["fps"]
TAPE_POS = GT["meta"]["tape_positions_m"]

# Anclas de escala (pares colineales del mismo brazo recto) — igual que la planta
ANCHORS = [(1, 3, 2.0, "ida"), (6, 8, 2.0, "ida"),
           (6, 7, 1.0, "vuelta"), (1, 3, 2.0, "vuelta")]

# Colores (RGB)
C_ON = [60, 120, 240]
C_OFF = [220, 60, 60]
C_GYRO = [20, 20, 20]
C_GT = [120, 120, 120]
C_TAPE = [230, 140, 30]

IMU_RATE_HZ = 406.0  # tasa de la IMU en el gym (gate del dataset)


# --------------------------------------------------------------------------
# Geometría (portado de make_planta_reframed_girodemarcado.py)
# --------------------------------------------------------------------------
def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def plane_normal(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    return Vt[2]


def project_plant(P, Q):
    n = plane_normal(P)
    up_cam = np.array([quat_to_R(q) @ np.array([0., -1., 0.]) for q in Q]).mean(0)
    if n @ up_cam < 0:
        n = -n
    k = min(len(P) - 1, int(1.5 * FPS))
    mot = P[k] - P[0]
    fwd = mot - (mot @ n) * n
    if np.linalg.norm(fwd) < 1e-6:
        z0 = quat_to_R(Q[0]) @ np.array([0., 0., 1.])
        fwd = z0 - (z0 @ n) * n
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, n)
    d = P - P[0]
    return d @ right, d @ fwd


def median_arm_scale(fr, x, y, spec):
    ss = []
    for ca, cb, gt_d, fase in ANCHORS:
        ta = spec[fase].get(ca)
        tb = spec[fase].get(cb)
        if ta is None or tb is None:
            continue
        ia = int(np.argmin(np.abs(fr - round(ta * FPS))))
        ib = int(np.argmin(np.abs(fr - round(tb * FPS))))
        ss.append(np.hypot(x[ib] - x[ia], y[ib] - y[ia]) / gt_d)
    return float(np.median(ss)) if ss else 1.0


def at_frame(fr, x, y, t_s):
    i = int(np.argmin(np.abs(fr - round(t_s * FPS))))
    return x[i], y[i]


# GT del recorrido = triángulo / "V" invertida (idéntico a la planta reencuadrada)
BASE_M = 3.0
HEIGHT_M = 3.0
_S_AP = 0.5 * (TAPE_POS[4] + TAPE_POS[5])
_S_MAX = max(TAPE_POS.values())
_INICIO = np.array([-BASE_M / 2, 0.0])
_APEX = np.array([0.0, HEIGHT_M])
_END = np.array([BASE_M / 2, 0.0])


def gt_pos(s):
    if s <= _S_AP:
        return _INICIO + (s / _S_AP) * (_APEX - _INICIO)
    return _APEX + ((s - _S_AP) / (_S_MAX - _S_AP)) * (_END - _APEX)


def build_gt():
    off = gt_pos(TAPE_POS[1])
    ss = np.linspace(0, max(TAPE_POS.values()), 100)
    poly = np.array([gt_pos(s) - off for s in ss])
    tapes = {t: gt_pos(TAPE_POS[t]) - off for t in TAPE_POS}
    return poly, tapes


def align_to_gt(fr, x, y, spec, gt_tapes):
    """Orientación rígida (rotación + posible reflexión, sin traslación) sobre las
    cintas de la ida — igual que make_planta_reframed_girodemarcado.py."""
    est, gt = [], []
    for c in spec["ida"]:
        if not isinstance(c, int) or c not in gt_tapes:
            continue
        cx, cy = at_frame(fr, x, y, spec["ida"][c])
        est.append([cx, cy])
        gt.append(gt_tapes[c])
    E, G = np.array(est), np.array(gt)
    U, _, Vt = np.linalg.svd(E.T @ G)
    W = U @ Vt
    return x * W[0, 0] + y * W[1, 0], x * W[0, 1] + y * W[1, 1]


# --------------------------------------------------------------------------
# Heading (portado de eval_heading_vs_gyro_gt.py)
# --------------------------------------------------------------------------
def gyro_heading(imu, t0_ns):
    ts = imu[:, 0] * 1e-9
    acc = imu[:, 1:4]
    gyr = imu[:, 4:7]
    w = max(1, int(0.5 * IMU_RATE_HZ))
    k = np.ones(w) / w
    g = np.stack([np.convolve(acc[:, i], k, "same") for i in range(3)], 1)
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    yaw_rate = np.sum(gyr * g, axis=1)
    dt = np.gradient(ts)
    yaw = np.degrees(np.cumsum(yaw_rate * dt))
    yaw -= np.interp(t0_ns * 1e-9, ts, yaw)
    return ts, yaw


def traj_heading(P, Q):
    """Azimut del eje óptico (+Z) sobre el plano del piso; cero en la 1a pose."""
    n = plane_normal(P)
    z0 = quat_to_R(Q[0]) @ np.array([0, 0, 1.0])
    e1 = z0 - (z0 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    head = []
    for q in Q:
        zc = quat_to_R(q) @ np.array([0, 0, 1.0])
        zp = zc - (zc @ n) * n
        head.append(np.degrees(np.arctan2(zp @ e2, zp @ e1)))
    head = np.degrees(np.unwrap(np.radians(head)))
    head -= head[0]
    return np.asarray(head)


# --------------------------------------------------------------------------
def run_dir(v, suffix):
    g = glob.glob(str(ROOT / f"results/*__gym_girodemarcado_v{v}_{suffix}__*"))
    return max(g, key=os.path.getmtime) if g else None


def load_side(v, suffix, spec, gt_tapes, ftimes, t0_ns):
    """Devuelve (fr, t_s, x, y, head, s) alineados por índice de keyframe recortado."""
    d = run_dir(v, suffix)
    if d is None:
        raise SystemExit(f"v{v}: sin run para sufijo '{suffix}'")
    a = np.loadtxt(Path(d) / "trajectory.txt")
    fr = a[:, 0].astype(int)
    f0, f1 = spec["trim_frames"]
    m = (fr >= f0) & (fr <= f1)
    fr = fr[m]
    P = a[m, 1:4]
    Q = a[m, 4:8]
    x, y = project_plant(P, Q)
    s = median_arm_scale(fr, x, y, spec)
    x, y = x / s, y / s
    x, y = align_to_gt(fr, x, y, spec, gt_tapes)
    head = traj_heading(P, Q)
    t_s = np.array([ftimes[i] for i in fr]) * 1e-9 - t0_ns * 1e-9
    return fr, t_s, x, y, head, s, Path(d).name


def xy2(x, y):
    """(x,y) planta -> arreglo (N,2) con Y hacia ARRIBA (rerun 2D es y-down)."""
    return np.column_stack([np.asarray(x), -np.asarray(y)])


def make_rrd(v, on_suffix, off_suffix, out_dir, gyro_every, mp4_dir,
             img_every, img_max_width, jpeg_quality):
    spec = GT[f"gym_girodemarcado_v{v}"]
    f0, f1 = spec["trim_frames"]
    gt_poly, gt_tapes = build_gt()

    ft = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/frame_times.csv",
                    delimiter=",", skiprows=1)
    ftimes = {int(r[0]): int(r[1]) for r in ft}
    t0_ns = ftimes[f0]
    imu = np.loadtxt(ROOT / f"results/imu/gym_girodemarcado_v{v}/imu.csv",
                     delimiter=",", skiprows=1)

    fr_on, t_on, x_on, y_on, h_on, s_on, name_on = load_side(v, on_suffix, spec, gt_tapes, ftimes, t0_ns)
    _, t_off, x_off, y_off, h_off, s_off, name_off = load_side(v, off_suffix, spec, gt_tapes, ftimes, t0_ns)

    # gyro (verdad física) — alinear signo con ON
    tg, yg = gyro_heading(imu, t0_ns)
    mg = (tg >= ftimes[f0] * 1e-9) & (tg <= ftimes[f1] * 1e-9)
    tg = tg[mg] - t0_ns * 1e-9
    yg = yg[mg]
    if np.sign(yg[np.argmax(np.abs(yg))]) != np.sign(h_on[np.argmax(np.abs(h_on))]):
        yg = -yg
    if gyro_every > 1:
        tg = tg[::gyro_every]
        yg = yg[::gyro_every]

    # ---- Rerun ----
    out_dir.mkdir(parents=True, exist_ok=True)
    rrd = out_dir / f"onoff_girodemarcado_v{v}.rrd"
    rr.init(f"onoff_girodemarcado_v{v}", spawn=False)

    # Blueprint: video (izq) + planta (centro) + rumbo (der). Si falla, seguimos.
    mp4 = mp4_dir / f"girodemarcado_v{v}.mp4" if mp4_dir else None
    have_mp4 = mp4 is not None and mp4.is_file()
    if mp4 is not None and not have_mp4:
        print(f"[onoff->rerun] WARN: sin MP4 {mp4} — .rrd sin panel de video")
    try:
        import rerun.blueprint as rrb
        views = [
            rrb.Spatial2DView(origin="/planta",
                              name="Planta cenital [m] (cualitativa)"),
            rrb.TimeSeriesView(origin="/heading",
                               name="Rumbo camara vs giroscopio [deg] (metrica lider)"),
        ]
        shares = [1.0, 1.3]
        if have_mp4:
            views.insert(0, rrb.Spatial2DView(origin="/video",
                                              name="Camara izquierda (frame# quemado)"))
            shares = [1.2, 1.0, 1.3]
        rr.send_blueprint(rrb.Blueprint(
            rrb.Horizontal(*views, column_shares=shares),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed"),
        ))
    except Exception as e:  # noqa: BLE001
        print(f"[onoff->rerun] blueprint omitido ({e})")

    rr.save(str(rrd))

    # --- PLANTA: contenido estático (se ve todo el tiempo) ---
    rr.log("planta/gt", rr.LineStrips2D([xy2(gt_poly[:, 0], gt_poly[:, 1])],
                                        colors=[C_GT], radii=0.012),
           static=True)
    tape_xy = np.array([gt_tapes[t] for t in sorted(gt_tapes)])
    rr.log("planta/gt_tapes",
           rr.Points2D(xy2(tape_xy[:, 0], tape_xy[:, 1]),
                       colors=[C_TAPE], radii=0.06,
                       labels=[str(t) for t in sorted(gt_tapes)]),
           static=True)
    rr.log("planta/ON", rr.LineStrips2D([xy2(x_on, y_on)], colors=[C_ON], radii=0.02),
           static=True)
    rr.log("planta/OFF", rr.LineStrips2D([xy2(x_off, y_off)], colors=[C_OFF], radii=0.02),
           static=True)
    # inicio comun (0,0) y fines
    rr.log("planta/inicio", rr.Points2D([[0.0, 0.0]], colors=[[40, 180, 80]], radii=0.09,
                                        labels=["inicio"]), static=True)

    # --- HEADING: apariencia de las series (estático) ---
    rr.log("heading/gyro", rr.SeriesLine(color=C_GYRO, width=2.5, name="gyro (fisica)"),
           static=True)
    rr.log("heading/ON", rr.SeriesLine(color=C_ON, width=2.0, name="ON (IMU tight)"),
           static=True)
    rr.log("heading/OFF", rr.SeriesLine(color=C_OFF, width=2.0, name="OFF (VO pura)"),
           static=True)

    # --- VIDEO: frames de la cámara izquierda sincronizados al timeline ---
    # Posición en el MP4 == grab_index (SVO contiguo, verificado). Decodificamos
    # secuencial (mucho más rápido que seek) y logueamos 1 de cada img_every
    # keyframes del tramo recortado.
    if have_mp4:
        import cv2
        wanted = {int(f): float(t) for f, t in zip(fr_on[::img_every],
                                                   t_on[::img_every])}
        cap = cv2.VideoCapture(str(mp4))
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if img_max_width and w_in > img_max_width:
            sc = img_max_width / w_in
            wh = (img_max_width, int(round(h_in * sc / 2)) * 2)
        else:
            wh = (w_in, h_in)
        n_img = 0
        idx = 0
        last = max(wanted)
        while idx <= last:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx in wanted:
                if (frame_bgr.shape[1], frame_bgr.shape[0]) != wh:
                    frame_bgr = cv2.resize(frame_bgr, wh, interpolation=cv2.INTER_AREA)
                rr.set_time_seconds("t", wanted[idx])
                rr.log("video/frame",
                       rr.Image(frame_bgr[..., ::-1]).compress(jpeg_quality=jpeg_quality))
                n_img += 1
            idx += 1
        cap.release()
        print(f"[v{v}] video: {n_img} frames embebidos @ {wh[0]}x{wh[1]} "
              f"(1 de cada {img_every}, jpeg q{jpeg_quality}) desde {mp4.name}")

    # gyro: serie densa (verdad física) sobre su propio reloj
    for t, y in zip(tg, yg):
        rr.set_time_seconds("t", float(t))
        rr.log("heading/gyro", rr.Scalar(float(y)))

    # ON / OFF: cursor de posición en la planta + escalar de rumbo, por keyframe
    def stream_side(t_s, x, y, head, path_entity):
        cur_col = C_ON if path_entity.endswith("ON") else C_OFF
        for i in range(len(t_s)):
            rr.set_time_seconds("t", float(t_s[i]))
            rr.log(f"planta/{path_entity}_cursor",
                   rr.Points2D(xy2([x[i]], [y[i]]), colors=[cur_col], radii=0.07))
            rr.log(f"heading/{path_entity}", rr.Scalar(float(head[i])))

    stream_side(t_on, x_on, y_on, h_on, "ON")
    stream_side(t_off, x_off, y_off, h_off, "OFF")

    size_mb = rrd.stat().st_size / 1e6

    def pk(t, h):
        i = int(np.argmax(np.abs(h)))
        return h[i], h[-1]
    g_pk, g_fin = pk(tg, yg)
    on_pk, on_fin = pk(t_on, h_on)
    off_pk, off_fin = pk(t_off, h_off)
    print(f"[v{v}] ON={name_on}  OFF={name_off}")
    print(f"[v{v}] escala-mediana  ON÷{s_on:.2f}  OFF÷{s_off:.2f}")
    print(f"[v{v}] heading pico/fin  gyro {g_pk:.0f}/{g_fin:.0f}  "
          f"ON {on_pk:.0f}/{on_fin:.0f} (|Δpico|={abs(on_pk - g_pk):.0f})  "
          f"OFF {off_pk:.0f}/{off_fin:.0f}")
    print(f"[v{v}] guardado: {rrd} ({size_mb:.1f} MB)\n")
    return rrd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="all", help="1|2|3|all (default all)")
    p.add_argument("--on", default="imu_siga15",
                   help="sufijo del run ON (default imu_siga15, near-metric)")
    p.add_argument("--off", default="imu_OFF", help="sufijo del run OFF (default imu_OFF)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/rerun_onoff")
    p.add_argument("--gyro-every", type=int, default=4,
                   help="subsamplea el gyro 1/N (default 4 ~100 Hz) para achicar el .rrd")
    p.add_argument("--mp4-dir", type=Path, default=ROOT / "results/mp4_gt",
                   help="carpeta con girodemarcado_v{v}.mp4 (frame# quemado) para el "
                        "panel de video; pasar '' para .rrd sin video")
    p.add_argument("--img-every", type=int, default=3,
                   help="embebe 1 frame de video cada N keyframes (default 3 ~20 fps)")
    p.add_argument("--img-max-width", type=int, default=640,
                   help="ancho máx de la imagen embebida (default 640; 0=nativa)")
    p.add_argument("--jpeg-quality", type=int, default=70,
                   help="calidad JPEG de la imagen embebida (default 70)")
    args = p.parse_args()

    mp4_dir = args.mp4_dir if str(args.mp4_dir) else None
    videos = [1, 2, 3] if args.video == "all" else [int(args.video)]
    made = []
    for v in videos:
        made.append(make_rrd(v, args.on, args.off, args.out_dir, args.gyro_every,
                             mp4_dir, args.img_every, args.img_max_width,
                             args.jpeg_quality))
    print("Ver:  rerun " + " ".join(str(m) for m in made))


if __name__ == "__main__":
    main()
