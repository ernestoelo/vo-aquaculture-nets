#!/usr/bin/env python3
"""Extrae el stream IMU high-freq + el mapa frame->timestamp de un SVO2 ZED 2i.

Insumo para el EKF loose-coupling del Hito 3 (fusión inercial). Resuelve dos
cosas que el playback ingenuo NO da:

1. **IMU high-freq real (~400 Hz)**. `get_sensors_data(TIME_REFERENCE.IMAGE)`
   entrega 1 sola muestra por frame de imagen (~30 Hz) — la más cercana al
   frame. El stream completo se recupera **polleando con
   `TIME_REFERENCE.CURRENT`** varias veces entre cada `grab()` y deduplicando
   por timestamp: el SDK expone ~13.5 muestras/frame = ~400 Hz medio sobre
   los gym SVO2 (validado 2026-06-19; el polling satura a ~40 polls/frame).

2. **Mapa frame->timestamp real**. La trayectoria TUM de
   `run_sdpvo_metric.py` usa el **índice de grab 0-based** como "timestamp"
   (skip descarta los primeros `skip` grabs; luego frame_no = índice de grab).
   Aquí se registra (grab_index, image_ts_ns) para CADA grab, de modo que el
   EKF pueda mapear cada pose DPVO `frame_no` a su tiempo real y alinearla con
   la IMU. Los timestamps de imagen e IMU comparten el reloj del SVO.

Convención de ejes IMU: la entregada por el ZED SDK (acc en m/s², gyro en
rad/s). La extrínseca cámara<-IMU del SDK se guarda en meta.json (sin tocar
calibración — restricción §6 del workspace; solo se LEE el valor de fábrica).

Salida en <out_dir>/:
  - imu.csv          ts_ns, ax, ay, az, wx, wy, wz   (ordenado, único por ts)
  - frame_times.csv  grab_index, ts_ns               (0-based, todos los grabs)
  - meta.json        extrínseca cam<-IMU 4x4, stats de tasa IMU, gravedad en la
                     ventana MÁS QUIETA, actividad de gyro y veredicto del gate

Gate de validez (impreso al final + en meta.json["gate"]) — endurecido
2026-06-29 tras los 2 caveats del dataset EN AIRE del Hito 3
(docs/findings/2026-06-29-home-air-dataset-recorded-gate-caveats.md):

  1. **Criterio por TASA, no por samples/frame.** `samples_per_frame` es
     `406 Hz ÷ fps` → ~13.6@30fps, ~6.8@60fps, ~4.1@100fps: a 60/100 fps
     queda bajo cualquier umbral fijo y eso NO es degradación. El gate
     decide por `rate_mean_hz` (IMU viva ≈406 Hz vs ~15 Hz muertos del
     underwater); `samples_per_frame` queda solo informativo.
  2. **Gravedad en la ventana MÁS QUIETA, no en la inicial.** La ventana
     inicial fija puede pillar la cámara ya en movimiento → `|g|` falso
     (vimos 8.44–9.55 en tomas válidas). Se barre el SVO en ventanas de
     ~1 s y se mide `|accel|` en la de menor `|gyro|` → recupera ~9.79–9.83.

Uso:
    .venv/bin/python scripts/extract_imu_svo.py \\
        data/recordings/gym_air/video_1.svo2 \\
        --out results/imu/gym_v1
    # --strict → exit ≠0 si el gate no pasa (para scripts de batch).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyzed.sl as sl

# polls/grab para barrer el buffer high-freq. ~40 satura (plateau medido
# 2026-06-19); 64 deja margen sin costo perceptible (es solo copia de RAM).
POLLS_PER_GRAB = 64
DEG2RAD = np.pi / 180.0

# Gate de tasa: la IMU high-freq viva del ZED 2i ronda 406 Hz; el underwater
# "muerto" da ~15 Hz (1/frame). 200 Hz separa ambos con margen amplio.
MIN_IMU_RATE_HZ = 200.0
# Banda de gravedad aceptable medida en la ventana más quieta (m/s²).
GRAVITY_OK_RANGE = (9.0, 10.6)
# Largo (s) de la ventana deslizante para buscar el tramo más quieto.
GRAVITY_WINDOW_S_DEFAULT = 1.0


def quietest_window_gravity(imu: np.ndarray, win_s: float) -> dict:
    """Gravedad medida en la ventana de ~`win_s` con menor |gyro| del SVO.

    La ventana inicial fija puede pillar la cámara ya en movimiento → `|g|`
    sesgado (caveat 2 del dataset Hito 3). Barriendo ventanas no solapadas y
    quedándonos con la de menor |gyro| medio se aísla el tramo realmente
    cuasi-estático, donde `|accel| ≈ g` (~9.81). `imu`: columnas
    ts_ns, ax, ay, az, wx, wy, wz (gyro ya en rad/s).
    """
    ts = imu[:, 0]
    acc = imu[:, 1:4]
    gyr = imu[:, 4:7]
    gyr_mag = np.linalg.norm(gyr, axis=1)            # rad/s
    win_ns = win_s * 1e9
    bin_id = ((ts - ts[0]) // win_ns).astype(np.int64)
    best = None                                       # (mean|gyro|, |g|, idx, n)
    for b in np.unique(bin_id):
        m = bin_id == b
        n = int(m.sum())
        if n < 10:                                    # ventana sin muestras útiles
            continue
        w = float(gyr_mag[m].mean())
        if best is None or w < best[0]:
            g = float(np.linalg.norm(acc[m].mean(axis=0)))
            best = (w, g, int(b), n)
    if best is None:                                  # SVO muy corto: usa todo
        return {
            "window_s": win_s, "n_samples": int(len(imu)),
            "gravity_magnitude_ms2": float(np.linalg.norm(acc.mean(axis=0))),
            "mean_gyro_deg_s": float(np.degrees(gyr_mag.mean())),
            "window_index": None,
        }
    return {
        "window_s": win_s,
        "n_samples": best[3],
        "gravity_magnitude_ms2": best[1],
        "mean_gyro_deg_s": float(np.degrees(best[0])),
        "window_index": best[2],
    }


def transform_to_matrix(tf: sl.Transform) -> list[list[float]]:
    """sl.Transform -> lista 4x4 (row-major) serializable a JSON.

    `tf.m` ya es un numpy 4x4. La traslación viene en **mm** (la IMU del
    ZED 2i está ~23 mm del lente izquierdo); la rotación cam(left)<-IMU es
    casi identidad. Brazo de palanca despreciable para loose coupling.
    """
    return np.asarray(tf.m, dtype=np.float64).tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svo", type=str, help="Ruta al SVO2 con IMU embebida.")
    ap.add_argument("--out", type=str, required=True,
                    help="Directorio de salida (se crea).")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap de grabs (debug). Default: todo el SVO.")
    ap.add_argument("--static-frames", type=int, default=30,
                    help="Nº de frames iniciales asumidos cuasi-estáticos para "
                         "estimar gravedad y bias de gyro (default 30 ~1 s). "
                         "Solo informativo en meta.json; las caminatas gym NO "
                         "arrancan perfectamente quietas — usar con criterio.")
    ap.add_argument("--gravity-window-s", type=float,
                    default=GRAVITY_WINDOW_S_DEFAULT,
                    help="Largo (s) de la ventana deslizante con la que se busca "
                         "el tramo más quieto para medir |g| (default 1.0). El |g| "
                         "honesto sale de ahí, no de la ventana inicial fija.")
    ap.add_argument("--strict", action="store_true",
                    help="Salir con código ≠0 si el gate de validez no pasa "
                         "(tasa < %.0f Hz o |g| fuera de [%.1f, %.1f]). Para "
                         "scripts de batch." % (MIN_IMU_RATE_HZ, *GRAVITY_OK_RANGE))
    args = ap.parse_args()

    svo_path = Path(args.svo)
    if not svo_path.exists():
        print(f"ERROR: no existe {svo_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    st = cam.open(init)
    if st != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: open SVO falló: {st}", file=sys.stderr)
        return 3

    # Extrínseca cámara(left)<-IMU de fábrica (solo lectura, no se modifica).
    cam_imu_tf = None
    try:
        cfg = cam.get_camera_information().sensors_configuration
        cam_imu_tf = transform_to_matrix(cfg.camera_imu_transform)
    except Exception as e:  # noqa: BLE001
        print(f"AVISO: no se pudo leer camera_imu_transform: {e}", file=sys.stderr)

    rt = sl.RuntimeParameters()
    sd = sl.SensorsData()

    frame_rows: list[tuple[int, int]] = []        # (grab_index, image_ts_ns)
    imu_seen: dict[int, tuple] = {}               # ts_ns -> (ax,ay,az,wx,wy,wz)
    g = 0
    while cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
        if args.max_frames is not None and g >= args.max_frames:
            break
        # timestamp de la imagen de este grab (reloj del SVO)
        cam.get_sensors_data(sd, sl.TIME_REFERENCE.IMAGE)
        img_ts = sd.get_imu_data().timestamp.get_nanoseconds()
        frame_rows.append((g, img_ts))
        # barrido high-freq: pollear CURRENT y deduplicar por timestamp
        for _ in range(POLLS_PER_GRAB):
            cam.get_sensors_data(sd, sl.TIME_REFERENCE.CURRENT)
            imu = sd.get_imu_data()
            ts = imu.timestamp.get_nanoseconds()
            if ts in imu_seen:
                continue
            a = imu.get_linear_acceleration()      # m/s² (ref a buffer reutilizado:
            w = imu.get_angular_velocity()         # copiar valores — gotcha pyzed)
            # GOTCHA: get_angular_velocity() devuelve **deg/s**, NO rad/s (la
            # validación 2026-06-12 ya reportaba |gyro| en °/s). Se convierte a
            # rad/s (SI canónico) para que el EKF integre correcto; sin esto la
            # orientación gira ~57× de más. get_linear_acceleration() sí es m/s².
            imu_seen[ts] = (a[0], a[1], a[2],
                            w[0] * DEG2RAD, w[1] * DEG2RAD, w[2] * DEG2RAD)
        g += 1
    cam.close()

    if not frame_rows:
        print("ERROR: no se grabó ningún frame.", file=sys.stderr)
        return 4

    # --- IMU ordenada por timestamp ---
    ts_sorted = sorted(imu_seen)
    imu = np.array([(ts, *imu_seen[ts]) for ts in ts_sorted], dtype=np.float64)
    frames = np.array(frame_rows, dtype=np.int64)

    # --- stats de tasa IMU ---
    dt_ms = np.diff(imu[:, 0]) / 1e6
    span_s = (imu[-1, 0] - imu[0, 0]) / 1e9
    rate_mean = (len(imu) - 1) / span_s if span_s > 0 else float("nan")
    rate_stats = {
        "n_samples": int(len(imu)),
        "rate_mean_hz": float(rate_mean),       # cifra honesta (cuenta/tiempo)
        "rate_median_dt_hz": float(1000.0 / np.median(dt_ms)),  # infla por ráfagas
        "dt_ms_min": float(dt_ms.min()),
        "dt_ms_p50": float(np.median(dt_ms)),
        "dt_ms_p95": float(np.percentile(dt_ms, 95)),
        "dt_ms_max": float(dt_ms.max()),
        "samples_per_frame": float(len(imu) / len(frames)),
    }

    # --- gravedad y bias de gyro en la ventana inicial cuasi-estática ---
    # (se conserva por compat; NO es el criterio del gate — ver caveat 2)
    nstat = min(args.static_frames, len(frames))
    t_cut = frames[nstat - 1, 1] if nstat > 0 else imu[-1, 0]
    mask0 = imu[:, 0] <= t_cut
    acc0 = imu[mask0, 1:4]
    gyr0 = imu[mask0, 4:7]
    grav_vec = acc0.mean(axis=0).tolist() if len(acc0) else None
    grav_mag = float(np.linalg.norm(acc0.mean(axis=0))) if len(acc0) else None
    gyro_bias = gyr0.mean(axis=0).tolist() if len(gyr0) else None
    gyro_std = gyr0.std(axis=0).tolist() if len(gyr0) else None

    # --- gravedad en la ventana MÁS QUIETA (caveat 2: el |g| honesto) ---
    grav_quiet = quietest_window_gravity(imu, args.gravity_window_s)
    grav_global_mean = float(np.linalg.norm(imu[:, 1:4].mean(axis=0)))

    # --- actividad de gyro (consolida el chequeo de "¿quedaron los giros?") ---
    gyro_mag_deg = np.degrees(np.linalg.norm(imu[:, 4:7], axis=1))
    gyro_activity = {
        "max_deg_s": float(gyro_mag_deg.max()),
        "median_deg_s": float(np.median(gyro_mag_deg)),
        "frac_above_10deg_s": float(np.mean(gyro_mag_deg > 10.0)),
    }

    # --- veredicto del gate (tasa + gravedad), criterio honesto ---
    g_quiet = grav_quiet["gravity_magnitude_ms2"]
    rate_ok = bool(rate_stats["rate_mean_hz"] >= MIN_IMU_RATE_HZ)
    grav_ok = bool(g_quiet is not None
                   and GRAVITY_OK_RANGE[0] <= g_quiet <= GRAVITY_OK_RANGE[1])
    gate_pass = rate_ok and grav_ok
    gate = {
        "pass": gate_pass,
        "rate_ok": rate_ok,
        "gravity_ok": grav_ok,
        "min_rate_hz": MIN_IMU_RATE_HZ,
        "gravity_ok_range_ms2": list(GRAVITY_OK_RANGE),
        "note": "Criterio por TASA (rate_mean_hz), NO por samples_per_frame "
                "(=406Hz/fps, fps-dependiente). Gravedad de la ventana más "
                "quieta, NO de la inicial fija. Ver finding 2026-06-29.",
    }

    meta = {
        "svo": str(svo_path),
        "n_frames": int(len(frames)),
        "image_ts_first_ns": int(frames[0, 1]),
        "image_ts_last_ns": int(frames[-1, 1]),
        "image_fps_mean": float((len(frames) - 1) /
                                ((frames[-1, 1] - frames[0, 1]) / 1e9))
        if len(frames) > 1 else None,
        "polls_per_grab": POLLS_PER_GRAB,
        "imu_rate": rate_stats,
        "camera_imu_transform": cam_imu_tf,   # cam(left)<-IMU 4x4 row-major
        "gate": gate,                            # veredicto honesto (tasa+gravedad)
        "gravity_quietest_window": grav_quiet,   # |g| del tramo de menor |gyro|
        "gravity_global_mean_ms2": grav_global_mean,  # control: media global de |accel|
        "gyro_activity": gyro_activity,          # ¿quedaron registrados los giros?
        "static_window": {                       # COMPAT: puede dar falsa alarma
            "n_frames": int(nstat),
            "gravity_mean_acc_ms2": grav_vec,    # dirección+magnitud de g en IMU
            "gravity_magnitude_ms2": grav_mag,   # ⚠ sesgado si la cámara ya se movía
            "gyro_bias_rad_s": gyro_bias,        # bias crudo (sesgado por mov.)
            "gyro_std_rad_s": gyro_std,          # si es grande, NO estaba quieta
        },
        "frame_no_convention": "TUM timestamp de run_sdpvo_metric == grab_index "
                               "0-based; mapear frame_no -> frame_times[grab_index].",
    }

    np.savetxt(out_dir / "imu.csv", imu,
               header="ts_ns,ax,ay,az,wx,wy,wz", delimiter=",",
               fmt=["%d", "%.6f", "%.6f", "%.6f", "%.9f", "%.9f", "%.9f"],
               comments="")
    np.savetxt(out_dir / "frame_times.csv", frames,
               header="grab_index,ts_ns", delimiter=",", fmt="%d", comments="")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"OK  {svo_path.name}")
    print(f"  frames        : {len(frames)}")
    print(f"  IMU samples   : {len(imu)}  "
          f"({rate_stats['samples_per_frame']:.2f}/frame, fps-dependiente)")
    print(f"  IMU rate mean : {rate_stats['rate_mean_hz']:.1f} Hz "
          f"(dt p50 {rate_stats['dt_ms_p50']:.3f} ms)  "
          f"[{'OK' if rate_ok else 'BAJO'}, mín {MIN_IMU_RATE_HZ:.0f} Hz]")
    if g_quiet is not None:
        print(f"  |g| quieto    : {g_quiet:.3f} m/s²  "
              f"(ventana {grav_quiet['mean_gyro_deg_s']:.1f}°/s, "
              f"media global {grav_global_mean:.3f})")
    else:
        print("  |g| quieto    : n/a")
    print(f"  gyro actividad: max {gyro_activity['max_deg_s']:.0f}°/s  "
          f"mediana {gyro_activity['median_deg_s']:.0f}°/s  "
          f">10°/s el {100*gyro_activity['frac_above_10deg_s']:.0f}%")
    print(f"  GATE          : {'PASS ✓' if gate_pass else 'FAIL ✗'}"
          f"  (tasa {'ok' if rate_ok else 'baja'}, "
          f"gravedad {'ok' if grav_ok else 'fuera de rango'})")
    print(f"  -> {out_dir}/")
    if args.strict and not gate_pass:
        print("ERROR: gate no pasó y --strict activo.", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
