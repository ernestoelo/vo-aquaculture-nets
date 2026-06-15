#!/usr/bin/env python3
"""run_zed_pt.py — ZED SDK Positional Tracking sobre un SVO → TUM.

Tercera línea base del benchmark (referencia industrial, caja negra,
visual-inercial con memoria espacial → ÚNICO de los tres con cierre de
lazo/relocalización). Reproduce el SVO offline y guarda la pose WORLD por
frame en formato TUM con timestamp = frame-index (la convención del repo).

Notas:
  - Usa la calibración de fábrica embebida (no se toca, §8 CLAUDE.md).
  - `enable_area_memory=True` (default del SDK) habilita relocalización.
  - La IMU se usa si el SVO la trae; los SVO antiguos (video_4) tienen
    giroscopio ~0 (finding 2026-05-26) → el resultado puede degradar.

Uso:
    .venv/bin/python scripts/run_zed_pt.py \
        --svo data/recordings/gym_air/video_1.svo2 \
        --out results/zed_pt/gym_video_1 [--depth-mode NEURAL]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyzed.sl as sl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--depth-mode", default="NEURAL",
                    choices=["NEURAL", "NEURAL_LIGHT", "ULTRA", "PERFORMANCE"])
    ap.add_argument("--mode", default=None, choices=["GEN_1", "GEN_2", "GEN_3"],
                    help="POSITIONAL_TRACKING_MODE. GEN_3 (default del SDK) exige "
                         "IMU de alta frecuencia (SVO2); para SVO v1 antiguos "
                         "(ej. video_4) usar GEN_1/GEN_2 + --no-imu-fusion.")
    ap.add_argument("--no-imu-fusion", action="store_true",
                    help="Deshabilita la fusión IMU (necesario en SVO v1 sin "
                         "sensores de alta frecuencia — incluso GEN_1/GEN_2 "
                         "fallan con HIGH FREQUENCY SENSORS DATA REQUIRED si "
                         "no se pasa este flag).")
    args = ap.parse_args()

    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo))
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)

    cam = sl.Camera()
    st = cam.open(init)
    if st != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR open(): {st}")
        return 2

    pt = sl.PositionalTrackingParameters()
    if args.mode is not None:
        pt.mode = getattr(sl.POSITIONAL_TRACKING_MODE, args.mode)
    if args.no_imu_fusion:
        pt.enable_imu_fusion = False
    st = cam.enable_positional_tracking(pt)
    if st != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR enable_positional_tracking(): {st}")
        return 3

    args.out.mkdir(parents=True, exist_ok=True)
    pose = sl.Pose()
    rows = []
    states = {}
    t0 = time.time()
    while True:
        st = cam.grab()
        if st == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if st != sl.ERROR_CODE.SUCCESS:
            print(f"grab(): {st} — abortando")
            break
        frame = cam.get_svo_position()
        state = cam.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        states[str(state)] = states.get(str(state), 0) + 1
        t = pose.get_translation(sl.Translation()).get()
        q = pose.get_orientation(sl.Orientation()).get()  # (ox, oy, oz, ow)
        rows.append(f"{float(frame):.1f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    elapsed = time.time() - t0
    cam.disable_positional_tracking()
    cam.close()

    traj = args.out / "trajectory_tum.txt"
    traj.write_text("\n".join(rows) + "\n")
    stats = {
        "svo": str(args.svo), "depth_mode": args.depth_mode,
        "frames": len(rows), "elapsed_s": round(elapsed, 1),
        "fps_effective": round(len(rows) / elapsed, 2),
        "tracking_states": states,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"OK {traj} — {len(rows)} poses, {elapsed:.0f} s "
          f"({len(rows)/elapsed:.1f} fps), states={states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
