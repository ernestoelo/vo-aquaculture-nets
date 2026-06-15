#!/usr/bin/env python3
"""svo_to_stereo_pngs.py — Convierte un SVO de la ZED 2i a un dataset
estéreo en formato `GeneralStereo` de MAC-VO:

    <output>/
    ├── left/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── right/
        ├── 000000.png
        ├── 000001.png
        └── ...

Las imágenes salen RECTIFICADAS por el SDK (calibración de fábrica),
así que MAC-VO recibe ya el espacio canónico estéreo.

Reglas (§8 CLAUDE.md):
  - Sólo LEE la calibración de fábrica embebida en el SVO. NO la modifica.
  - LEFT y RIGHT comparten matriz K (rectificadas), bl = T_x de fábrica.

Uso:
  python scripts/svo_to_stereo_pngs.py \\
      --input data/recordings/video_4.svo \\
      --output data/recordings/video_4_stereo/ \\
      [--skip N] [--stride N] [--max-frames N] [--scale 0.5]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path,
                   help="Ruta al archivo .svo (entrada).")
    p.add_argument("--output", required=True, type=Path,
                   help="Directorio destino (se crearán left/ y right/).")
    p.add_argument("--skip", type=int, default=0,
                   help="Saltar primeros N frames (ej: 150 para evitar "
                        "frames quemados del start).")
    p.add_argument("--stride", type=int, default=1,
                   help="Procesar 1 de cada N frames.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Limitar frames exportados (smoke).")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Factor de resize (1.0=full, 0.5=half).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"ERROR: no existe {args.input}", file=sys.stderr)
        return 1
    left_dir = args.output / "left"
    right_dir = args.output / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(str(args.input))
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.NONE

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: cam.open() devolvió {status}", file=sys.stderr)
        return 2

    info = cam.get_camera_information()
    res = info.camera_configuration.resolution
    fps = info.camera_configuration.fps
    cal = info.camera_configuration.calibration_parameters
    # baseline: el ZED SDK reporta en mm si coordinate_units no se setea —
    # acá lo seteamos a METER en init, pero la stereo_transform sigue en mm
    # según el SDK 5.3. Convertimos a m por las dudas.
    bl_raw = float(cal.stereo_transform.get_translation().get()[0])
    bl_m = bl_raw if bl_raw < 1.0 else bl_raw / 1000.0
    total = cam.get_svo_number_of_frames()

    out_w = int(res.width * args.scale)
    out_h = int(res.height * args.scale)
    print(f"[svo→stereo] input:  {args.input}  ({total} frames @ {fps} fps, "
          f"{res.width}×{res.height})")
    print(f"[svo→stereo] output: {args.output}  scale={args.scale} "
          f"→ {out_w}×{out_h}")
    print(f"[svo→stereo] calib LEFT (rectificada de fábrica):")
    print(f"  fx={cal.left_cam.fx:.4f}  fy={cal.left_cam.fy:.4f}  "
          f"cx={cal.left_cam.cx:.4f}  cy={cal.left_cam.cy:.4f}")
    print(f"[svo→stereo] baseline = {bl_m:.6f} m")
    print(f"[svo→stereo] skip={args.skip}  stride={args.stride}  "
          f"max_frames={args.max_frames}")

    # Skip iniciales
    for _ in range(args.skip):
        rt = sl.RuntimeParameters()
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            break

    imgL = sl.Mat()
    imgR = sl.Mat()
    n_out = 0
    t0 = time.time()

    while True:
        rt = sl.RuntimeParameters()
        # Stride: leer (stride-1) frames sin escribir, luego leer 1 y escribir
        for _ in range(args.stride - 1):
            cam.grab(rt)
        status = cam.grab(rt)
        if status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if status != sl.ERROR_CODE.SUCCESS:
            continue

        cam.retrieve_image(imgL, sl.VIEW.LEFT)
        cam.retrieve_image(imgR, sl.VIEW.RIGHT)
        L = imgL.get_data()
        R = imgR.get_data()
        if L.shape[2] == 4:
            L = cv2.cvtColor(L, cv2.COLOR_BGRA2BGR)
            R = cv2.cvtColor(R, cv2.COLOR_BGRA2BGR)
        if args.scale != 1.0:
            L = cv2.resize(L, (out_w, out_h), interpolation=cv2.INTER_AREA)
            R = cv2.resize(R, (out_w, out_h), interpolation=cv2.INTER_AREA)

        idx = f"{n_out:06d}"
        cv2.imwrite(str(left_dir / f"{idx}.png"), L)
        cv2.imwrite(str(right_dir / f"{idx}.png"), R)
        n_out += 1

        if n_out % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{n_out}/{total}] {n_out/elapsed:.1f} fps", flush=True)

        if args.max_frames and n_out >= args.max_frames:
            print(f"  → alcanzado --max-frames {args.max_frames}")
            break

    cam.close()
    elapsed = time.time() - t0
    print(f"[svo→stereo] done: {n_out} pares en {elapsed:.1f}s "
          f"({n_out/elapsed:.1f} fps)")

    # Calibración escalada a la resolución de salida
    s = args.scale
    print()
    print(f"[svo→stereo] config sequence YAML sugerido:")
    print(f"  bl: {bl_m:.6f}")
    print(f"  camera:")
    print(f"    fx: {cal.left_cam.fx * s:.6f}")
    print(f"    fy: {cal.left_cam.fy * s:.6f}")
    print(f"    cx: {cal.left_cam.cx * s:.6f}")
    print(f"    cy: {cal.left_cam.cy * s:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
