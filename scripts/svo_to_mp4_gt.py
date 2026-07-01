#!/usr/bin/env python3
"""SVO -> MP4 H.264 liviano con OVERLAY de frame# (grab_index 0-based) + segundos.

Pensado para anotar ground truth: el usuario ve el video y apunta en qué
frame/segundo cae cada cinta naranja y el giro de la cámara, además de los
segundos de basura a recortar al inicio/fin. El frame# quemado == el índice
de fila de trajectory.txt (grab_index 0-based), así el mapeo es directo.

Uso:
  python scripts/svo_to_mp4_gt.py --input X.svo2 --output Y.mp4 [--width 960] [--crf 30]
"""
import argparse, subprocess, sys, time
from pathlib import Path
import cv2
import pyzed.sl as sl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--width", type=int, default=960, help="ancho de salida (downscale).")
    p.add_argument("--crf", type=int, default=30, help="calidad H.264 (mayor=más liviano).")
    p.add_argument("--view", choices=("left", "right"), default="left")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if not a.input.is_file():
        print(f"ERROR: no existe {a.input}", file=sys.stderr); return 1
    a.output.parent.mkdir(parents=True, exist_ok=True)

    init = sl.InitParameters()
    init.set_from_svo_file(str(a.input))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    cam = sl.Camera()
    st = cam.open(init)
    if st != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: cam.open() = {st}", file=sys.stderr); return 2

    info = cam.get_camera_information()
    res = info.camera_configuration.resolution
    fps = float(info.camera_configuration.fps)
    total = cam.get_svo_number_of_frames()
    ow = a.width
    oh = int(round(res.height * ow / res.width)); oh -= oh % 2
    print(f"[gt-mp4] {a.input.name}: {total} frames @ {fps} fps  {res.width}x{res.height} -> {ow}x{oh}", flush=True)

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}", "-r", f"{fps}",
         "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.output)],
        stdin=subprocess.PIPE)

    view = sl.VIEW.LEFT if a.view == "left" else sl.VIEW.RIGHT
    mat = sl.Mat()
    rt = sl.RuntimeParameters()
    idx = 0
    t0 = time.time()
    while True:
        s = cam.grab(rt)
        if s == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break
        if s != sl.ERROR_CODE.SUCCESS:
            idx += 1
            continue
        cam.retrieve_image(mat, view)
        frame = mat.get_data()
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.resize(frame, (ow, oh))
        label = f"frame {idx}   {idx/fps:6.2f} s"
        cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        ff.stdin.write(frame.tobytes())
        idx += 1
        if idx % 300 == 0:
            print(f"  [{idx}/{total}] {idx/(time.time()-t0):.1f} fps", flush=True)

    ff.stdin.close(); ff.wait(); cam.close()
    mb = a.output.stat().st_size / 1e6 if a.output.exists() else 0
    print(f"[gt-mp4] done {a.output} ({mb:.1f} MB, {idx} frames, {time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
