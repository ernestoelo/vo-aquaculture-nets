#!/usr/bin/env python3
"""run_sdpvo_offline.py — Procesa un video con S-DPVO y guarda trayectoria.

Pipeline DPVO real (Teed et al. 2023, integrada 2026-05-17):

    slam = DPVO(cfg, network, ht, wd, viz=bool)
    slam(t, image, intrinsics)   # por frame
    pred_traj = slam.terminate() # TUM-compat

Ver docs/04-sdpvo-build.md sección "API S-DPVO" para detalles.

Fuentes soportadas:
    --imagedir <path.mp4 | dir-de-PNGs>      (ruta directa, modo demo.py)
    --imagedir <path.svo>                    (convertir a MP4 antes; ver
                                              scripts/svo_to_mp4.sh o
                                              docs/ops/svo-to-mp4.md)

Outputs en results/<fecha>__<name>__<hash>/:
    - trajectory.txt    (TUM format)
    - <name>.ply        (si --save-reconstruction)
    - <name>.pdf        (si --plot)
    - stats.json        (FPS efectivo, configuración, sha del modelo)
    - run.log           (stdout/stderr del run)

Uso típico (headless, sin viewer):
    python scripts/run_sdpvo_offline.py \\
        --imagedir data/recordings/video_4.mp4 \\
        --calib third_party/S_DPVO/calib/zed2i_left.txt \\
        --network third_party/S_DPVO/weights/dpvo.pth \\
        --stride 2 --name smoke

Para corrida con viewer (requiere DISPLAY=:0 nativo NVIDIA Tegra o HDMI físico):
    LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0 \\
    DISPLAY=:0 \\
    python scripts/run_sdpvo_offline.py --viz ...

Pre-condiciones del entorno:
    - S-DPVO compilado (workarounds §1-§3, §7 — ver docs/ops/workarounds.md).
    - dpvo.pth en third_party/S_DPVO/weights/dpvo.pth (bootstrap/build_sdpvo.sh).
    - cv2, numpy, plyfile, yacs en venv.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SDPVO_DIR = REPO_ROOT / "third_party" / "S_DPVO"
DEFAULT_NETWORK = SDPVO_DIR / "weights" / "dpvo.pth"
DEFAULT_CONFIG_SDPVO = SDPVO_DIR / "config" / "default.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run S-DPVO over a video and save trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--imagedir", type=Path, required=True,
                   help="MP4 file or directory of PNG/JPG images.")
    p.add_argument("--calib", type=Path, required=True,
                   help="Calib file: 'fx fy cx cy [k1 k2 p1 p2]' (8 values; "
                        "los últimos 4 = 0 para imagen rectificada del ZED SDK).")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK,
                   help="Path to dpvo.pth checkpoint.")
    p.add_argument("--config-sdpvo", type=Path, default=DEFAULT_CONFIG_SDPVO,
                   help="DPVO config YAML (yacs).")
    p.add_argument("--stride", type=int, default=2,
                   help="Procesa 1 de cada N frames (default 2 = demo.py).")
    p.add_argument("--skip", type=int, default=0,
                   help="Frames iniciales a saltar.")
    p.add_argument("--stereo-side", choices=["left", "right", "full"],
                   default="left",
                   help="Recorte para video estéreo lado-a-lado (ej. svo_export "
                        "del SDK ZED genera 3840×1080): 'left'/'right'. Para un "
                        "MP4 YA MONOCULAR (1920 ancho, salida de "
                        "svo_to_mp4_x86.py) usar 'full': el default 'left' "
                        "recortaría la mitad de un frame ya mono → deja cx=967 "
                        "fuera del recorte → geometría rota → 'BA failed' "
                        "(distinto del mismatch cuSOLVER; ver finding "
                        "2026-06-03-dpvo-embedded-cusolver-ba-broken).")
    p.add_argument("--buffer", type=int, default=2048,
                   help="DPVO BUFFER_SIZE — max #keyframes.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap absoluto sobre frames procesados (debug).")
    p.add_argument("--viz", action="store_true",
                   help="Activa viewer Pangolin (requiere DISPLAY funcional).")
    p.add_argument("--plot", action="store_true",
                   help="Genera PDF top-down con trayectoria.")
    p.add_argument("--save-trajectory", action="store_true", default=True,
                   help="Guarda trajectory.txt en TUM format (default ON).")
    p.add_argument("--save-reconstruction", action="store_true",
                   help="Guarda <name>.ply de la nube sparse final.")
    p.add_argument("--timeit", action="store_true",
                   help="Imprime tiempo por frame en SLAM.")
    p.add_argument("--inline-reader", action="store_true",
                   help="Lee frames inline (sin multiprocessing.Process). "
                        "Workaround para hangs aarch64 fork-CUDA: en Jetson "
                        "JP5.1.1 el fork del Process hereda estado torch CUDA "
                        "y a veces se cuelga antes del primer slam(). "
                        "Off por default (más rápido); on para diagnóstico.")
    p.add_argument("--name", type=str, default=None,
                   help="Etiqueta para el directorio results/ (default: stem del input).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directorio base para results/ (default: repo/results).")
    return p.parse_args()


def config_hash(d: dict) -> str:
    blob = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def make_out_dir(name: str, run_meta: dict, out_base: Path | None) -> Path:
    base = out_base or (REPO_ROOT / "results")
    date = datetime.date.today().isoformat()
    chash = config_hash(run_meta)
    out = base / f"{date}__{name}__{chash}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stereo_video_stream(queue, imagedir: str, calib: str, stride: int,
                         skip: int, side: str) -> None:
    """Variante de dpvo.stream.video_stream que recorta lado de un MP4
    estéreo side-by-side (LEFT|RIGHT, 2W×H) si side != 'full'.

    Mantiene el resize 0.5× y crop a múltiplo de 16 idéntico al original
    de dpvo, sólo que opera sobre la mitad seleccionada del frame.
    """
    import cv2  # importar adentro: el Process se forkea con copias limpias

    calib_arr = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib_arr[:4]
    K = np.eye(3)
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy

    cap = cv2.VideoCapture(imagedir)
    if not cap.isOpened():
        queue.put((-1, None, np.array([fx, fy, cx, cy])))
        return

    # Skip iniciales
    for _ in range(skip):
        ret, _ = cap.read()
        if not ret:
            queue.put((-1, None, np.array([fx, fy, cx, cy])))
            return

    t = 0
    image = None
    intrinsics = np.array([fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5])
    while True:
        ret, image = None, None
        for _ in range(stride):
            ret, image = cap.read()
            if not ret:
                break
        if not ret:
            break

        # Recorte estéreo
        if side != "full":
            h, w, _ = image.shape
            mid = w // 2
            image = image[:, :mid] if side == "left" else image[:, mid:]

        if len(calib_arr) > 4 and np.any(calib_arr[4:] != 0):
            image = cv2.undistort(image, K, calib_arr[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[: h - h % 16, : w - w % 16]
        intrinsics = np.array([fx * 0.5, fy * 0.5, cx * 0.5, cy * 0.5])

        queue.put((t, image, intrinsics))
        t += 1

    queue.put((-1, image if image is not None else None, intrinsics))
    cap.release()


def main() -> int:
    args = parse_args()

    # Validaciones tempranas (antes de importar torch/dpvo, que son caros)
    for p in (args.imagedir, args.calib, args.network, args.config_sdpvo):
        if not p.exists():
            print(f"ERROR: no existe {p}", file=sys.stderr)
            return 2

    name = args.name or args.imagedir.stem
    run_meta = {
        "imagedir": str(args.imagedir),
        "calib": str(args.calib),
        "network": str(args.network),
        "config_sdpvo": str(args.config_sdpvo),
        "stride": args.stride,
        "skip": args.skip,
        "buffer": args.buffer,
        "viz": args.viz,
        "max_frames": args.max_frames,
    }
    out_dir = make_out_dir(name, run_meta, args.out_dir)
    print(f"input:   {args.imagedir}")
    print(f"output:  {out_dir}")

    # Imports pesados después de validaciones
    import torch
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    from dpvo.stream import image_stream, video_stream
    from dpvo.utils import Timer

    print(f"torch:   {torch.__version__}  cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA no disponible — wheel torch incorrecto.", file=sys.stderr)
        return 3

    # Config DPVO. No usar cfg.merge_from_file: yacs abre el YAML con el
    # encoding del locale y en shells no-interactivas (nohup/ssh) cae a ASCII
    # -> UnicodeDecodeError con los comentarios en español (finding
    # 2026-06-11-dpvo-metric-embedded-baseline). Leer UTF-8 explícito:
    cfg.merge_from_other_cfg(cfg.load_cfg(args.config_sdpvo.read_text(encoding="utf-8")))
    cfg.BUFFER_SIZE = args.buffer

    # SHA256 del checkpoint
    net_sha = file_sha256(args.network)
    print(f"network: {args.network.name}  sha256={net_sha[:12]}…")

    # Pipeline (igual al run() de demo.py upstream, con soporte estéreo)
    slam: DPVO | None = None
    n_processed = 0
    t_start = time.monotonic()
    per_frame_ms: list[float] = []

    def _process_one(t: int, image: np.ndarray, intrinsics: np.ndarray) -> bool:
        """Procesa un frame. Retorna False si se debe parar el loop."""
        nonlocal slam, n_processed
        image_t = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics_t = torch.from_numpy(intrinsics).cuda()
        if slam is None:
            ht, wd = image_t.shape[1], image_t.shape[2]
            # En el commit f7266f7 (no-viewer) DPVO.__init__ no acepta viz=.
            # Sólo lo pasamos si el flag está prendido (commit 15a4585 con viewer).
            print(f"slam:    DPVO ht={ht} wd={wd} viz={args.viz}", flush=True)
            kw = {"viz": True} if args.viz else {}
            slam = DPVO(cfg, str(args.network), ht=ht, wd=wd, **kw)
        f0 = time.monotonic()
        # torch.inference_mode evita que el grafo de autograd se acumule
        # entre llamadas a slam(); DPVO upstream solo hace .eval() (que NO
        # desactiva autograd) — sin esto la memoria GPU crece linealmente
        # y revienta a ~50 frames en una RTX 3060 12 GB.
        with Timer("SLAM", enabled=args.timeit), torch.inference_mode():
            slam(t, image_t, intrinsics_t)
        per_frame_ms.append((time.monotonic() - f0) * 1000.0)
        n_processed += 1
        if n_processed <= 3 or n_processed % 50 == 0:
            fps = n_processed / (time.monotonic() - t_start)
            print(f"  [t={t}] processed={n_processed}  fps≈{fps:.2f}  "
                  f"last={per_frame_ms[-1]:.0f}ms", flush=True)
        return not (args.max_frames is not None and n_processed >= args.max_frames)

    if args.inline_reader:
        # Workaround: lee frames inline (sin Process) — evita fork+CUDA hang
        print("reader:  inline (single-thread)", flush=True)
        import cv2
        calib_arr = np.loadtxt(args.calib, delimiter=" ")
        fx, fy, cx, cy = calib_arr[:4]
        K = np.eye(3); K[0, 0] = fx; K[0, 2] = cx; K[1, 1] = fy; K[1, 2] = cy
        cap = cv2.VideoCapture(str(args.imagedir))
        # skip
        for _ in range(args.skip):
            if not cap.read()[0]:
                break
        t_idx = 0
        try:
            while True:
                ret, image = None, None
                for _ in range(args.stride):
                    ret, image = cap.read()
                    if not ret:
                        break
                if not ret:
                    break
                if args.stereo_side != "full":
                    h_in, w_in, _ = image.shape
                    mid = w_in // 2
                    image = image[:, :mid] if args.stereo_side == "left" \
                            else image[:, mid:]
                if len(calib_arr) > 4 and np.any(calib_arr[4:] != 0):
                    image = cv2.undistort(image, K, calib_arr[4:])
                image = cv2.resize(image, None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_AREA)
                h, w, _ = image.shape
                image = image[:h - h % 16, :w - w % 16]
                intrinsics = np.array([fx * 0.5, fy * 0.5,
                                       cx * 0.5, cy * 0.5])
                if not _process_one(t_idx, image, intrinsics):
                    break
                t_idx += 1
        finally:
            cap.release()
    else:
        print("reader:  multiprocessing.Process", flush=True)
        queue: Queue = Queue(maxsize=8)
        if args.imagedir.is_dir():
            reader = Process(
                target=image_stream,
                args=(queue, str(args.imagedir), str(args.calib),
                      args.stride, args.skip),
            )
        else:
            reader = Process(
                target=_stereo_video_stream,
                args=(queue, str(args.imagedir), str(args.calib),
                      args.stride, args.skip, args.stereo_side),
            )
        reader.start()
        try:
            while True:
                t, image, intrinsics = queue.get()
                if t < 0:
                    break
                if not _process_one(t, image, intrinsics):
                    break
        finally:
            reader.terminate()
            reader.join()

    if slam is None:
        print("ERROR: no se procesó ningún frame (queue cerró antes).", file=sys.stderr)
        return 4

    print("Draining DPVO buffer (12 updates)...")
    for _ in range(12):
        slam.update()

    if args.save_reconstruction:
        from plyfile import PlyData, PlyElement
        points = slam.points_.cpu().numpy()[: slam.m]
        colors = slam.colors_.view(-1, 3).cpu().numpy()[: slam.m]
        arr = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
            dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
        el = PlyElement.describe(arr, "vertex")
        ply_path = out_dir / f"{name}.ply"
        PlyData([el], text=True).write(str(ply_path))
        print(f"plyfile: {ply_path}  ({len(arr)} points)")
        pred_traj = slam.terminate()
    else:
        pred_traj = slam.terminate()

    elapsed = time.monotonic() - t_start
    fps = n_processed / elapsed if elapsed > 0 else 0.0
    per_frame_ms_arr = np.array(per_frame_ms)

    # Trajectory
    if args.save_trajectory:
        from dpvo.plot_utils import save_trajectory_tum_format
        traj_path = out_dir / "trajectory.txt"
        save_trajectory_tum_format(pred_traj, str(traj_path))

    # Plot
    if args.plot:
        from dpvo.plot_utils import plot_trajectory
        pdf_path = out_dir / f"{name}.pdf"
        plot_trajectory(
            pred_traj,
            title=f"S-DPVO trajectory for {name}",
            filename=str(pdf_path),
        )

    # Stats
    stats = {
        "imagedir": str(args.imagedir),
        "name": name,
        "n_frames_processed": n_processed,
        "elapsed_sec": round(elapsed, 3),
        "fps_effective": round(fps, 2),
        "ms_per_frame_p50": round(float(np.percentile(per_frame_ms_arr, 50)), 2)
                            if len(per_frame_ms_arr) else None,
        "ms_per_frame_p95": round(float(np.percentile(per_frame_ms_arr, 95)), 2)
                            if len(per_frame_ms_arr) else None,
        "ms_per_frame_p5": round(float(np.percentile(per_frame_ms_arr, 5)), 2)
                           if len(per_frame_ms_arr) else None,
        "stride": args.stride,
        "skip": args.skip,
        "buffer": args.buffer,
        "viz": args.viz,
        "stereo_side": args.stereo_side,
        "network_path": str(args.network),
        "network_sha256": net_sha,
        "calib_path": str(args.calib),
        "config_sdpvo_path": str(args.config_sdpvo),
        "submodule_head": _git_head_short(SDPVO_DIR),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"stats:   {out_dir / 'stats.json'}")
    print(f"FPS efectivo: {fps:.2f}  ({n_processed} frames en {elapsed:.2f}s)")
    if len(per_frame_ms_arr):
        print(f"  ms/frame: p5={stats['ms_per_frame_p5']} "
              f"p50={stats['ms_per_frame_p50']} p95={stats['ms_per_frame_p95']}")

    return 0


def _git_head_short(path: Path) -> str | None:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
