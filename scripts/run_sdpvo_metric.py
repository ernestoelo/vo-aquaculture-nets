#!/usr/bin/env python3
"""run_sdpvo_metric.py — DPVO mono con inyección de escala métrica (fase 2).

Recupera la **escala métrica** de DPVO mono sembrando la profundidad inversa
de los parches (`dpvo.py`, canal 2, antes `torch.rand_like` → escala-ambiguo)
con la **depth estéreo del ZED SDK**. Lee imagen LEFT **y** depth directo del
SVO con pyzed → alineación pixel-perfecta sin juego MP4↔depth.

Punto de inyección y plan: finding
docs/findings/2026-06-05-dpvo-sigma-vs-macvo-covariance-audit.md. Requiere el
hook `DPVO._seed_inverse_depth` + arg `depth=` en `__call__` (mismo finding).

Modos de inyección (`--inject`):
  off    — baseline DPVO (random inverse-depth). Equivale a run_sdpvo_offline.
  init   — siembra depth métrica SOLO durante el init de 8 frames; la
           propagación por mediana arrastra la escala (variante A del finding).
  always — siembra depth métrica en CADA frame (variante B): ancla la
           profundidad dentro del BA → ataca escala Y jitter.

Uso:
  .venv/bin/python scripts/run_sdpvo_metric.py \\
      --svo data/recordings/video_4.svo \\
      --calib third_party/S_DPVO/calib/zed2i_left.txt \\
      --inject always --depth-mode NEURAL --skip 150 --stride 2 --name v4_metric

RECETA CANÓNICA GYM (la del baseline ATE 0.135 m / 19.1 fps — OJO: el
default del runner es default.yaml con 96 patches, NO es la canónica;
ver finding docs/findings/2026-06-12-dpvo-resolution-sweep-gym-v1.md):
  .venv/bin/python scripts/run_sdpvo_metric.py \\
      --svo data/recordings/gym_air/video_1.svo2 \\
      --calib third_party/S_DPVO/calib/zed2i_gym_video1.txt \\
      --config-sdpvo third_party/S_DPVO/config/x86_smoke.yaml \\
      --inject prior_insolver --prior-strength 1000 \\
      --depth-mode NEURAL --stride 1 --skip 15 --scale 0.5

Semántica del stats.json (no mezclar): fps_effective = end-to-end
(grab+depth+resize+slam; en x86 acotado por el SDK) vs ms_per_frame_* =
SOLO la llamada a slam().

Salida en results/<fecha>__<name>__<hash>/: trajectory.txt (TUM) + stats.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SDPVO_DIR = REPO_ROOT / "third_party" / "S_DPVO"
DEFAULT_NETWORK = SDPVO_DIR / "weights" / "dpvo.pth"
DEFAULT_CONFIG = SDPVO_DIR / "config" / "default.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DPVO mono con inyección de escala métrica del ZED SDK.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--svo", type=Path, required=True, help="Archivo .svo de entrada.")
    p.add_argument("--calib", type=Path, required=True,
                   help="Calib full-res: 'fx fy cx cy [k1..]' (8+ valores).")
    p.add_argument("--network", type=Path, default=DEFAULT_NETWORK)
    p.add_argument("--config-sdpvo", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--inject", choices=["off", "init", "always", "prior", "prior_insolver"],
                   default="always",
                   help="Modo de inyección de depth métrica (default always). "
                        "prior = blend post-BA (gauge-libre); prior_insolver = "
                        "variante C in-solver (factor unario en la BA de Python).")
    p.add_argument("--prior-alpha", type=float, default=0.5,
                   help="Peso del blend post-BA (solo --inject prior). "
                        "0<α≤1; α=1 ≈ fija la depth (hard), α<1 ancla blanda. Default 0.5.")
    p.add_argument("--prior-strength", type=float, default=10.0,
                   help="Fuerza del factor unario in-solver (solo --inject "
                        "prior_insolver), relativa al término de datos (wp = "
                        "strength·mediana(C)). strength=0 ≈ BA Python sin prior "
                        "(sanity vs fastba); 1≈moderado; 10≈fuerte; 100≈casi-hard. Default 10.")
    p.add_argument("--quality-boost", type=float, default=1.0,
                   help="Peso 2D informado (solo prior_insolver): factor de "
                        "confianza extra para el residual de reproyección de los "
                        "parches con depth métrica ZED válida. 1.0=off; 2-4 sube "
                        "la influencia de los parches anclados (reduce jitter). Default 1.0.")
    p.add_argument("--depth-mode", default="NEURAL",
                   help="DEPTH_MODE del ZED SDK (NEURAL|ULTRA|...). Default NEURAL.")
    p.add_argument("--confidence", type=int, default=None,
                   help="RuntimeParameters.confidence_threshold del SDK (1-100): "
                        "descarta píxeles de depth con confianza bajo el umbral "
                        "(quedan NaN). Default None = default del SDK (se imprime "
                        "al abrir, para baseline honesto del sweep).")
    p.add_argument("--texture-conf", type=int, default=None,
                   help="RuntimeParameters.texture_confidence_threshold del SDK "
                        "(1-100): descarta píxeles de zonas sin textura. Default "
                        "None = default del SDK.")
    p.add_argument("--depth-max", type=float, default=None,
                   help="Descarta depth > M metros ANTES del resize/siembra "
                        "(NaN → parche sin ancla, peso 0 en el factor unario). "
                        "Semántica 'como MAC-VO max_depth': DESCARTA, no clampea. "
                        "Default None = sin tope (el clamp del BA equivale a "
                        "Z ∈ [0.1, 1000] m).")
    p.add_argument("--stride", type=int, default=2,
                   help="Procesa 1 de cada N frames (default 2).")
    p.add_argument("--skip", type=int, default=0, help="Frames iniciales a saltar.")
    p.add_argument("--scale", type=float, default=0.5,
                   help="Factor de resize aplicado a imagen y depth (default 0.5).")
    p.add_argument("--buffer", type=int, default=2048, help="DPVO BUFFER_SIZE.")
    p.add_argument("--seed", type=int, default=None,
                   help="Fija la semilla RNG (torch/cuda) para reproducibilidad. "
                        "DPVO muestrea parches con torch.randint -> sin semilla la "
                        "escala varía run-a-run. Default None (no fija).")
    p.add_argument("--max-frames", type=int, default=None, help="Cap de frames (debug).")
    p.add_argument("--smooth-window", type=int, default=9,
                   help="Ventana del filtro Savitzky-Golay post-proceso (impar; 0 "
                        "desactiva). Genera trajectory_smooth.txt con el jitter de "
                        "alta frecuencia removido (path length coherente) SIN tocar "
                        "escala ni ATE. Costo ~1 ms — no afecta fps. Default 9.")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def config_hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def make_out_dir(name: str, meta: dict, base: Path | None) -> Path:
    base = base or (REPO_ROOT / "results")
    out = base / f"{datetime.date.today().isoformat()}__{name}__{config_hash(meta)}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def crop16(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return img[: h - h % 16, : w - w % 16]


def main() -> int:
    args = parse_args()
    for pth in (args.svo, args.calib, args.network, args.config_sdpvo):
        if not pth.exists():
            print(f"ERROR: no existe {pth}", file=sys.stderr)
            return 2

    try:
        import pyzed.sl as sl
    except ImportError as exc:
        print("ERROR: pyzed.sl no importable (activar .venv con ZED SDK).", file=sys.stderr)
        return 1

    calib = np.loadtxt(args.calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    name = args.name or f"{args.svo.stem}_{args.inject}"
    meta = {"svo": str(args.svo), "inject": args.inject, "depth_mode": args.depth_mode,
            "stride": args.stride, "skip": args.skip, "scale": args.scale,
            "prior_alpha": args.prior_alpha if args.inject == "prior" else None,
            "prior_strength": args.prior_strength if args.inject == "prior_insolver" else None,
            "quality_boost": args.quality_boost if args.inject == "prior_insolver" else None,
            "confidence": args.confidence, "texture_conf": args.texture_conf,
            "depth_max": args.depth_max,
            "config_sdpvo": str(args.config_sdpvo), "seed": args.seed}
    out_dir = make_out_dir(name, meta, args.out_dir)
    print(f"input:   {args.svo}")
    print(f"output:  {out_dir}")
    print(f"inject:  {args.inject}   depth_mode={args.depth_mode}")

    # --- abrir SVO con depth ---
    try:
        depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    except AttributeError:
        print(f"ERROR: DEPTH_MODE desconocido: {args.depth_mode}", file=sys.stderr)
        return 2
    init = sl.InitParameters()
    init.set_from_svo_file(str(args.svo))
    init.svo_real_time_mode = False
    init.depth_mode = depth_mode
    init.coordinate_units = sl.UNIT.METER
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR abriendo SVO: {err}", file=sys.stderr)
        return 3
    cam_res = cam.get_camera_information().camera_configuration.resolution
    print(f"camera:  sn={cam.get_camera_information().serial_number} "
          f"{cam_res.width}×{cam_res.height}")

    # --- imports torch/dpvo (después de abrir cámara) ---
    import torch
    from dpvo.config import cfg
    from dpvo.dpvo import DPVO
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print(f"seed:    {args.seed} (torch+cuda RNG fijada)")
    print(f"torch:   {torch.__version__}  cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA no disponible.", file=sys.stderr)
        return 3
    if args.inject != "off" and not hasattr(DPVO, "_seed_inverse_depth"):
        print("ERROR: el dpvo instalado NO tiene el hook _seed_inverse_depth. "
              "Copiar third_party/S_DPVO/dpvo/dpvo.py a site-packages o reinstalar.",
              file=sys.stderr)
        return 5
    if args.inject == "prior" and not hasattr(DPVO, "_apply_depth_prior"):
        print("ERROR: el dpvo instalado NO tiene _apply_depth_prior (variante C). "
              "Re-correr bootstrap/apply_dpvo_patches.sh para sincronizar el venv.",
              file=sys.stderr)
        return 5
    if args.inject == "prior_insolver" and not hasattr(DPVO, "_ba_python_prior"):
        print("ERROR: el dpvo instalado NO tiene _ba_python_prior (variante C "
              "in-solver). Sincronizar dpvo.py + ba.py al venv.", file=sys.stderr)
        return 5

    # No usar cfg.merge_from_file: yacs abre el YAML con el encoding del
    # locale y en shells no-interactivas (nohup/ssh) cae a ASCII ->
    # UnicodeDecodeError con los comentarios en español (finding
    # 2026-06-11-dpvo-metric-embedded-baseline). Leer UTF-8 explícito:
    cfg.merge_from_other_cfg(cfg.load_cfg(args.config_sdpvo.read_text(encoding="utf-8")))
    cfg.BUFFER_SIZE = args.buffer

    import cv2
    runtime = sl.RuntimeParameters()
    # Sweep de confidence (handoff 2026-06-12): capturar los defaults REALES
    # del SDK (varían entre 5.x) para que la celda "default" tenga baseline
    # honesto, y setear solo lo pedido por CLI.
    sdk_conf_default = runtime.confidence_threshold
    sdk_tex_default = runtime.texture_confidence_threshold
    if args.confidence is not None:
        runtime.confidence_threshold = args.confidence
    if args.texture_conf is not None:
        runtime.texture_confidence_threshold = args.texture_conf
    print(f"depth:   confidence={runtime.confidence_threshold} "
          f"(SDK default {sdk_conf_default})  "
          f"texture_conf={runtime.texture_confidence_threshold} "
          f"(SDK default {sdk_tex_default})  "
          f"depth_max={args.depth_max if args.depth_max is not None else 'inf'}")
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    # skip inicial
    for _ in range(args.skip):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print("ERROR: SVO terminó durante el skip.", file=sys.stderr)
            cam.close()
            return 4

    slam: DPVO | None = None
    n_proc = 0
    frame_no = args.skip - 1   # índice de frame SVO del último grab consumido
    t_start = time.monotonic()
    per_ms: list[float] = []
    valid_fracs: list[float] = []
    capped_fracs: list[float] = []   # fracción descartada por --depth-max
    cand_fracs: list[float] = []     # PATCH_SELECTION depth_valid: fracción de
                                     # candidatos con depth válida (≈valid_depth
                                     # si el mapeo res/4→imagen está alineado)
    s = args.scale

    while True:
        ok = sl.ERROR_CODE.SUCCESS
        for _ in range(args.stride):           # stride: avanza N grabs, procesa 1
            ok = cam.grab(runtime)
            if ok != sl.ERROR_CODE.SUCCESS:
                break
        if ok != sl.ERROR_CODE.SUCCESS:
            break
        frame_no += args.stride                # índice de frame SVO procesado (= timestamp TUM)

        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        bgr = img_mat.get_data()[:, :, :3]                       # BGRA -> BGR
        depth = np.asarray(depth_mat.get_data(), dtype=np.float32)  # HxW metros

        if args.depth_max is not None:
            # Tope "como MAC-VO" (max_depth: 5.0): DESCARTAR el depth lejano,
            # no clampearlo — el parche queda sin ancla (peso 0 en el factor
            # unario vía _seed_inverse_depth). Antes del resize: INTER_NEAREST
            # no mezcla los NaN.
            with np.errstate(invalid="ignore"):
                far = depth > args.depth_max
            capped_fracs.append(float(far.mean()))
            depth = np.where(far, np.float32("nan"), depth)

        image = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        depth_r = cv2.resize(depth, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        image = np.ascontiguousarray(crop16(image))
        depth_r = np.ascontiguousarray(crop16(depth_r))
        intrinsics = np.array([fx * s, fy * s, cx * s, cy * s])
        valid_fracs.append(float(np.isfinite(depth_r).mean()))

        image_t = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intr_t = torch.from_numpy(intrinsics).cuda()
        depth_t = torch.from_numpy(depth_r).cuda() if args.inject != "off" else None

        if slam is None:
            ht, wd = image_t.shape[1], image_t.shape[2]
            print(f"slam:    DPVO ht={ht} wd={wd} inject={args.inject}", flush=True)
            slam = DPVO(cfg, str(args.network), ht=ht, wd=wd)
            slam.depth_inject_mode = args.inject
            slam.depth_prior_alpha = args.prior_alpha if args.inject == "prior" else 0.0
            slam.depth_prior_strength = args.prior_strength if args.inject == "prior_insolver" else 0.0
            slam.depth_quality_boost = args.quality_boost if args.inject == "prior_insolver" else 1.0

        f0 = time.monotonic()
        # prior_insolver corre la BA de Python (autograd.Function CholeskySolver):
        # inference_mode crea "inference tensors" que no se pueden guardar para
        # backward -> usar no_grad (también evita el leak de autograd graph).
        ctx = torch.no_grad() if args.inject == "prior_insolver" else torch.inference_mode()
        with ctx:
            slam(frame_no, image_t, intr_t, depth=depth_t)   # tstamp = índice SVO
        per_ms.append((time.monotonic() - f0) * 1000.0)
        n_proc += 1
        cf = getattr(slam.network.patchify, "last_valid_candidate_frac", None)
        if cf is not None:
            cand_fracs.append(cf)
        if n_proc <= 3 or n_proc % 50 == 0:
            fps = n_proc / (time.monotonic() - t_start)
            cand_str = f" cand_valid={100*cand_fracs[-1]:.0f}%" if cand_fracs else ""
            print(f"  [{n_proc}] fps≈{fps:.2f} last={per_ms[-1]:.0f}ms "
                  f"valid_depth={100*valid_fracs[-1]:.0f}%{cand_str}", flush=True)
        if args.max_frames is not None and n_proc >= args.max_frames:
            break

    cam.close()
    if slam is None:
        print("ERROR: no se procesó ningún frame.", file=sys.stderr)
        return 4

    print("Draining DPVO buffer (12 updates)...")
    with torch.no_grad():
        for _ in range(12):
            slam.update()
    pred_traj = slam.terminate()

    elapsed = time.monotonic() - t_start
    fps = n_proc / elapsed if elapsed > 0 else 0.0

    from dpvo.plot_utils import save_trajectory_tum_format
    traj_path = out_dir / "trajectory.txt"
    save_trajectory_tum_format(pred_traj, str(traj_path))

    # --- post-proceso: filtro de jitter (alta frecuencia) -> path length coherente ---
    # El jitter de DPVO mono es ruido de alta frecuencia (oscila alrededor del
    # camino); un savgol lo quita sin mover escala ni ATE. Es O(N) en CPU (~1 ms),
    # corre tras terminate() -> NO afecta el throughput del SLAM. Ver finding
    # 2026-06-07-dpvo-jitter-highfreq-vs-solver-levers.
    smooth_path = None
    if args.smooth_window and args.smooth_window > 0:
        try:
            from smooth_trajectory import smooth_tum
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from smooth_trajectory import smooth_tum
        traj_arr = np.loadtxt(traj_path)
        if traj_arr.ndim == 2 and traj_arr.shape[0] > args.smooth_window:
            sm = smooth_tum(traj_arr, args.smooth_window, poly=2)
            smooth_path = out_dir / "trajectory_smooth.txt"
            np.savetxt(smooth_path, sm, fmt=["%.6f"] + ["%.9f"] * 7)

            def _plen(p):
                return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            xyz, xyz_s = traj_arr[:, 1:4], sm[:, 1:4]
            net = float(np.linalg.norm(xyz[-1] - xyz[0]))
            if net > 1e-6:
                print(f"smooth:  crudo path/neto={_plen(xyz)/net:.1f} -> "
                      f"suavizado path/neto={_plen(xyz_s)/net:.1f} "
                      f"(w{args.smooth_window}, +{0}ms)  {smooth_path.name}")

    per_arr = np.array(per_ms)
    stats = {
        "svo": str(args.svo), "name": name, "inject": args.inject,
        "prior_alpha": args.prior_alpha if args.inject == "prior" else None,
        "prior_strength": args.prior_strength if args.inject == "prior_insolver" else None,
        "quality_boost": args.quality_boost if args.inject == "prior_insolver" else None,
        "config_sdpvo": str(args.config_sdpvo), "seed": args.seed,
        "smooth_window": args.smooth_window if smooth_path else None,
        "confidence": args.confidence, "texture_conf": args.texture_conf,
        "sdk_confidence_default": sdk_conf_default,
        "sdk_texture_conf_default": sdk_tex_default,
        "depth_max": args.depth_max,
        "depth_capped_frac_mean": round(float(np.mean(capped_fracs)), 4) if capped_fracs else None,
        "depth_mode": args.depth_mode, "n_frames_processed": n_proc,
        "elapsed_sec": round(elapsed, 3), "fps_effective": round(fps, 2),
        "ms_per_frame_p50": round(float(np.percentile(per_arr, 50)), 2) if len(per_arr) else None,
        "ms_per_frame_p95": round(float(np.percentile(per_arr, 95)), 2) if len(per_arr) else None,
        "valid_depth_frac_mean": round(float(np.mean(valid_fracs)), 4) if valid_fracs else None,
        "patch_cand_valid_frac_mean": round(float(np.mean(cand_fracs)), 4) if cand_fracs else None,
        "stride": args.stride, "skip": args.skip, "scale": args.scale, "buffer": args.buffer,
        "calib_path": str(args.calib), "config_sdpvo_path": str(args.config_sdpvo),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"traj:    {traj_path}")
    print(f"stats:   {out_dir / 'stats.json'}")
    print(f"FPS efectivo: {fps:.2f}  ({n_proc} frames en {elapsed:.2f}s)  "
          f"valid_depth medio={100*np.mean(valid_fracs):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
