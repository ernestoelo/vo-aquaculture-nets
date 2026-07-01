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
    p.add_argument("--stereo-sparse", action="store_true",
                   help="P2 Hito 3: en vez del depth DENSO del SDK, triangula "
                        "DISPERSO LEFT↔RIGHT por correspondencia (matching de "
                        "bloques 1-D + chequeo de unicidad y consistencia L-R) y "
                        "siembra solo esos píxeles. Fuente LIMPIA inmune al 'queso "
                        "suizo' y a la malla repetitiva. Abre el SVO con "
                        "DEPTH_MODE.NONE (no paga el motor denso). Default off.")
    p.add_argument("--stereo-win", type=int, default=9,
                   help="Tamaño de la ventana de correlación (impar) del matching "
                        "estéreo disperso (solo --stereo-sparse). Default 9.")
    p.add_argument("--stereo-max-kp", type=int, default=1500,
                   help="Máximo de puntos de interés (goodFeaturesToTrack) a "
                        "triangular por frame (solo --stereo-sparse). Default 1500.")
    p.add_argument("--stereo-min-ncc", type=float, default=0.6,
                   help="Umbral mínimo de correlación NCC para aceptar un match "
                        "(solo --stereo-sparse). Default 0.6.")
    p.add_argument("--stereo-z-max", type=float, default=20.0,
                   help="Profundidad máxima (m) triangulable; acota el rango de "
                        "disparidad (solo --stereo-sparse). Default 20.0.")
    p.add_argument("--stereo-validate", action="store_true",
                   help="P2 Hito 3 (híbrido recomendado): mantiene el depth DENSO "
                        "del SDK como prior, pero VALIDA/refina cada parche con un "
                        "matching estéreo ACOTADO a ±--stereo-disp-halfwin px "
                        "alrededor de la disparidad del SDK. Rompe el aliasing (la "
                        "ventana es < período de la malla) y DESCARTA el 'queso "
                        "suizo' (los píxeles cuyo denso es espurio no tienen match "
                        "consistente en la ventana). Requiere el motor denso. "
                        "Excluyente con --stereo-sparse. Default off.")
    p.add_argument("--stereo-disp-halfwin", type=float, default=4.0,
                   help="Semi-ventana de búsqueda de disparidad (px) alrededor del "
                        "prior del SDK (solo --stereo-validate). Debe ser < período "
                        "de la textura repetitiva para no re-introducir aliasing. "
                        "Default 4.0.")
    p.add_argument("--plane-anchor", action="store_true",
                   help="P1 Hito 3: ajusta el plano dominante (RANSAC) de la nube "
                        "estéreo por frame y reemplaza el depth de los píxeles "
                        "INLIER por el del plano (rayo-plano), des-ruidando el "
                        "'queso suizo'; el resto queda sin ancla (NaN). Inmune a "
                        "la ambigüedad de la malla repetitiva. Default off.")
    p.add_argument("--plane-inlier-thresh", type=float, default=0.05,
                   help="Distancia punto-plano (m) para contar como inlier del "
                        "RANSAC (solo --plane-anchor). Default 0.05 m.")
    p.add_argument("--plane-min-inliers", type=float, default=0.2,
                   help="Fracción mínima de inliers para aceptar el plano (solo "
                        "--plane-anchor); si no la alcanza, usa el depth crudo "
                        "(degradación segura). Default 0.2.")
    p.add_argument("--plane-ransac-iters", type=int, default=200,
                   help="Iteraciones de RANSAC del plano (solo --plane-anchor "
                        "o --plane-insolver). Default 200.")
    p.add_argument("--plane-insolver", action="store_true",
                   help="P2 Hito 3: restricción de coplanaridad BLANDA del plano "
                        "del net DENTRO de la BA métrica (factor `π·X_world≈0` por "
                        "parche), JUNTO al prior de depth (NO en vez de él). El "
                        "depth ancla la ESCALA; el plano ancla el LATERAL "
                        "(anti-abombamiento del loop). Requiere --inject "
                        "prior_insolver. El plano se reestima por RANSAC sobre la "
                        "nube cam-local (validada si --stereo-validate, si no la "
                        "densa) cada --plane-refit-every frames. Default off.")
    p.add_argument("--plane-strength", type=float, default=500.0,
                   help="Fuerza del factor de plano in-solver relativa al término "
                        "de datos (como --prior-strength). Default 500.")
    p.add_argument("--plane-trim", type=float, default=0.5,
                   help="Trim del factor de plano por distancia euclídea al plano "
                        "(m): descarta parches a más de N m (foreground fuera del "
                        "net) para no forzarlos a la malla. 0 = sin trim. Default 0.5.")
    p.add_argument("--plane-refit-every", type=int, default=3,
                   help="Reestima el plano cada N frames procesados (alternancia; "
                        "entre refits la BA reusa el último plano válido). Default 3.")
    p.add_argument("--plane-subsample", type=int, default=3000,
                   help="Máximo de puntos de la nube para el RANSAC del plano "
                        "in-solver (sub-muestreo → costo acotado). Default 3000.")
    p.add_argument("--plane-mode", choices=["frozen", "window"], default="frozen",
                   help="Fuente del plano in-solver. 'frozen' (P2): un plano "
                        "per-frame cam-local del SDK, transformado a mundo y "
                        "congelado entre refits (global). 'window' (B1): plano "
                        "LOCAL re-ajustado por RANSAC sobre los puntos 3-D de los "
                        "parches de la ventana de BA, cada paso (respeta la órbita; "
                        "no usa depth del SDK para el plano). Default frozen.")
    p.add_argument("--plane-vertical", action="store_true",
                   help="Gap 2 Hito 3: restringe el RANSAC del plano de ventana (B1) "
                        "a planos VERTICALES (normal ⊥ gravedad). La red de la jaula "
                        "cuelga vertical → quita 1 DOF a la normal y elimina la "
                        "varianza que hundió a B1. Usa SOLO el acelerómetro (gravedad "
                        "mundo) → NO necesita gyro → corre en video_4 (gyro muerto). "
                        "Requiere --plane-insolver --plane-mode window y --plane-grav-dir.")
    p.add_argument("--plane-grav-dir", type=Path, default=None,
                   help="Dir con imu.csv/meta.json (extract_imu_svo.py) del que se "
                        "computa la gravedad mundo para --plane-vertical. Solo lee el "
                        "acelerómetro (no el gyro). Independiente de --imu (tight).")
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
    # --- tight coupling IMU (estrategia B Hito 3) ---
    p.add_argument("--imu", type=Path, default=None,
                   help="Dir con imu.csv/frame_times.csv/meta.json de "
                        "extract_imu_svo.py. Habilita el factor inercial de "
                        "preintegración DENTRO de la BA (tight coupling). Requiere "
                        "--inject prior_insolver (corre la BA de Python).")
    p.add_argument("--imu-strength", type=float, default=1.0,
                   help="Fuerza global del factor IMU relativa al término visual "
                        "(como prior-strength). 0=off; 1=moderado; 10=fuerte. Default 1.")
    p.add_argument("--imu-sig-g", type=float, default=0.01,
                   help="Ruido del gyro (peso del residual de rotación r_ΔR). Default 0.01.")
    p.add_argument("--imu-sig-a", type=float, default=0.2,
                   help="Ruido del acelerómetro (peso de r_Δv/r_Δp; controla cuánto "
                        "tira la doble integración de la escala). Default 0.2.")
    p.add_argument("--imu-v-max", type=float, default=0.0,
                   help="GUARD: cap físico de ‖v‖ por keyframe (m/s). REFUTADO como "
                        "restaurador de escala (2026-06-19: clampar NO baja la escala "
                        "inflada, el optimizador rutea la inflación a las poses; incluso "
                        "empeora). Útil solo como DETECTOR de divergencia (miles de "
                        "clamps = run divergiendo). 0=off (default).")
    p.add_argument("--imu-max-step", type=float, default=2.0,
                   help="GUARD: cap de traslación de pose por paso de BA (m). Si se "
                        "supera (o el solve es no-finito), se RECHAZA el factor IMU y "
                        "se cae a la BA visual pura ese paso. 0=off. Default 2.")
    p.add_argument("--imu-v-reg", type=float, default=10.0,
                   help="ROBUSTEZ (default ON): Tikhonov in-solver sobre ‖v‖ "
                        "(½·v_reg·‖v‖² dentro del GN aumentado). Remueve la dirección "
                        "casi-nula (Δpos⊕v grandes) que infla la escala en muestreos "
                        "patológicos → 9/9 gym convergen (vs 8/9 sin él, v1 seed2 "
                        "divergía ×7.6). Calibrado para --imu-strength 10 (v_reg≪info "
                        "IMU ~5000: regulariza solo el modo mal condicionado). >100 "
                        "sobre-corrige (deflacta). 0=off. Finding 2026-06-19-robustness.")
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


def load_imu(imu_dir: Path):
    """Carga el stream IMU + el mapa frame->ts (extract_imu_svo.py) y rota acc/gyro
    al frame CÁMARA con la extrínseca de fábrica R_cam<-imu. acc en m/s², gyro en
    rad/s (ya convertido por extract_imu_svo, gotcha deg/s 2026-06-19).

    Devuelve (ts (M,) int64, acc (M,3) f32, gyr (M,3) f32, gi2ts dict grab->ts_ns).
    """
    imu = np.loadtxt(imu_dir / "imu.csv", delimiter=",", skiprows=1)
    ft = np.loadtxt(imu_dir / "frame_times.csv", delimiter=",", skiprows=1).astype(np.int64)
    meta = json.loads((imu_dir / "meta.json").read_text())
    R_ci = np.eye(3)
    if meta.get("camera_imu_transform"):
        R_ci = np.array(meta["camera_imu_transform"])[:3, :3]
    ts = imu[:, 0].astype(np.int64)
    acc = (R_ci @ imu[:, 1:4].T).T.astype(np.float32)
    gyr = (R_ci @ imu[:, 4:7].T).T.astype(np.float32)
    gi2ts = {int(g): int(t) for g, t in ft}
    return ts, acc, gyr, gi2ts


def gravity_world(ts: np.ndarray, acc_cam: np.ndarray, t0_ns: int,
                  window_s: float = 0.5) -> np.ndarray:
    """Gravedad en el mundo (= frame DPVO 0, identidad) de la fuerza específica
    media en la ventana inicial: g_world = -mean(acc_cam[t0:t0+win]) renormalizada
    a 9.81 m/s² (misma convención que el EKF loose-coupling)."""
    win = (ts >= t0_ns) & (ts <= t0_ns + window_s * 1e9)
    f_bar = acc_cam[win].mean(axis=0) if win.sum() else acc_cam[:50].mean(axis=0)
    g = -f_bar
    n = np.linalg.norm(g)
    return (g / n * 9.81).astype(np.float32) if n > 1e-6 else np.array([0, 0, -9.81], np.float32)


def main() -> int:
    args = parse_args()
    if args.stereo_sparse and args.stereo_validate:
        print("ERROR: --stereo-sparse y --stereo-validate son excluyentes.", file=sys.stderr)
        return 2
    if args.plane_insolver and args.inject != "prior_insolver":
        print("ERROR: --plane-insolver requiere --inject prior_insolver "
              "(el factor de plano entra junto al prior de depth in-solver).",
              file=sys.stderr)
        return 2
    if args.plane_insolver and args.stereo_sparse:
        print("ERROR: --plane-insolver necesita una nube densa para el plano; "
              "es excluyente con --stereo-sparse (DEPTH_MODE.NONE).", file=sys.stderr)
        return 2
    if args.plane_vertical:
        if not (args.plane_insolver and args.plane_mode == "window"):
            print("ERROR: --plane-vertical requiere --plane-insolver --plane-mode "
                  "window (la verticalidad restringe el RANSAC del plano de ventana).",
                  file=sys.stderr)
            return 2
        if args.plane_grav_dir is None or not args.plane_grav_dir.exists():
            print("ERROR: --plane-vertical requiere --plane-grav-dir DIR con "
                  "imu.csv/meta.json (la gravedad mundo viene del acelerómetro).",
                  file=sys.stderr)
            return 2
    if args.imu is not None:
        if args.inject != "prior_insolver":
            print("ERROR: --imu requiere --inject prior_insolver (el factor IMU "
                  "entra en la BA de Python).", file=sys.stderr)
            return 2
        if not args.imu.exists():
            print(f"ERROR: no existe el dir IMU {args.imu}", file=sys.stderr)
            return 2
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
            "depth_max": args.depth_max, "stereo_sparse": args.stereo_sparse,
            "stereo_validate": args.stereo_validate,
            "plane_insolver": args.plane_insolver,
            "plane_strength": args.plane_strength if args.plane_insolver else None,
            "plane_trim": args.plane_trim if args.plane_insolver else None,
            "plane_mode": args.plane_mode if args.plane_insolver else None,
            "plane_vertical": args.plane_vertical if args.plane_insolver else None,
            "config_sdpvo": str(args.config_sdpvo), "seed": args.seed}
    out_dir = make_out_dir(name, meta, args.out_dir)
    print(f"input:   {args.svo}")
    print(f"output:  {out_dir}")
    print(f"inject:  {args.inject}   depth_mode={args.depth_mode}")

    # --- abrir SVO con depth ---
    # En modo stereo-sparse no se usa el motor denso del SDK (triangulamos
    # nosotros LEFT↔RIGHT) → abrir con DEPTH_MODE.NONE ahorra ese cómputo.
    try:
        depth_mode = (sl.DEPTH_MODE.NONE if args.stereo_sparse
                      else getattr(sl.DEPTH_MODE, args.depth_mode))
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
    cam_info = cam.get_camera_information()
    cam_res = cam_info.camera_configuration.resolution
    # Baseline real de ESTA cámara (varía por SN: gym 34111953 ≈ 0.11995 m,
    # video_4 36802538 ≈ 0.11978 m) — del SDK, nunca hardcodear.
    baseline = float(cam_info.camera_configuration.calibration_parameters
                     .get_camera_baseline())
    print(f"camera:  sn={cam_info.serial_number} "
          f"{cam_res.width}×{cam_res.height}  baseline={baseline:.6f} m")

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
    img_mat, depth_mat, right_mat = sl.Mat(), sl.Mat(), sl.Mat()

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
    plane_inlier_fracs: list[float] = []   # P1: inlier_frac del plano por frame
    plane_fitted_n = 0                     # frames donde el RANSAC convergió
    stereo_valid_n: list[int] = []         # P2: nº de puntos triangulados/frame
    stereo_match_rates: list[float] = []   # P2: tasa de matches válidos/kp
    plane_rng = np.random.default_rng(args.seed if args.seed is not None else 0)
    plane_insolver_fitted_n = 0            # P2: frames donde se reestimó el plano
    plane_insolver_inl_fracs: list[float] = []
    last_plane_cam = None                  # P2: último plano cam-local válido
    if args.plane_anchor:
        from plane_anchor import plane_denoise_depth, plane_fill_over_mask
    if args.plane_insolver:
        from plane_anchor import backproject, fit_plane_ransac
    if args.stereo_sparse:
        from stereo_triangulate import sparse_stereo_depth
    if args.stereo_validate:
        from stereo_triangulate import stereo_validate_dense_depth
    s = args.scale

    # --- tight coupling: cargar stream IMU + gravedad mundo (una vez) ---
    imu_ts = imu_acc = imu_gyr = None
    imu_gi2ts: dict[int, int] = {}
    imu_g_world = None
    if args.imu is not None:
        imu_ts, imu_acc, imu_gyr, imu_gi2ts = load_imu(args.imu)
        first_fno = args.skip - 1 + args.stride        # primer grab procesado
        t0_ns = imu_gi2ts.get(first_fno, int(imu_ts[0]))
        imu_g_world = gravity_world(imu_ts, imu_acc, t0_ns)
        print(f"imu:     {len(imu_ts)} muestras  g_world={imu_g_world.round(2)} "
              f"|g|={np.linalg.norm(imu_g_world):.2f}  strength={args.imu_strength}")

    # --- Gap 2: gravedad mundo SOLO para el plano vertical (no tight, no gyro) ---
    plane_grav_world = None
    if args.plane_vertical:
        g_ts, g_acc, _g_gyr, g_gi2ts = load_imu(args.plane_grav_dir)
        first_fno = args.skip - 1 + args.stride
        t0_ns = g_gi2ts.get(first_fno, int(g_ts[0]))
        plane_grav_world = gravity_world(g_ts, g_acc, t0_ns)
        print(f"plane:   VERTICAL (gravedad accel) g_world={plane_grav_world.round(2)} "
              f"|g|={np.linalg.norm(plane_grav_world):.2f}")

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
        bgr = img_mat.get_data()[:, :, :3]                       # BGRA -> BGR
        image = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        image = np.ascontiguousarray(crop16(image))
        intrinsics = np.array([fx * s, fy * s, cx * s, cy * s])
        plane_src = None      # nube LIMPIA p/ fit del plano in-solver (validada);
                              # si queda None se usa depth_r (denso) como fuente.

        if args.stereo_sparse:
            # --- P2 Hito 3: depth DISPERSO por triangulación LEFT↔RIGHT ---
            # Fuente LIMPIA (matching de bloques + chequeo unicidad/L-R) en vez
            # del depth denso "queso suizo". Se triangula sobre el par ya
            # redimensionado y crop16 → misma grilla que la imagen de DPVO.
            cam.retrieve_image(right_mat, sl.VIEW.RIGHT)
            rbgr = right_mat.get_data()[:, :, :3]
            right_img = cv2.resize(rbgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            right_img = np.ascontiguousarray(crop16(right_img))
            lg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rg = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            depth_r, sinfo = sparse_stereo_depth(
                lg, rg, float(intrinsics[0]), baseline,
                z_max=args.stereo_z_max, win=args.stereo_win,
                max_kp=args.stereo_max_kp, min_ncc=args.stereo_min_ncc)
            depth_r = np.ascontiguousarray(depth_r)
            stereo_valid_n.append(sinfo["n_valid"])
            stereo_match_rates.append(sinfo["match_rate"])
        else:
            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
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
            depth_r = cv2.resize(depth, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            depth_r = np.ascontiguousarray(crop16(depth_r))

            if args.stereo_validate:
                # --- P2 Hito 3 (híbrido): valida/refina el denso con estéreo
                # acotado al prior del SDK → rompe aliasing + mata el queso suizo.
                cam.retrieve_image(right_mat, sl.VIEW.RIGHT)
                rbgr = right_mat.get_data()[:, :, :3]
                right_img = np.ascontiguousarray(crop16(
                    cv2.resize(rbgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)))
                lg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rg = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
                dense_mask = depth_r                  # SDK denso = máscara de la malla
                validated, sinfo = stereo_validate_dense_depth(
                    lg, rg, dense_mask, float(intrinsics[0]), baseline,
                    disp_halfwin=args.stereo_disp_halfwin, win=args.stereo_win,
                    min_ncc=args.stereo_min_ncc, max_kp=args.stereo_max_kp)
                stereo_valid_n.append(sinfo["n_valid"])
                stereo_match_rates.append(sinfo["keep_rate"])
                if args.plane_anchor:
                    # densificar el plano LIMPIO (validado) sobre la máscara densa
                    # → geometría limpia + cobertura suficiente para anclar la BA
                    # (la nube validada sola es demasiado dispersa → colapsa escala).
                    depth_r, pinfo = plane_fill_over_mask(
                        validated, dense_mask, intrinsics,
                        thresh=args.plane_inlier_thresh,
                        iters=args.plane_ransac_iters,
                        min_inliers_frac=args.plane_min_inliers, rng=plane_rng)
                    depth_r = np.ascontiguousarray(depth_r)
                    if pinfo["fitted"]:
                        plane_fitted_n += 1
                        plane_inlier_fracs.append(pinfo["inlier_frac"])
                elif args.plane_insolver:
                    # P2 in-solver DESACOPLADO: el prior de depth sigue usando el
                    # DENSO (ancla la escala uw ~1.18); la nube VALIDADA (limpia,
                    # sin aliasing ni queso suizo) alimenta SOLO el fit del plano
                    # lateral. NO sobrescribir depth_r (eso colapsaría la escala).
                    plane_src = np.ascontiguousarray(validated)
                else:
                    depth_r = np.ascontiguousarray(validated)

        # --- P1 Hito 3: anclaje por plano (des-ruida el depth antes de la siembra) ---
        # Ajusta el plano dominante de la nube (RANSAC) y reemplaza el depth de los
        # inliers por el del plano; el resto queda NaN (sin ancla). Inmune a la
        # ambigüedad de la malla repetitiva. Si no converge, deja el depth crudo.
        # (Con --stereo-validate el plano ya se densificó arriba con la nube LIMPIA
        # → no re-aplicar aquí sobre el denso basura.)
        if args.plane_anchor and not args.stereo_validate:
            depth_r, pinfo = plane_denoise_depth(
                depth_r, intrinsics, thresh=args.plane_inlier_thresh,
                iters=args.plane_ransac_iters,
                min_inliers_frac=args.plane_min_inliers, rng=plane_rng)
            depth_r = np.ascontiguousarray(depth_r)
            if pinfo["fitted"]:
                plane_fitted_n += 1
                plane_inlier_fracs.append(pinfo["inlier_frac"])

        valid_fracs.append(float(np.isfinite(depth_r).mean()))

        # --- P2 Hito 3: reestimar el plano cam-local para el factor de plano
        # in-solver. RANSAC sobre la nube del frame ACTUAL (la cleanest disponible:
        # `validated` si --stereo-validate, si no la densa) sub-muestreada. Se pasa
        # SOLO en frames de refit (el plano es cam-local a SU frame); DPVO lo
        # transforma a mundo con la pose de ese frame y lo reusa hasta el próximo
        # refit. No-refit / RANSAC fallido → plane=None (DPVO mantiene el anterior).
        plane_cam = None
        if (args.plane_insolver and args.plane_mode != "window"
                and (n_proc % args.plane_refit_every == 0)):
            src = plane_src if plane_src is not None else depth_r   # validada > denso
            pts, _ = backproject(src, float(intrinsics[0]), float(intrinsics[1]),
                                 float(intrinsics[2]), float(intrinsics[3]))
            if len(pts) >= 3:
                if len(pts) > args.plane_subsample:
                    sel = plane_rng.choice(len(pts), size=args.plane_subsample, replace=False)
                    pts = pts[sel]
                res = fit_plane_ransac(pts, thresh=args.plane_inlier_thresh,
                                       iters=args.plane_ransac_iters,
                                       min_inliers_frac=args.plane_min_inliers, rng=plane_rng)
                if res is not None:
                    nrm, off, inl = res
                    plane_cam = np.array([nrm[0], nrm[1], nrm[2], off], dtype=np.float64)
                    plane_insolver_fitted_n += 1
                    plane_insolver_inl_fracs.append(float(inl.mean()))

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
            slam.plane_strength = args.plane_strength if args.plane_insolver else 0.0
            slam.plane_trim = args.plane_trim if args.plane_insolver else 0.0
            slam.plane_mode = args.plane_mode      # 'frozen' (P2) | 'window' (B1)
            if args.plane_vertical:
                slam.plane_vertical = True
                slam.plane_grav = torch.from_numpy(plane_grav_world).cuda()
                print(f"slam:    plano VERTICAL ON (normal ⊥ gravedad)", flush=True)
            if args.imu is not None:
                slam.imu_tight = True
                slam.imu_ts = torch.from_numpy(imu_ts).cuda()
                slam.imu_acc = torch.from_numpy(imu_acc).cuda()
                slam.imu_gyr = torch.from_numpy(imu_gyr).cuda()
                slam.imu_g = torch.from_numpy(imu_g_world).cuda()
                slam.imu_sig_g = args.imu_sig_g
                slam.imu_sig_a = args.imu_sig_a
                slam.imu_strength = args.imu_strength
                slam.imu_v_max = args.imu_v_max
                slam.imu_max_step = args.imu_max_step
                slam.imu_v_reg = args.imu_v_reg
                print(f"slam:    IMU tight coupling ON (strength={args.imu_strength} "
                      f"sig_g={args.imu_sig_g} sig_a={args.imu_sig_a} "
                      f"v_max={args.imu_v_max} max_step={args.imu_max_step} "
                      f"v_reg={args.imu_v_reg})", flush=True)

        f0 = time.monotonic()
        # prior_insolver corre la BA de Python (autograd.Function CholeskySolver):
        # inference_mode crea "inference tensors" que no se pueden guardar para
        # backward -> usar no_grad (también evita el leak de autograd graph).
        ctx = torch.no_grad() if args.inject == "prior_insolver" else torch.inference_mode()
        frame_ts = imu_gi2ts.get(frame_no) if args.imu is not None else None
        with ctx:
            slam(frame_no, image_t, intr_t, depth=depth_t, plane=plane_cam,
                 frame_ts=frame_ts)   # tstamp = índice SVO; frame_ts = ns real
        per_ms.append((time.monotonic() - f0) * 1000.0)
        n_proc += 1
        cf = getattr(slam.network.patchify, "last_valid_candidate_frac", None)
        if cf is not None:
            cand_fracs.append(cf)
        if n_proc <= 3 or n_proc % 50 == 0:
            fps = n_proc / (time.monotonic() - t_start)
            cand_str = f" cand_valid={100*cand_fracs[-1]:.0f}%" if cand_fracs else ""
            plane_str = (f" plane_inl={100*plane_inlier_fracs[-1]:.0f}%"
                         if args.plane_anchor and plane_inlier_fracs else "")
            if args.plane_insolver and plane_insolver_inl_fracs:
                plane_str += f" pl_isolv_inl={100*plane_insolver_inl_fracs[-1]:.0f}%"
            stereo_str = (f" stereo_pts={stereo_valid_n[-1]} "
                          f"keep={100*stereo_match_rates[-1]:.0f}%"
                          if (args.stereo_sparse or args.stereo_validate) and stereo_valid_n else "")
            print(f"  [{n_proc}] fps≈{fps:.2f} last={per_ms[-1]:.0f}ms "
                  f"valid_depth={100*valid_fracs[-1]:.1f}%{cand_str}{plane_str}{stereo_str}",
                  flush=True)
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
        "plane_anchor": args.plane_anchor,
        "plane_inlier_thresh": args.plane_inlier_thresh if args.plane_anchor else None,
        "plane_fitted_frac": round(plane_fitted_n / n_proc, 4) if (args.plane_anchor and n_proc) else None,
        "plane_inlier_frac_mean": round(float(np.mean(plane_inlier_fracs)), 4) if plane_inlier_fracs else None,
        "plane_insolver": args.plane_insolver,
        "plane_strength": args.plane_strength if args.plane_insolver else None,
        "plane_trim": args.plane_trim if args.plane_insolver else None,
        "plane_refit_every": args.plane_refit_every if args.plane_insolver else None,
        "plane_insolver_fitted_n": plane_insolver_fitted_n if args.plane_insolver else None,
        "plane_insolver_inl_frac_mean": (round(float(np.mean(plane_insolver_inl_fracs)), 4)
                                         if plane_insolver_inl_fracs else None),
        "stereo_sparse": args.stereo_sparse,
        "stereo_validate": args.stereo_validate,
        "stereo_disp_halfwin": args.stereo_disp_halfwin if args.stereo_validate else None,
        "baseline_m": round(baseline, 6),
        "stereo_pts_mean": round(float(np.mean(stereo_valid_n)), 1) if stereo_valid_n else None,
        "stereo_keep_rate_mean": round(float(np.mean(stereo_match_rates)), 4) if stereo_match_rates else None,
        "stereo_win": args.stereo_win if (args.stereo_sparse or args.stereo_validate) else None,
        "stereo_max_kp": args.stereo_max_kp if (args.stereo_sparse or args.stereo_validate) else None,
        "stereo_min_ncc": args.stereo_min_ncc if (args.stereo_sparse or args.stereo_validate) else None,
        "imu": str(args.imu) if args.imu is not None else None,
        "imu_strength": args.imu_strength if args.imu is not None else None,
        "imu_v_max": args.imu_v_max if args.imu is not None else None,
        "imu_max_step": args.imu_max_step if args.imu is not None else None,
        "imu_v_reg": args.imu_v_reg if args.imu is not None else None,
        "imu_n_clamp": int(getattr(slam, "imu_n_clamp", 0)) if args.imu is not None else None,
        "imu_n_reject": int(getattr(slam, "imu_n_reject", 0)) if args.imu is not None else None,
        "stride": args.stride, "skip": args.skip, "scale": args.scale, "buffer": args.buffer,
        "calib_path": str(args.calib), "config_sdpvo_path": str(args.config_sdpvo),
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"traj:    {traj_path}")
    print(f"stats:   {out_dir / 'stats.json'}")
    print(f"FPS efectivo: {fps:.2f}  ({n_proc} frames en {elapsed:.2f}s)  "
          f"valid_depth medio={100*np.mean(valid_fracs):.0f}%")
    if args.imu is not None:
        print(f"IMU guard: vels acotadas={getattr(slam, 'imu_n_clamp', 0)}  "
              f"pasos rechazados (fallback visual)={getattr(slam, 'imu_n_reject', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
