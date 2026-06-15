#!/usr/bin/env python3
"""eval_ate_tape_gt.py — ATE absoluto contra el GT de cintas (recta 1-D).

El GT de cintas (`scripts/build_gt_tum.py`) es una RECTA perfecta →
covarianza rango-1 → `evo_ape` aborta ("Degenerate covariance rank,
Umeyama alignment is not possible"). Este script implementa la métrica
correcta para ese GT: **ATE along-track**.

Método:
  1. Sincronizar est↔GT por timestamp (frame-index entero, match exacto).
  2. Proyectar la trayectoria estimada sobre su eje principal (PCA sobre
     el tramo sincronizado) → coordenada 1-D s(t) a lo largo de la malla.
  3. Alinear 1-D con el GT x(t):
       - SE(1): solo offset (medias igualadas) + signo. **Métrica honesta
         para un sistema que afirma escala métrica.**
       - Sim(1): offset + escala (mínimos cuadrados). El factor de escala
         resultante es el chequeo de escala contra GT interpolado.
  4. ATE = RMS del residuo along-track. Se reporta además el desvío
     lateral RMS de la estimación respecto a su propia recta (sin GT
     lateral: incluye el balanceo handheld real, no es error puro).

Es exactamente la pregunta del caso de uso (posicionamiento relativo a la
malla): ¿en qué punto A LO LARGO de la malla está la cámara en cada
instante?

Uso:
    .venv/bin/python scripts/eval_ate_tape_gt.py \
        --gt results/gt/gym_video_1_gt_tum.txt \
        --est results/2026-06-09__gym_v1_metric_r1__9c41b983/trajectory.txt \
        [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1:4]


def along_track_series(gt_path: Path, est_path: Path):
    """Series sincronizadas para graficar: (frames, gt_1d, est_1d_alineada SE(1))."""
    t_gt, xyz_gt = load_tum(gt_path)
    t_est, xyz_est = load_tum(est_path)
    common, gi, ei = np.intersect1d(np.round(t_gt).astype(int),
                                    np.round(t_est).astype(int),
                                    return_indices=True)
    g = xyz_gt[gi][:, 0]
    e = xyz_est[ei]
    c = e - e.mean(axis=0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    s = c @ u
    if np.dot(s, g - g.mean()) < 0:
        s = -s
    return common, g, s - s.mean() + g.mean()


def evaluate(gt_path: Path, est_path: Path) -> dict:
    t_gt, xyz_gt = load_tum(gt_path)
    t_est, xyz_est = load_tum(est_path)

    # 1. sync por timestamp exacto (frame-index)
    common, gi, ei = np.intersect1d(np.round(t_gt).astype(int),
                                    np.round(t_est).astype(int),
                                    return_indices=True)
    if len(common) < 10:
        raise SystemExit(f"solo {len(common)} timestamps comunes — revisar convención")
    g = xyz_gt[gi][:, 0]            # GT es 1-D sobre X por construcción
    e = xyz_est[ei]

    # 2. proyección de la estimación sobre su eje principal
    c = e - e.mean(axis=0)
    u = np.linalg.svd(c, full_matrices=False)[2][0]
    s = c @ u
    lateral = c - np.outer(s, u)
    lateral_rms = float(np.sqrt((np.linalg.norm(lateral, axis=1) ** 2).mean()))

    # 3. signo + alineaciones 1-D
    gc = g - g.mean()
    if np.dot(s, gc) < 0:
        s = -s
    # SE(1): solo offset (las medias ya están en 0)
    res_se = s - gc
    ate_se = float(np.sqrt((res_se ** 2).mean()))
    # Sim(1): escala por mínimos cuadrados
    scale = float(np.dot(s, gc) / np.dot(s, s)) if np.dot(s, s) > 0 else float("nan")
    res_sim = scale * s - gc
    ate_sim = float(np.sqrt((res_sim ** 2).mean()))

    # 4. RPE traslacional along-track, segmentos de 1 m de camino GT
    #    (evo_rpe clásico también degenera con GT colineal; mismo principio:
    #    error del desplazamiento estimado vs GT en cada segmento de 1 m).
    cg = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(g)))])
    rpe_errs = []
    j = 0
    for i in range(len(g)):
        while j < len(g) and cg[j] - cg[i] < 1.0:
            j += 1
        if j >= len(g):
            break
        rpe_errs.append(abs((s[j] - s[i]) - (g[j] - g[i])))
    rpe = np.array(rpe_errs) if rpe_errs else np.array([np.nan])

    # 5. deriva: error del desplazamiento neto como % del camino GT total
    net_err = abs((s[-1] - s[0]) - (g[-1] - g[0]))
    gt_path_len = float(cg[-1])
    drift_pct = 100.0 * net_err / gt_path_len if gt_path_len > 0 else float("nan")

    # 6. métricas de loop / endpoint — HONESTAS para secuencias ida-y-vuelta.
    #    El ATE along-track centra (resta media) y proyecta a 1-D → es CIEGO al
    #    colapso de un loop. ‖fin−inicio‖ en unidades CRUDAS (sin corrección de
    #    escala) y la excursión máxima alcanzada SÍ lo reflejan: si el GT cierra
    #    (gt_loop_closure≈0) pero la estimación termina lejos del inicio, o nunca
    #    alcanza la excursión del GT, la trayectoria colapsó. Para secuencias de
    #    una sola pasada (gym) loop_closure≈excursión≈recorrido es lo ESPERADO,
    #    no un error — interpretar solo cuando gt_loop_closure≈0.
    #    Se usan los extremos REALES de cada trayectoria completa (no la
    #    ventana sincronizada) → cifra independiente del solape est↔GT y GT
    #    idéntico para todos los modelos del mismo video.
    start = xyz_est[0]
    loop_closure_m = float(np.linalg.norm(xyz_est[-1] - start))
    max_excursion_m = float(np.linalg.norm(xyz_est - start, axis=1).max())
    gx = xyz_gt[:, 0]
    gt_loop_closure_m = float(abs(gx[-1] - gx[0]))
    gt_excursion_m = float(np.abs(gx - gx[0]).max())
    # ATE 2-D: re-incorpora el desvío lateral como error (válido cuando la
    #    referencia es una recta a distancia ~constante de la malla).
    ate_2d_m = float(np.hypot(ate_se, lateral_rms))

    return {
        "gt": str(gt_path), "est": str(est_path),
        "n_synced": int(len(common)),
        "frames": [int(common[0]), int(common[-1])],
        "ate_along_se1_m": round(ate_se, 4),
        "ate_along_sim1_m": round(ate_sim, 4),
        "scale_gt_over_est": round(scale, 4),
        "max_err_se1_m": round(float(np.abs(res_se).max()), 4),
        "lateral_rms_m": round(lateral_rms, 4),
        "gt_span_m": round(float(g.max() - g.min()), 3),
        "gt_path_m": round(gt_path_len, 3),
        "rpe_t_1m_mean_m": round(float(np.nanmean(rpe)), 4),
        "rpe_t_1m_p95_m": round(float(np.nanpercentile(rpe, 95)), 4),
        "drift_net_err_m": round(float(net_err), 4),
        "drift_pct": round(float(drift_pct), 2),
        "ate_2d_se1_m": round(ate_2d_m, 4),
        "loop_closure_m": round(loop_closure_m, 4),
        "max_excursion_m": round(max_excursion_m, 4),
        "gt_loop_closure_m": round(gt_loop_closure_m, 4),
        "gt_excursion_m": round(gt_excursion_m, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--est", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    r = evaluate(args.gt, args.est)
    print(f"{Path(args.est).parent.name}: "
          f"ATE_along SE(1)={r['ate_along_se1_m']:.3f} m  "
          f"Sim(1)={r['ate_along_sim1_m']:.3f} m  "
          f"escala GT/est={r['scale_gt_over_est']:.3f}  "
          f"max={r['max_err_se1_m']:.3f} m  "
          f"lateral_rms={r['lateral_rms_m']:.3f} m  "
          f"(n={r['n_synced']}, GT {r['gt_span_m']} m)")
    print(f"    loop: ‖fin−inicio‖={r['loop_closure_m']:.3f} m "
          f"(GT {r['gt_loop_closure_m']:.3f} m)  "
          f"excursión={r['max_excursion_m']:.3f}/{r['gt_excursion_m']:.3f} m  "
          f"ATE_2D={r['ate_2d_se1_m']:.3f} m")
    if args.json:
        args.json.write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
