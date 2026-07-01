#!/usr/bin/env python3
"""plane_anchor.py — Ajuste del plano dominante (RANSAC) sobre la nube estéreo
del ZED para des-ruidar el mapa de depth antes de la siembra métrica de DPVO.

Mejora secundaria del Hito 3 (anclaje por PLANO, reencuadre del guía
2026-06-15): la malla repetitiva produce cientos de mediciones coplanares → un
plano se ajusta del AGREGADO y es **inmune a la ambigüedad del patrón
repetitivo** (no necesita correspondencias correctas, solo el plano).

Esta es la **etapa P1** ("el plano des-ruida el depth prior"): se ajusta el
plano dominante de la nube por keyframe y se reemplaza la profundidad de los
píxeles INLIER por la del plano (intersección rayo-plano), filtrando el "queso
suizo" del depth denso del SDK; los píxeles fuera del plano quedan **sin ancla**
(NaN → peso 0 en el factor unario). Vive del lado del runner → **NO toca el
submódulo `third_party/S_DPVO`** (gitlink).

P2 (no implementado aquí) llevará el plano a una restricción de coplanaridad
DENTRO de la BA métrica (acople global a un plano del mundo). Ver handoff
`docs/handoffs/2026-06-18-hito3-plane-constraints-metric-ba.md` y finding
`docs/findings/2026-06-07-zed-dense-depth-sparse-stereo-triangulation-direction.md`.
"""

from __future__ import annotations

import numpy as np


def backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """Retroproyecta los píxeles válidos del mapa de depth a 3-D (cámara-local).

    Retorna `points` (K, 3) en metros y `flat_idx` (K,) con los índices planos
    (`v*W + u`) de los píxeles válidos, para mapear de vuelta a la imagen.
    """
    H, W = depth.shape
    vs, us = np.nonzero(np.isfinite(depth) & (depth > 1e-3))
    Z = depth[vs, us].astype(np.float64)
    X = (us - cx) / fx * Z
    Y = (vs - cy) / fy * Z
    pts = np.stack([X, Y, Z], axis=1)
    return pts, vs * W + us


def fit_plane_ransac(points: np.ndarray, thresh: float = 0.05, iters: int = 200,
                     min_inliers_frac: float = 0.2, rng=None):
    """RANSAC del plano dominante `n·X + d = 0` (|n|=1) sobre `points` (K, 3).

    Devuelve `(n, d, inlier_mask)` o `None` si no alcanza `min_inliers_frac`.
    Refit final por SVD (mínimos cuadrados) sobre los inliers del mejor modelo.
    `thresh` en metros (distancia punto-plano para contar como inlier).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    K = len(points)
    if K < 3:
        return None

    best_inliers = None
    best_count = 0
    for _ in range(iters):
        i0, i1, i2 = rng.choice(K, size=3, replace=False)
        n = np.cross(points[i1] - points[i0], points[i2] - points[i0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:          # 3 puntos colineales → hipótesis degenerada
            continue
        n = n / nn
        d = -float(n @ points[i0])
        inliers = np.abs(points @ n + d) < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_inliers is None or best_count < max(3, int(min_inliers_frac * K)):
        return None

    # refit por SVD: la normal es el vector singular menor de la nube centrada
    P = points[best_inliers]
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]
    n = n / np.linalg.norm(n)
    d = -float(n @ c)
    inliers = np.abs(points @ n + d) < thresh
    return n, d, inliers


def plane_denoise_depth(depth: np.ndarray, intrinsics, thresh: float = 0.05,
                        iters: int = 200, min_inliers_frac: float = 0.2, rng=None):
    """Ajusta el plano dominante del `depth` y devuelve `(depth_out, info)`.

    `depth_out` (H, W float32): copia donde los píxeles INLIER del plano reciben
    la profundidad del plano (intersección rayo-plano) y **el resto queda NaN**
    (sin ancla). Si el RANSAC no converge (pocos puntos / pocos inliers), devuelve
    el `depth` original intacto con `info['fitted']=False` (degradación segura).

    `info`: dict con `fitted`, `n_valid`, `inlier_frac` (sobre los válidos),
    `inlier_px`, `normal`, `offset`.
    """
    fx, fy, cx, cy = (float(intrinsics[0]), float(intrinsics[1]),
                      float(intrinsics[2]), float(intrinsics[3]))
    H, W = depth.shape
    pts, flat_idx = backproject(depth, fx, fy, cx, cy)
    info = {"fitted": False, "n_valid": int(len(pts)), "inlier_frac": 0.0,
            "inlier_px": 0, "normal": None, "offset": None}
    if len(pts) < 3:
        return depth, info

    res = fit_plane_ransac(pts, thresh=thresh, iters=iters,
                           min_inliers_frac=min_inliers_frac, rng=rng)
    if res is None:
        return depth, info
    n, d, inliers = res

    # intersección rayo-plano por píxel inlier: para el rayo r=[(u-cx)/fx,(v-cy)/fy,1]
    # el punto del plano es t·r con n·(t·r)+d=0 → t = -d/(n·r); su profundidad Z = t.
    inl_flat = flat_idx[inliers]
    vs = inl_flat // W
    us = inl_flat % W
    rx = (us - cx) / fx
    ry = (vs - cy) / fy
    denom = n[0] * rx + n[1] * ry + n[2]
    Zp = np.full(len(inl_flat), np.nan, dtype=np.float64)
    ok = np.abs(denom) > 1e-6
    Zp[ok] = -d / denom[ok]
    keep = ok & (Zp > 1e-3)

    out = np.full((H, W), np.nan, dtype=np.float32)
    out[vs[keep], us[keep]] = Zp[keep].astype(np.float32)
    info.update(fitted=True, inlier_frac=float(inliers.mean()),
                inlier_px=int(inliers.sum()),
                normal=[float(x) for x in n], offset=float(d))
    return out, info


def plane_fill_over_mask(clean_depth: np.ndarray, mask_depth: np.ndarray,
                         intrinsics, thresh: float = 0.05, iters: int = 300,
                         min_inliers_frac: float = 0.1, rng=None):
    """Ajusta el plano sobre la nube LIMPIA y lo DENSIFICA sobre la máscara.

    Pensado para el híbrido P2 (Hito 3): `clean_depth` es la nube **validada
    por estéreo** (`stereo_triangulate.stereo_validate_dense_depth`, dispersa
    pero correcta) y `mask_depth` es el depth **denso del SDK** (máscara de la
    malla texturada, geométricamente basura bajo el agua). Se estima el plano
    `n·X+d=0` con los puntos LIMPIOS (RANSAC+SVD) y se rellena la profundidad
    rayo-plano de **todos** los píxeles válidos de `mask_depth` → geometría
    limpia + cobertura densa = anclaje suficiente para la BA (lo que la nube
    dispersa sola no da: colapsa la escala por falta de parches anclados).

    Devuelve `(depth_out, info)`; si el RANSAC no converge sobre la nube limpia,
    degrada a `mask_depth` intacto con `info['fitted']=False`. `info`: `fitted`,
    `n_clean`, `inlier_frac` (sobre la nube limpia), `fill_px`, `normal`, `offset`.
    """
    fx, fy, cx, cy = (float(intrinsics[0]), float(intrinsics[1]),
                      float(intrinsics[2]), float(intrinsics[3]))
    H, W = mask_depth.shape
    pts, _ = backproject(clean_depth, fx, fy, cx, cy)
    info = {"fitted": False, "n_clean": int(len(pts)), "inlier_frac": 0.0,
            "fill_px": 0, "normal": None, "offset": None}
    if len(pts) < 3:
        return mask_depth, info

    res = fit_plane_ransac(pts, thresh=thresh, iters=iters,
                           min_inliers_frac=min_inliers_frac, rng=rng)
    if res is None:
        return mask_depth, info
    n, d, inliers = res

    # rellenar rayo-plano sobre los píxeles válidos de la máscara densa
    vs, us = np.nonzero(np.isfinite(mask_depth) & (mask_depth > 1e-3))
    rx = (us - cx) / fx
    ry = (vs - cy) / fy
    denom = n[0] * rx + n[1] * ry + n[2]
    Zp = np.full(len(vs), np.nan, dtype=np.float64)
    ok = np.abs(denom) > 1e-6
    Zp[ok] = -d / denom[ok]
    keep = ok & (Zp > 1e-3)

    out = np.full((H, W), np.nan, dtype=np.float32)
    out[vs[keep], us[keep]] = Zp[keep].astype(np.float32)
    info.update(fitted=True, inlier_frac=float(inliers.mean()),
                fill_px=int(keep.sum()),
                normal=[float(x) for x in n], offset=float(d))
    return out, info
