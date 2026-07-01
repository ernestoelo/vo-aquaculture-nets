#!/usr/bin/env python3
"""stereo_triangulate.py — Triangulación estéreo DISPERSA por correspondencia
sobre el par rectificado de la ZED (LEFT/RIGHT), como se hace para biomasa.

Motivación (Hito 3): el depth DENSO del ZED SDK es un "queso suizo" bajo el
agua — esparso, sesgado y volátil; el RANSAC del plano (P1) sobre esa nube
ajusta los matches espurios y colapsa la escala underwater (finding
`docs/findings/2026-06-18-dpvo-plane-anchor-p1-air-helps-uw-hurts.md`). La vía
limpia es **triangular sólo los píxeles texturados emparejables**: por cada
punto de interés de la imagen LEFT se busca su correspondencia en RIGHT sobre
la línea epipolar (imágenes rectificadas → búsqueda 1-D en x), se valida con un
chequeo de unicidad + consistencia izquierda↔derecha, y se calcula
`Z = fx·B/disparidad`. Eso convierte la validez de **sintáctica** (finito>0 en
el mapa denso) a **geométrica** (emparejable y consistente).

Es la versión EMBARCABLE de lo que MAC-VO hace con un transformer FlowFormer:
misma geometría (`Z = fx·B/disp`), pero con matching de bloques clásico (sin
red), inmune también a la ambigüedad repetitiva de la malla gracias al chequeo
de unicidad. Ver finding
`docs/findings/2026-06-07-zed-dense-depth-sparse-stereo-triangulation-direction.md`.

Salida: un mapa de profundidad **disperso** (H×W float32, NaN salvo en los
píxeles triangulados y validados) — drop-in para `plane_anchor.plane_denoise_depth`
y para la siembra métrica de DPVO (`run_sdpvo_metric.py`, los NaN → peso 0 en el
factor unario).

Convención de signo (ZED rectificada, cámara derecha desplazada +X·baseline):
un punto 3-D aparece en RIGHT a la IZQUIERDA de su columna en LEFT, con
disparidad `d = u_left - u_right > 0`.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # permite importar el módulo sin cv2 para introspección
    cv2 = None


def sparse_stereo_depth(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    fx: float,
    baseline: float,
    *,
    z_min: float = 0.3,
    z_max: float = 20.0,
    win: int = 9,
    max_kp: int = 1500,
    gftt_quality: float = 0.01,
    gftt_min_dist: int = 7,
    min_ncc: float = 0.6,
    uniqueness_ratio: float = 0.9,
    lr_check: bool = True,
    lr_max_diff: float = 1.5,
    subpixel: bool = True,
):
    """Triangula disperso LEFT↔RIGHT y devuelve `(depth, info)`.

    `left_gray`, `right_gray`: imágenes RECTIFICADAS en escala de grises (uint8),
    misma resolución. `fx`, `cx`, `cy` deben estar en píxeles a ESA resolución
    (escalar si se redimensionó). `baseline` en metros (del SDK
    `get_camera_baseline()`, NO hardcodear: varía por cámara).

    `depth` (H, W float32): NaN salvo en los píxeles con match validado, donde
    vale `Z = fx·baseline/disparidad`. `info`: dict con `n_kp`, `n_valid`,
    `valid_frac`, `match_rate`, `disp_med`, `z_med`.
    """
    if cv2 is None:
        raise RuntimeError("cv2 no disponible: instalar opencv-python.")
    H, W = left_gray.shape
    info = {"n_kp": 0, "n_valid": 0, "valid_frac": 0.0, "match_rate": 0.0,
            "disp_med": None, "z_med": None}
    depth = np.full((H, W), np.nan, dtype=np.float32)

    fxB = float(fx) * float(baseline)
    d_min = max(1.0, fxB / float(z_max))   # disparidad mínima (Z lejano)
    d_max = fxB / float(z_min)             # disparidad máxima (Z cercano)
    hw = win // 2

    # Puntos de interés en LEFT: texturados = emparejables (lo que DPVO también
    # prefiere). goodFeaturesToTrack ya impone separación mínima y umbral de
    # calidad → evita zonas planas donde la correlación es ambigua.
    corners = cv2.goodFeaturesToTrack(
        left_gray, maxCorners=int(max_kp), qualityLevel=float(gftt_quality),
        minDistance=int(gftt_min_dist))
    if corners is None:
        return depth, info
    kps = corners.reshape(-1, 2)
    info["n_kp"] = int(len(kps))

    disps, zs = [], []
    for (uf, vf) in kps:
        u, v = int(round(uf)), int(round(vf))
        # margen para la ventana (filas) y para el rango de búsqueda (columnas)
        if v - hw < 0 or v + hw >= H or u - hw < 0 or u + hw >= W:
            continue
        templ = left_gray[v - hw:v + hw + 1, u - hw:u + hw + 1]

        # franja de búsqueda en RIGHT: centros candidatos x_c ∈ [u-d_max, u-d_min]
        # → top-left x ∈ [u-d_max-hw, u-d_min-hw]; añadir win-1 al final.
        x0 = max(0, int(np.floor(u - d_max - hw)))
        x1 = min(W, int(np.ceil(u - d_min + hw)) + 1)
        if x1 - x0 < win + 1:                # franja demasiado corta para buscar
            continue
        strip = right_gray[v - hw:v + hw + 1, x0:x1]
        res = cv2.matchTemplate(strip, templ, cv2.TM_CCOEFF_NORMED).ravel()
        if res.size < 3:
            continue
        i_best = int(np.argmax(res))
        c_best = float(res[i_best])
        if c_best < min_ncc:
            continue

        # unicidad: rechaza si hay un segundo pico casi tan alto (textura
        # repetitiva → match ambiguo, p.ej. la malla). Ignora ±2 alrededor del
        # pico para no contar su propia ladera.
        mask = np.ones(res.size, dtype=bool)
        lo, hi = max(0, i_best - 2), min(res.size, i_best + 3)
        mask[lo:hi] = False
        if mask.any():
            second = float(res[mask].max())
            if second > uniqueness_ratio * c_best:
                continue

        # refinamiento sub-píxel por parábola sobre el pico (en índice)
        di = 0.0
        if subpixel and 0 < i_best < res.size - 1:
            cm, c0, cp = res[i_best - 1], res[i_best], res[i_best + 1]
            denom = (cm - 2 * c0 + cp)
            if abs(denom) > 1e-9:
                di = 0.5 * (cm - cp) / denom
        center_x = x0 + hw + (i_best + di)   # columna del match en RIGHT
        disp = u - center_x
        if disp < d_min or disp > d_max:
            continue

        # consistencia izquierda↔derecha: re-buscar el match (derecha→izquierda)
        # y exigir que vuelva a ~u. Mata medias-correspondencias y oclusiones.
        if lr_check:
            xr = int(round(center_x))
            if xr - hw < 0 or xr + hw >= W:
                continue
            templ_r = right_gray[v - hw:v + hw + 1, xr - hw:xr + hw + 1]
            # en LEFT el punto está a la DERECHA de RIGHT por la disparidad
            lx0 = max(0, int(np.floor(xr + d_min - hw)))
            lx1 = min(W, int(np.ceil(xr + d_max + hw)) + 1)
            if lx1 - lx0 < win + 1:
                continue
            lstrip = left_gray[v - hw:v + hw + 1, lx0:lx1]
            lres = cv2.matchTemplate(lstrip, templ_r, cv2.TM_CCOEFF_NORMED).ravel()
            if lres.size < 1:
                continue
            jl = int(np.argmax(lres))
            u_back = lx0 + hw + jl
            if abs(u_back - u) > lr_max_diff:
                continue

        Z = fxB / disp
        depth[v, u] = np.float32(Z)
        disps.append(disp)
        zs.append(Z)

    n_valid = len(zs)
    info.update(
        n_valid=int(n_valid),
        valid_frac=float(np.isfinite(depth).mean()),
        match_rate=float(n_valid / info["n_kp"]) if info["n_kp"] else 0.0,
        disp_med=float(np.median(disps)) if disps else None,
        z_med=float(np.median(zs)) if zs else None,
    )
    return depth, info


def stereo_validate_dense_depth(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    dense_depth: np.ndarray,
    fx: float,
    baseline: float,
    *,
    disp_halfwin: float = 4.0,
    win: int = 9,
    min_ncc: float = 0.6,
    max_kp: int = 3000,
    gftt_quality: float = 0.01,
    gftt_min_dist: int = 7,
    lr_check: bool = True,
    lr_max_diff: float = 1.5,
    subpixel: bool = True,
):
    """Valida y refina el depth DENSO del SDK con matching estéreo ACOTADO.

    Por cada punto de interés con depth denso válido se busca la correspondencia
    en RIGHT **solo en una ventana estrecha** alrededor de la disparidad que
    implica el SDK (`d_sdk = fx·B/Z_sdk ± disp_halfwin`). Eso:
      1. **rompe el aliasing** (la ventana es más angosta que el período de la
         textura repetitiva → los aliases quedan fuera; cf. finding
         `2026-06-18-sparse-stereo-classic-matching-aliases-repetitive`),
      2. **descarta el "queso suizo"**: los píxeles cuyo depth denso es espurio
         tienen su disparidad real LEJOS de `d_sdk` → no hay match consistente
         en la ventana → se rechazan,
      3. **refina** sub-píxel los que sobreviven.

    Devuelve `(depth, info)` con `depth` (H, W float32) NaN salvo en los píxeles
    validados (subconjunto del denso). `info`: `n_kp`, `n_dense_valid`,
    `n_valid`, `valid_frac`, `keep_rate` (validados/kp-con-denso),
    `disp_med`, `z_med`.
    """
    if cv2 is None:
        raise RuntimeError("cv2 no disponible: instalar opencv-python.")
    H, W = left_gray.shape
    info = {"n_kp": 0, "n_dense_valid": 0, "n_valid": 0, "valid_frac": 0.0,
            "keep_rate": 0.0, "disp_med": None, "z_med": None}
    depth = np.full((H, W), np.nan, dtype=np.float32)
    fxB = float(fx) * float(baseline)
    hw = win // 2

    corners = cv2.goodFeaturesToTrack(
        left_gray, maxCorners=int(max_kp), qualityLevel=float(gftt_quality),
        minDistance=int(gftt_min_dist))
    if corners is None:
        return depth, info
    kps = corners.reshape(-1, 2)
    info["n_kp"] = int(len(kps))

    n_dense, disps, zs = 0, [], []
    for (uf, vf) in kps:
        u, v = int(round(uf)), int(round(vf))
        if v - hw < 0 or v + hw >= H or u - hw < 0 or u + hw >= W:
            continue
        zd = dense_depth[v, u]
        if not (np.isfinite(zd) and zd > 1e-3):
            continue                       # solo validamos donde el denso opina
        n_dense += 1
        d_sdk = fxB / float(zd)
        d_lo = max(1.0, d_sdk - disp_halfwin)
        d_hi = d_sdk + disp_halfwin

        templ = left_gray[v - hw:v + hw + 1, u - hw:u + hw + 1]
        x0 = max(0, int(np.floor(u - d_hi - hw)))
        x1 = min(W, int(np.ceil(u - d_lo + hw)) + 1)
        if x1 - x0 < win + 1:
            continue
        strip = right_gray[v - hw:v + hw + 1, x0:x1]
        res = cv2.matchTemplate(strip, templ, cv2.TM_CCOEFF_NORMED).ravel()
        if res.size < 1:
            continue
        i_best = int(np.argmax(res))
        if float(res[i_best]) < min_ncc:
            continue
        di = 0.0
        if subpixel and 0 < i_best < res.size - 1:
            cm, c0, cp = res[i_best - 1], res[i_best], res[i_best + 1]
            denom = (cm - 2 * c0 + cp)
            if abs(denom) > 1e-9:
                di = 0.5 * (cm - cp) / denom
        center_x = x0 + hw + (i_best + di)
        disp = u - center_x
        if disp < d_lo - 0.5 or disp > d_hi + 0.5:
            continue                       # el pico cayó en el borde → dudoso

        if lr_check:
            xr = int(round(center_x))
            if xr - hw < 0 or xr + hw >= W:
                continue
            templ_r = right_gray[v - hw:v + hw + 1, xr - hw:xr + hw + 1]
            lx0 = max(0, int(np.floor(xr + d_lo - hw)))
            lx1 = min(W, int(np.ceil(xr + d_hi + hw)) + 1)
            if lx1 - lx0 < win + 1:
                continue
            lstrip = left_gray[v - hw:v + hw + 1, lx0:lx1]
            lres = cv2.matchTemplate(lstrip, templ_r, cv2.TM_CCOEFF_NORMED).ravel()
            if lres.size < 1:
                continue
            u_back = lx0 + hw + int(np.argmax(lres))
            if abs(u_back - u) > lr_max_diff:
                continue

        Z = fxB / disp
        depth[v, u] = np.float32(Z)
        disps.append(disp)
        zs.append(Z)

    n_valid = len(zs)
    info.update(
        n_dense_valid=int(n_dense), n_valid=int(n_valid),
        valid_frac=float(np.isfinite(depth).mean()),
        keep_rate=float(n_valid / n_dense) if n_dense else 0.0,
        disp_med=float(np.median(disps)) if disps else None,
        z_med=float(np.median(zs)) if zs else None,
    )
    return depth, info


def _self_test() -> int:
    """Verifica la geometría con un plano fronto-paralelo sintético texturado.

    Construye LEFT/RIGHT con una disparidad ENTERA conocida (sin interpolación)
    y comprueba que la profundidad recuperada coincide con `Z = fx·B/disp`.
    """
    rng = np.random.default_rng(0)
    H, W = 360, 640
    fx, B = 262.0, 0.12
    disp = 16
    Z_expected = fx * B / disp
    # lienzo de textura aleatoria suavizada (correlable, no ruido puro)
    canvas = rng.integers(0, 255, (H, W + 2 * disp), dtype=np.uint8)
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    # contenido en columna C del lienzo → LEFT u=C-disp, RIGHT u=C-2·disp
    # ⇒ disparidad u_left-u_right = disp (>0, RIGHT a la izquierda). OK.
    left = canvas[:, disp:disp + W].copy()
    right = canvas[:, 2 * disp:2 * disp + W].copy()

    depth, info = sparse_stereo_depth(left, right, fx, B, z_min=0.3, z_max=5.0,
                                      win=9, max_kp=2000, min_ncc=0.5)
    zvals = depth[np.isfinite(depth)]
    if zvals.size < 50:
        print(f"SELF-TEST FAIL: muy pocos matches ({zvals.size})")
        return 1
    z_med = float(np.median(zvals))
    err_mm = abs(z_med - Z_expected) * 1000.0
    print(f"self-test: disp={disp}px Z_exp={Z_expected:.4f}m  "
          f"Z_med={z_med:.4f}m  err={err_mm:.2f}mm  "
          f"n_kp={info['n_kp']} n_valid={info['n_valid']} "
          f"match_rate={100*info['match_rate']:.0f}%")
    # tolerancia 1 px de disparidad ≈ el espaciado discreto del matching entero
    if err_mm > Z_expected * 1000.0 * (1.0 / disp) * 1.2:
        print("SELF-TEST FAIL: error de profundidad fuera de tolerancia")
        return 1
    print("SELF-TEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
