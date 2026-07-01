import torch

from .scatter_ops import scatter_sum
from . import fastba
import lietorch
from lietorch import SE3

from .utils import Timer

from . import projective_ops as pops

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U, info = torch.linalg.cholesky_ex(H)
            on_cpu = False
        except RuntimeError:
            # JP6/Tegra: libtorch_cuda_linalg.so (torch 2.8) referencia
            # cusolverDnXsyevBatched_* que el cusolver Tegra 11.6.4 NO exporta
            # -> el dlopen lazy del primer linalg CUDA revienta. Mismo bug que
            # fastba (workarounds §20, finding 2026-06-03-dpvo-embedded-
            # cusolver-ba-broken). H es chica (~6*ventana <= 120 dims) ->
            # resolver en CPU cuesta despreciable. En x86 el try no falla y la
            # numerica queda bit-exacta.
            U, info = torch.linalg.cholesky_ex(H.cpu())
            on_cpu = True

        if torch.any(info):
            ctx.failed = True
            return torch.zeros_like(b)

        xs = torch.cholesky_solve(b.cpu() if on_cpu else b, U)
        if on_cpu:
            xs = xs.to(H.device)
        ctx.save_for_backward(U, xs)
        ctx.failed = False

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def block_matmul(A, B):
    """ block matrix multiply """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)
    return torch.matmul(A, B).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

def block_solve(A, B, ep=1.0, lm=1e-4):
    """ block matrix solve """
    b, n1, m1, p1, q1 = A.shape
    b, n2, m2, p2, q2 = B.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    B = B.permute(0, 1, 3, 2, 4).reshape(b, n2*p2, m2*q2)

    A = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)

    X = CholeskySolver.apply(A, B)
    return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)


def block_show(A):
    import matplotlib.pyplot as plt
    b, n1, m1, p1, q1 = A.shape
    A = A.permute(0, 1, 3, 2, 4).reshape(b, n1*p1, m1*q1)
    plt.imshow(A[0].detach().cpu().numpy())
    plt.show()


def _plane_factor(poses, patches, intrinsics, plane, kx, patch_anchor, p):
    """Residual de coplanaridad homogéneo + jacobianos (factor de plano in-solver,
    P2 Hito 3). El plano `π=[n,d]` (coords del MUNDO/frame 0) restringe la posición
    3-D de cada parche único `kx` a ser coplanar: `r = π·X_world`, con
    `X_world = G_ancla⁻¹·X0` (X0 = retroproyección homogénea del centro del parche
    en su frame ancla). En coords homogéneas el residual es LINEAL en X_world →
    reusa los mismos generadores que `pops.transform` (Ja homogéneo + 4ª columna
    de la matriz como ∂X/∂d) y el mismo truco del adjunto para la pose ancla.

    Devuelve:
      r  (1, m)       residual homogéneo π·X_world (= W·dist_euclídea al plano)
      W  (1, m)       4ª coord homogénea (profundidad inversa en mundo) p/ trim
      Jp (1, m, 1, 6) ∂r/∂(pose ancla)  (perturbación izq. lietorch, como `Ji`)
      Jd (1, m, 1, 1) ∂r/∂(profundidad inversa del parche)
    Ver handoff docs/handoffs/2026-06-18-p2-stereo-frontend-evaluated-next-insolver-plane.md.
    """
    m = kx.shape[0]
    anchor = patch_anchor[kx]                                   # (m,) frame ancla
    X0 = pops.iproj(patches[:, kx], intrinsics[:, anchor])      # (1, m, P, P, 4)
    X0c = X0[:, :, p // 2, p // 2, :]                           # (1, m, 4) centro
    Giw = poses[:, anchor].inv()                                # (1, m) SE3 mundo<-ancla
    Miw = Giw.matrix()                                          # (1, m, 4, 4)
    Xw = torch.matmul(Miw, X0c[..., None]).squeeze(-1)          # (1, m, 4) punto mundo (homog)

    pl = plane.view(1, 1, 4).to(Xw)
    r = (pl * Xw).sum(-1)                                       # (1, m)
    W = Xw[..., 3]                                              # (1, m)

    # ∂X_world/∂d = 4ª columna de Miw (porque ∂X0/∂d = [0,0,0,1])
    Jd = (pl * Miw[..., :, 3]).sum(-1)[..., None, None]         # (1, m, 1, 1)

    # generador homogéneo en X_world (idéntico al `Ja` de pops.transform)
    X, Y, Z, H = Xw.unbind(-1)
    o = torch.zeros_like(H)
    Ja = torch.stack([
        H, o, o,  o,  Z, -Y,
        o, H, o, -Z,  o,  X,
        o, o, H,  Y, -X,  o,
        o, o, o,  o,  o,  o,
    ], dim=-1).view(1, m, 4, 6)
    Jposej = (pl.view(1, 1, 4, 1) * Ja).sum(2)[:, :, None, :]   # (1, m, 1, 6) lado mundo
    Jp = -Giw[:, :, None].adjT(Jposej)                          # (1, m, 1, 6) lado ancla
    return r, W, Jp, Jd


def _solve_augmented_imu(S, y, poses, imu, fixedp, nprime, ep, lm=1e-4):
    """Resuelve el sistema reducido de pose VISUAL (S, y) AUMENTADO con el factor
    inercial de preintegración (tight coupling, estrategia B Hito 3).

    El estado libre = poses[fixedp:n] (6 c/u) ⊕ velocidades[fixedp:n] (3 c/u),
    ordenado [todas las poses | todas las velocidades] → D = nprime*9. Las
    profundidades YA están marginalizadas en S (Schur visual) y NO acoplan con v
    → la marginalización no cambia. El factor IMU agrega curvatura en
    pose-pose / pose-vel / vel-vel (Gauss-Newton: H=JᵀΩJ, g=-JᵀΩr) entre
    keyframes consecutivos. El factor que cruza el borde fijo (fixedp-1, fixedp)
    aporta solo al lado libre (el fijo se descarta, como en el factor de plano).

    Devuelve dX (b, nprime, 1, 6, 1) (update de pose, alimenta la back-subst. de
    depth idéntica al baseline) y vels_new (n_total, 3) con las velocidades libres
    actualizadas. El solve es en CPU (D pequeño; esquiva el cuSOLVER Tegra JP6).

    ROBUSTEZ (guard de divergencia, paso 1 Hito 3 — finding
    2026-06-19-imu-tight-coupling-robustness). El estado aumentado (pose⊕vel)
    puede mal-condicionarse para ciertos muestreos de parches y la BA infla la
    escala PROGRESIVAMENTE (1/9 runs del primer experimento: v1 seed2 ×7.6, sin
    crash). Mecanismo: la velocidad absorbe la libertad de escala del gauge mono
    — `r_Δp = Riᵀ(Δpos − v·Δt − ½g·Δt²) − Δp_imu` con `Δp_imu` FIJO métrico →
    inflar `Δpos` exige inflar `v`. Por eso la dirección degenerada se mata
    ACOTANDO ‖v‖ a un máximo físico (`v_max`, proyección post-solve): si la pose
    quiere inflar, la velocidad topa el cap y el residual `r_Δp` crece →
    restablece la escala métrica. Además, si el solve es no-finito o el paso de
    pose es absurdo (`max_step`), se RECHAZA el factor IMU y se cae a la BA visual
    pura (`block_solve`) para ese paso. v_max=0 / max_step=0 ⇒ guard apagado
    (comportamiento del primer experimento). El guard es no-op en runs físicos
    (‖v‖ de marcha < cap) → no toca el anclaje de gym_v2/v3.
    """
    from .imu_preint import imu_factor_linearize, imu_information

    dev, dt = S.device, S.dtype
    Dp, Dv = nprime * 6, nprime * 3
    D = Dp + Dv
    H = torch.zeros(D, D, device=dev, dtype=dt)
    g = torch.zeros(D, device=dev, dtype=dt)

    # bloque visual (Schur de pose) -> esquina pose-pose
    S2 = S.permute(0, 1, 3, 2, 4).reshape(Dp, Dp)
    y1 = y.permute(0, 1, 3, 2, 4).reshape(Dp)
    H[:Dp, :Dp] += S2
    g[:Dp] += y1

    vels = imu["vels"]                       # (n_total, 3) body-en-mundo
    gw = imu["g"]
    sig_g, sig_a, strength = imu["sig_g"], imu["sig_a"], imu["strength"]
    v_max = float(imu.get("v_max", 0.0))     # cap físico de ‖v‖ (m/s; 0=off)
    max_step = float(imu.get("max_step", 0.0))  # cap de traslación por paso (m; 0=off)
    v_reg = float(imu.get("v_reg", 0.0))     # Tikhonov in-solver sobre ‖v‖ (0=off)
    pdata = poses.data[0]                     # (N, 7) world->cam
    for (i_g, j_g, pre) in imu["preints"]:
        a_i, a_j = i_g - fixedp, j_g - fixedp
        if a_j < 0 or a_j >= nprime:          # fijo, o fuera del sistema reducido
            continue
        vi, vj = vels[i_g], vels[j_g]
        r, Jpi, Jvi, Jpj, Jvj = imu_factor_linearize(
            pdata[i_g], pdata[j_g], vi, vj, gw, pre)
        Om = imu_information(pre, sig_g, sig_a, strength, dev, dt)
        subs = []
        if a_i >= 0:
            subs.append((Jpi, 6 * a_i))            # pose i (libre)
            subs.append((Jvi, Dp + 3 * a_i))       # vel i
        subs.append((Jpj, 6 * a_j))                # pose j (libre)
        subs.append((Jvj, Dp + 3 * a_j))           # vel j
        for (Ja, oa) in subs:
            da = Ja.shape[1]
            JaTOm = Ja.transpose(0, 1) @ Om        # (da, 9)
            g[oa:oa + da] += -(JaTOm @ r)
            for (Jb, ob) in subs:
                db = Jb.shape[1]
                H[oa:oa + da, ob:ob + db] += JaTOm @ Jb

    # --- regularización in-solver de velocidad (Tikhonov, robustez paso 1) ---
    # La dirección degenerada que infla la escala es `Δpos grande ⊕ v grande`
    # (se cancelan en r_Δp contra el Δp_imu FIJO). El cap POST-solve no la
    # previene (queda fuera del GN). Penalizar ‖v‖ DENTRO del sistema
    # (½·v_reg·‖v‖²: H[v,v]+=v_reg·I, g[v]-=v_reg·v) sí remueve esa libertad y
    # acota el drift de escala. v_reg=0 ⇒ comportamiento del primer experimento.
    if v_reg > 0:
        vfree = vels[fixedp:fixedp + nprime].reshape(-1).to(dt)   # (Dv,)
        idx = torch.arange(Dp, D, device=dev)
        H[idx, idx] += v_reg
        g[Dp:] += -v_reg * vfree

    # damping Levenberg (pose: igual que block_solve; vel: piso pequeño)
    diag = torch.diagonal(H).clone()
    damp = torch.empty(D, device=dev, dtype=dt)
    damp[:Dp] = ep + lm * diag[:Dp]
    damp[Dp:] = 1e-6 + lm * diag[Dp:]
    H = H + torch.diag(damp)

    def _cap_vels(vn):
        """Proyección física: acota ‖v‖ de las velocidades LIBRES a v_max."""
        if v_max <= 0:
            return vn, 0
        free = vn[fixedp:fixedp + nprime]
        sp = free.norm(dim=-1)
        over = sp > v_max
        nclamp = int(over.sum())
        if nclamp:
            scale = torch.where(over, v_max / sp.clamp(min=1e-9),
                                torch.ones_like(sp))
            vn[fixedp:fixedp + nprime] = free * scale[:, None]
        return vn, nclamp

    # --- guarda de divergencia: solve robusto + rechazo a visual-only ---
    reject = False
    try:
        delta = torch.linalg.solve(H.cpu(), g.cpu()).to(dev)
    except RuntimeError:
        delta, reject = None, True
    if (delta is None) or (not torch.isfinite(delta).all()):
        reject = True
    if (not reject) and max_step > 0:
        tstep = delta[:Dp].view(nprime, 6)[:, :3]          # traslación por keyframe
        if float(tstep.norm(dim=-1).max()) > max_step:
            reject = True

    if reject:
        # paso IMU descartado: BA visual pura + velocidades acotadas (no se
        # propaga la inflación). Worst-case = baseline (que en v1 ya era bueno).
        dX = block_solve(S, y, ep=ep, lm=lm)
        vels_new, _ = _cap_vels(vels.clone())
        imu["_reject"] = imu.get("_reject", 0) + 1
        return dX, vels_new

    dpose = delta[:Dp]
    dvel = delta[Dp:].reshape(nprime, 3)
    vels_new = vels.clone()
    vels_new[fixedp:fixedp + nprime] = vels[fixedp:fixedp + nprime] + dvel
    vels_new, nclamp = _cap_vels(vels_new)                 # cap físico: mata la dirección degenerada
    imu["_nclamp"] = imu.get("_nclamp", 0) + nclamp
    dX = dpose.view(1, nprime, 1, 6, 1)
    return dX, vels_new


def BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, ep=100.0, PRINT=False, fixedp=1, structure_only=False, prior_d=None, prior_w=None, prior_strength=0.0, quality_boost=1.0, plane=None, plane_strength=0.0, plane_trim=0.0, patch_anchor=None, imu=None):
    """ bundle adjustment

    prior_d / prior_w / prior_strength (variante C in-solver): factor unario de
    profundidad `d ~ N(1/Z_zed, 1/wp)` por parche. `prior_d`, `prior_w` son
    tensores planos indexados por el indice GLOBAL de parche (mismo que `kk`);
    `prior_w` es la validez (0/1) y `prior_strength` la fuerza relativa al
    termino de datos (wp = strength * mediana(C_validos)). A diferencia del blend
    post-BA, este termino agrega curvatura a la dimension de profundidad y, via
    el complemento de Schur, fuerza poses metricas (rompe el modo plano del gauge
    mono). Backward-compatible: prior_w=None => BA original.
    Ver finding 2026-06-05-dpvo-variant-c-scale-nondeterministic.

    plane / plane_strength / plane_trim / patch_anchor (factor de plano in-solver,
    P2 Hito 3): restricción de coplanaridad BLANDA `π·X_world ≈ 0` por parche, que
    entra JUNTO al factor unario de depth (no en vez de él). `plane` es el plano
    (4,) `[n,d]` en coords del MUNDO (lo arma dpvo desde el plano cam-local +
    pose de su frame de referencia). `plane_strength` la fuerza relativa al dato
    (como prior_strength); `plane_trim` (m) descarta parches fuera del net por
    distancia euclídea al plano; `patch_anchor` = mapa frame-ancla por índice
    GLOBAL de parche (`self.ix`). El depth ancla la ESCALA, el plano ancla el
    LATERAL (anti-abombamiento). plane=None ⇒ sin factor de plano.
    Ver handoff docs/handoffs/2026-06-18-p2-stereo-frontend-evaluated-next-insolver-plane.md.
    """

    b = 1
    n = max(ii.max().item(), jj.max().item()) + 1
    n_total = n                                   # nº de frames activos (antes del offset fixedp)

    coords, v, (Ji, Jj, Jz) = \
        pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    p = coords.shape[3]
    r = targets - coords[...,p//2,p//2,:]

    v *= (r.norm(dim=-1) < 250).float()

    in_bounds = \
        (coords[...,p//2,p//2,0] > bounds[0]) & \
        (coords[...,p//2,p//2,1] > bounds[1]) & \
        (coords[...,p//2,p//2,0] < bounds[2]) & \
        (coords[...,p//2,p//2,1] < bounds[3])

    v *= in_bounds.float()

    if PRINT:
        print((r * v[...,None]).norm(dim=-1).mean().item())

    r = (v[...,None] * r).unsqueeze(dim=-1)
    weights = (v[...,None] * weights).unsqueeze(dim=-1)

    # --- peso 2D informado (campaña Hito 3, reducción de jitter) ---
    # Aumenta la confianza del residual de reproyección de las observaciones cuyo
    # parche tiene depth métrica ZED válida (anclaje fiable) y deja igual al resto.
    # Los parches sin anclaje (zonas homogéneas/lejanas) suelen dar flujo ruidoso
    # que mete jitter en las poses. `kk` aquí es aún el índice GLOBAL de parche
    # (el remap a único es más abajo), así que prior_w[kk] da la validez por
    # observación. quality_boost=1.0 ⇒ comportamiento original.
    if (prior_w is not None) and (quality_boost != 1.0):
        qf = (1.0 + (quality_boost - 1.0) * prior_w[kk]).view(1, -1, 1, 1)
        weights = weights * qf

    wJiT = (weights * Ji).transpose(2,3)
    wJjT = (weights * Jj).transpose(2,3)
    wJzT = (weights * Jz).transpose(2,3)

    Bii = torch.matmul(wJiT, Ji)
    Bij = torch.matmul(wJiT, Jj)
    Bji = torch.matmul(wJjT, Ji)
    Bjj = torch.matmul(wJjT, Jj)

    Eik = torch.matmul(wJiT, Jz)
    Ejk = torch.matmul(wJjT, Jz)

    vi = torch.matmul(wJiT, r)
    vj = torch.matmul(wJjT, r)

    # fix first pose
    ii = ii.clone()
    jj = jj.clone()

    n = n - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    B = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
        safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)

    E = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
        safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1) 

    C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)

    v = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
        safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)

    w = safe_scatter_add_vec(torch.matmul(wJzT,  r), kk, m)

    Cdata = C.view(b, m)[0].clone()                       # (m,) Hessian de datos (limpio)

    # --- variante C in-solver: prior unario de profundidad (curvatura -> Schur) ---
    if (prior_w is not None) and (prior_d is not None) and (prior_strength > 0):
        pw = prior_w[kx]                                   # (m,) validez 0/1
        if torch.any(pw > 0):
            cref = Cdata[pw > 0].median() if (pw > 0).any() else C.new_tensor(1.0)
            wp = (prior_strength * cref * pw).view(b, m, 1, 1)
            d_cur = patches[:, kx, 2, p // 2, p // 2].view(b, m, 1, 1)
            d_zed = prior_d[kx].view(b, m, 1, 1)
            C = C + wp                                     # +curvatura en profundidad
            w = w + wp * (d_zed - d_cur)                   # +gradiente del prior

    # --- factor de plano in-solver: coplanaridad blanda π·X_world ≈ 0 (P2 Hito 3) ---
    # Ancla la geometría LATERAL del net (anti-abombamiento) sin tocar la escala
    # (la fija el factor unario de depth de arriba). Residual homogéneo por parche
    # → curvatura en pose ancla + profundidad, acopladas vía el complemento de
    # Schur (B, E, C, v, w). Las contribuciones de pose se descartan solas para
    # parches anclados a poses FIJAS (safe_scatter guarda índice <0); su depth sí
    # entra (C, w). Ver handoff 2026-06-18-p2-stereo-frontend-evaluated.
    if (plane is not None) and (plane_strength > 0) and (patch_anchor is not None):
        r_pl, W_pl, Jp_pl, Jd_pl = _plane_factor(
            poses, patches, intrinsics, plane, kx, patch_anchor, p)
        pw_pl = (prior_w[kx] if prior_w is not None
                 else torch.ones_like(r_pl[0])).clone()    # (m,) validez en el net
        if plane_trim > 0:
            # trim por distancia EUCLÍDEA al plano (r = W·dist) — protege de parches
            # fuera del net (foreground) sin forzarlos a la malla.
            dist = (r_pl / W_pl.clamp(min=1e-6))[0]        # (m,)
            pw_pl = pw_pl * (dist.abs() < plane_trim).float()
        if torch.any(pw_pl > 0):
            cref_pl = Cdata[pw_pl > 0].median()
            wpl = (plane_strength * cref_pl * pw_pl).view(b, m, 1, 1)   # (b,m,1,1)
            JpT = Jp_pl.transpose(-1, -2)                  # (1,m,6,1)
            neg_r = (-r_pl).view(b, m, 1, 1)               # (b,m,1,1)
            Bpl = torch.matmul(JpT, Jp_pl) * wpl           # (1,m,6,6)
            Epl = torch.matmul(JpT, Jd_pl) * wpl           # (1,m,6,1)
            Cpl = (Jd_pl ** 2).view(b, m, 1, 1) * wpl      # (b,m,1,1)
            vpl = JpT * (wpl * neg_r)                      # (1,m,6,1)
            wpl_d = (Jd_pl.view(b, m, 1, 1)) * (wpl * neg_r)               # (b,m,1,1)
            a_idx = (patch_anchor[kx] - fixedp).to(ii.device)             # (m,) pose remap
            kidx = torch.arange(m, device=ii.device)
            B = B + safe_scatter_add_mat(Bpl, a_idx, a_idx, n, n).view(b, n, n, 6, 6)
            E = E + safe_scatter_add_mat(Epl, a_idx, kidx, n, m).view(b, n, m, 6, 1)
            v = v + safe_scatter_add_vec(vpl, a_idx, n).view(b, n, 1, 6, 1)
            C = C + Cpl
            w = w + wpl_d

    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)
        
    Q = 1.0 / (C + lmbda)
    
    ### solve w/ schur complement ###
    EQ = E * Q[:,None]

    vels_out = None
    if structure_only or n == 0:
        dZ = (Q * w).view(b, -1, 1, 1)

    else:
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        if (imu is not None) and (len(imu.get("preints", [])) > 0):
            # tight coupling: sistema reducido de pose AUMENTADO con velocidades
            # + factor de preintegración IMU (estrategia B Hito 3).
            dX, vels_out = _solve_augmented_imu(S, y, poses, imu, fixedp, n, ep)
        else:
            dX = block_solve(S, y, ep=ep, lm=1e-4)

        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    if not structure_only and n > 0:
        poses = pose_retr(poses, dX, fixedp + torch.arange(n))

    if imu is not None:
        return poses, patches, vels_out
    return poses, patches
