"""imu_preint.py — Preintegración IMU (Forster et al. 2017) + factor inercial
para tight coupling DPVO-métrico (estrategia B del Hito 3,
`docs/06-imu-integration-roadmap.md` §B).

A diferencia del EKF loose-coupling (`scripts/ekf_fuse_imu.py`, estrategia A),
aquí la IMU **entra a la optimización** (la BA de Python de `ba.py`): la
doble integración del acelerómetro restringe la ESCALA y la GRAVEDAD junto a la
VO. El residual de preintegración conecta dos keyframes consecutivos de la
ventana (pose_i, v_i) ↔ (pose_j, v_j).

CONVENCIONES
- Body = cámara izquierda (la IMU se rota al frame cámara con R_cam<-imu de
  fábrica ANTES de llamar aquí; aquí acc/gyro ya están en el frame cámara, acc
  en m/s², gyro en **rad/s** — la conversión deg/s→rad/s la hace
  `extract_imu_svo.py`, gotcha pyzed 2026-06-19).
- DPVO guarda poses **world→cam** (G = T_cam<-world). El body-en-mundo es
  R_wb = R(G)ᵀ, p_wb = -R(G)ᵀ t(G). Esta capa trabaja con (R_wb, p_wb, v) en
  el mundo (frame DPVO 0). El puente world→cam ↔ body-en-mundo lo hace
  `pose_to_body()`.
- v1 SIMPLE (handoff 2026-06-19): bias≈0 FIJO (los biases del static window del
  gym están contaminados por movimiento → no fiables), gravedad FIJA estimada
  al inicio. Solo se optimizan poses (ya en DPVO) + velocidades (3/keyframe).

Referencia: C. Forster, L. Carlone, F. Dellaert, D. Scaramuzza, "On-Manifold
Preintegration for Real-Time Visual-Inertial Odometry," IEEE T-RO 33(1), 2017.
"""
from __future__ import annotations

import torch


# ------------------------------- SO(3) -----------------------------------
def so3_hat(w: torch.Tensor) -> torch.Tensor:
    """(...,3) -> (...,3,3) matriz antisimétrica [w]×."""
    o = torch.zeros_like(w[..., 0])
    return torch.stack([
        o, -w[..., 2], w[..., 1],
        w[..., 2], o, -w[..., 0],
        -w[..., 1], w[..., 0], o,
    ], dim=-1).reshape(*w.shape[:-1], 3, 3)


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Exp de SO(3) (Rodrigues). (...,3) -> (...,3,3)."""
    theta = w.norm(dim=-1, keepdim=True)                       # (...,1)
    small = theta < 1e-8
    th = theta.clamp(min=1e-8)
    k = w / th
    K = so3_hat(k)
    s = torch.sin(th)[..., None]
    c = torch.cos(th)[..., None]
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
    R = eye + s * K + (1 - c) * (K @ K)
    # límite θ→0: R ≈ I + [w]×
    Rsmall = eye + so3_hat(w)
    return torch.where(small[..., None], Rsmall, R)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Log de SO(3). (...,3,3) -> (...,3)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_t = ((tr - 1) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_t)                                  # (...)
    w = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)                                                 # (...,3) = 2 sinθ · axis
    small = theta < 1e-5
    sin_t = torch.sin(theta).clamp(min=1e-8)
    coef = torch.where(small, torch.full_like(theta, 0.5),
                       theta / (2 * sin_t))
    return coef[..., None] * w


# --------------------------- Preintegración ------------------------------
class Preint:
    """Medición preintegrada entre dos keyframes (bias fijo)."""
    __slots__ = ("dR", "dv", "dp", "DT")

    def __init__(self, dR, dv, dp, DT):
        self.dR = dR        # (3,3) ΔR_ij
        self.dv = dv        # (3,)  Δv_ij
        self.dp = dp        # (3,)  Δp_ij
        self.DT = DT        # float Δt_ij


def preintegrate(acc: torch.Tensor, gyro: torch.Tensor, dts: torch.Tensor,
                 ba: torch.Tensor | None = None,
                 bg: torch.Tensor | None = None) -> Preint:
    """Preintegra un segmento IMU (Forster discreto, ZOH al inicio del intervalo).

    acc, gyro : (K,3) muestras en el frame cámara (acc m/s², gyro rad/s).
    dts       : (K,) dt de cada paso (dt_k = t_{k+1}-t_k). El último dt puede ser
                el resto hasta t_j. Se integran K pasos.
    ba, bg    : (3,) bias acc/gyro (default 0).

    Devuelve Preint(dR, dv, dp, DT). Δp usa dv ANTES de actualizarla (orden
    Forster). Determinista; sin gravedad (es relativa).
    """
    dev, dt_type = acc.device, acc.dtype
    if ba is None:
        ba = torch.zeros(3, device=dev, dtype=dt_type)
    if bg is None:
        bg = torch.zeros(3, device=dev, dtype=dt_type)
    dR = torch.eye(3, device=dev, dtype=dt_type)
    dv = torch.zeros(3, device=dev, dtype=dt_type)
    dp = torch.zeros(3, device=dev, dtype=dt_type)
    DT = 0.0
    K = acc.shape[0]
    for k in range(K):
        dt = float(dts[k])
        if dt <= 0:
            continue
        a = acc[k] - ba
        w = gyro[k] - bg
        Ra = dR @ a
        dp = dp + dv * dt + 0.5 * Ra * dt * dt
        dv = dv + Ra * dt
        dR = dR @ so3_exp(w * dt)
        DT += dt
    return Preint(dR, dv, dp, DT)


# ------------------------------ Factor IMU -------------------------------
def pose_to_body(G_data: torch.Tensor):
    """world→cam (lietorch SE3 data, (...,7) [tx,ty,tz,qx,qy,qz,qw]) → body-en-mundo.

    Devuelve (R_wb (...,3,3), p_wb (...,3)). R_wb = R(G)ᵀ, p_wb = -R(G)ᵀ t.
    """
    from lietorch import SE3
    M = SE3(G_data).matrix()                          # (...,4,4) world→cam
    R_cw = M[..., :3, :3]
    t_cw = M[..., :3, 3]
    R_wb = R_cw.transpose(-1, -2)
    p_wb = -(R_wb @ t_cw[..., None]).squeeze(-1)
    return R_wb, p_wb


def imu_residual(Ri, pi, vi, Rj, pj, vj, g, pre: Preint) -> torch.Tensor:
    """Residual de preintegración (9,) [r_ΔR(3); r_Δv(3); r_Δp(3)] (Forster eq 37,
    bias fijo). Todos los estados en el mundo (body-en-mundo).

    r_ΔR = Log(ΔRᵀ Riᵀ Rj)
    r_Δv = Riᵀ(vj - vi - g·Δt) - Δv
    r_Δp = Riᵀ(pj - pi - vi·Δt - 0.5 g·Δt²) - Δp
    """
    DT = pre.DT
    RiT = Ri.transpose(-1, -2)
    r_R = so3_log(pre.dR.transpose(-1, -2) @ RiT @ Rj)
    r_v = (RiT @ (vj - vi - g * DT)[..., None]).squeeze(-1) - pre.dv
    r_p = (RiT @ (pj - pi - vi * DT - 0.5 * g * DT * DT)[..., None]).squeeze(-1) - pre.dp
    return torch.cat([r_R, r_v, r_p], dim=-1)


def vel_jacobians(Ri, DT):
    """Jacobianos analíticos del residual (9,) respecto a v_i y v_j (3 c/u).

    ∂r/∂v_i = [[0],[-Riᵀ],[-Riᵀ·Δt]]   ∂r/∂v_j = [[0],[Riᵀ],[0]]
    Devuelve (Jvi (9,3), Jvj (9,3)).
    """
    RiT = Ri.transpose(-1, -2)
    Z = torch.zeros_like(RiT)
    Jvi = torch.cat([Z, -RiT, -RiT * DT], dim=-2)             # (9,3)
    Jvj = torch.cat([Z, RiT, Z], dim=-2)                      # (9,3)
    return Jvi, Jvj


def so3_jr_inv(phi: torch.Tensor) -> torch.Tensor:
    """Inversa de la jacobiana DERECHA de SO(3) en `phi`. (...,3) -> (...,3,3).

    J_r⁻¹(φ) = I + ½[φ]× + c₂·[φ]×²   con
        c₂ = 1/θ² - cos(θ/2)/(2θ·sin(θ/2))     (θ=‖φ‖, → 1/12 cuando θ→0).
    La forma con cot(θ/2) evita la singularidad de (1+cosθ)/sinθ en θ=π (sin(θ/2)
    solo se anula en θ=0, cubierto por la rama small). La inversa IZQUIERDA es la
    transpuesta: J_l⁻¹(φ) = J_r⁻¹(φ)ᵀ (porque [φ]×ᵀ=-[φ]× y ([φ]×²)ᵀ=[φ]×²).
    """
    theta = phi.norm(dim=-1, keepdim=True)                    # (...,1)
    small = theta < 1e-4
    th = theta.clamp(min=1e-8)
    half = 0.5 * th
    c2 = 1.0 / (th * th) - torch.cos(half) / (2.0 * th * torch.sin(half))
    c2 = torch.where(small, torch.full_like(c2, 1.0 / 12.0), c2)   # (...,1)
    K = so3_hat(phi)                                          # (...,3,3)
    K2 = K @ K
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand_as(K)
    return eye + 0.5 * K + c2[..., None] * K2                 # (...,3,3)


def pose_jacobians(Ri, Rj, RiT_av, RiT_ap, r_R, dR):
    """Jacobianos ANALÍTICOS de pose del residual IMU (9,) respecto a las
    perturbaciones lietorch `retr` de pose_i y pose_j.

    CONVENCIÓN (la trampa del handoff 2026-06-20): DPVO actualiza poses con
    `poses.retr(dx)` = `Exp(dx)·G` (perturbación IZQUIERDA de la pose world→cam,
    `dx=[ρ(tras), ω(rot)]` en ese orden — el tangente SE3 de lietorch). Propagada
    por `pose_to_body`, a primer orden induce sobre el estado body-en-mundo:
        δR_wb = -R_wb[ω]×      (retracción DERECHA de R_wb por -ω)
        δp_wb = -R_wb·ρ
    De la regla de la cadena sobre `imu_residual` salen estos bloques no nulos:

      ∂r_R/∂ω_i =  J_l⁻¹(r_R)·ΔRᵀ
      ∂r_v/∂ω_i = -[Riᵀ·a_v]×
      ∂r_p/∂ρ_i =  I            ∂r_p/∂ω_i = -[Riᵀ·a_p]×
      ∂r_R/∂ω_j = -J_r⁻¹(r_R)
      ∂r_p/∂ρ_j = -Riᵀ·Rj
    con a_v = v_j-v_i-g·Δt, a_p = p_j-p_i-v_i·Δt-½g·Δt², y la identidad útil
    Riᵀ·a_v = r_v+Δv, Riᵀ·a_p = r_p+Δp (= `RiT_av`, `RiT_ap`, reusa el residual).

    Ri,Rj (3,3); RiT_av,RiT_ap (3,); r_R (3,); dR (3,3)=`pre.dR`.
    Devuelve Jpi (9,6), Jpj (9,6) [cols 0-2 = ρ (tras), 3-5 = ω (rot)].
    """
    dev, dt = Ri.device, Ri.dtype
    RiT = Ri.transpose(-1, -2)
    Jr_inv = so3_jr_inv(r_R)                                  # (3,3)
    Jl_inv = Jr_inv.transpose(-1, -2)                         # J_l⁻¹ = (J_r⁻¹)ᵀ

    Jpi = torch.zeros(9, 6, device=dev, dtype=dt)
    Jpj = torch.zeros(9, 6, device=dev, dtype=dt)
    I3 = torch.eye(3, device=dev, dtype=dt)

    # ---- pose i ----
    Jpi[0:3, 3:6] = Jl_inv @ dR.to(dt).transpose(-1, -2)      # ∂r_R/∂ω_i
    Jpi[3:6, 3:6] = -so3_hat(RiT_av)                          # ∂r_v/∂ω_i
    Jpi[6:9, 0:3] = I3                                        # ∂r_p/∂ρ_i
    Jpi[6:9, 3:6] = -so3_hat(RiT_ap)                          # ∂r_p/∂ω_i

    # ---- pose j ----
    Jpj[0:3, 3:6] = -Jr_inv                                   # ∂r_R/∂ω_j
    Jpj[6:9, 0:3] = -(RiT @ Rj)                               # ∂r_p/∂ρ_j
    return Jpi, Jpj


def imu_information(pre: Preint, sig_g: float, sig_a: float,
                   strength: float, device, dtype) -> torch.Tensor:
    """Matriz de información Ω (9,9) diagonal del residual de preintegración.

    Aproximación de escalado temporal (v1, en vez de la covarianza recursiva
    completa de Forster): el error de Log(ΔR) crece ~√Δt (var ~Δt), el de Δv
    ~√Δt, y el de Δp (doble integración) ~Δt^1.5 (var ~Δt³). Con un piso para no
    explotar en Δt→0:
        σ_R² = sig_g²·DT      σ_v² = sig_a²·DT     σ_p² = sig_a²·DT³/3
    Ω = strength · diag(1/σ²). `strength` escala el peso global del factor IMU
    relativo al término visual (como prior_strength/plane_strength).
    """
    DT = max(pre.DT, 1e-3)
    s_R2 = sig_g * sig_g * DT
    s_v2 = sig_a * sig_a * DT
    s_p2 = sig_a * sig_a * DT * DT * DT / 3.0
    inv = torch.tensor(
        [1.0 / s_R2] * 3 + [1.0 / s_v2] * 3 + [1.0 / s_p2] * 3,
        device=device, dtype=dtype)
    return strength * torch.diag(inv)


def segment_preint(imu_ts: torch.Tensor, imu_acc: torch.Tensor,
                   imu_gyr: torch.Tensor, ts_i: int, ts_j: int,
                   ba=None, bg=None) -> Preint:
    """Preintegra el segmento IMU entre dos timestamps de keyframe (ns).

    imu_ts (M,) long ORDENADO, imu_acc/imu_gyr (M,3) en frame cámara. Toma las
    muestras en [ts_i, ts_j) (ZOH al inicio del intervalo) y el último dt cierra
    hasta ts_j. Sin muestras (gap minúsculo) → Preint identidad con DT correcto.
    """
    dev, dt_type = imu_acc.device, imu_acc.dtype
    ti = torch.tensor(int(ts_i), device=dev, dtype=imu_ts.dtype)
    tj = torch.tensor(int(ts_j), device=dev, dtype=imu_ts.dtype)
    lo = int(torch.searchsorted(imu_ts, ti, right=False))
    hi = int(torch.searchsorted(imu_ts, tj, right=False))
    DT = (int(ts_j) - int(ts_i)) / 1e9
    if hi - lo < 1:
        eye = torch.eye(3, device=dev, dtype=dt_type)
        z = torch.zeros(3, device=dev, dtype=dt_type)
        return Preint(eye, z, z, DT)
    times = imu_ts[lo:hi].to(torch.float64)
    acc = imu_acc[lo:hi]
    gyr = imu_gyr[lo:hi]
    L = times.shape[0]
    dts = torch.empty(L, device=dev, dtype=dt_type)
    if L > 1:
        dts[:L - 1] = ((times[1:] - times[:-1]) / 1e9).to(dt_type)
    dts[L - 1] = float((int(ts_j) - float(times[-1])) / 1e9)
    return preintegrate(acc, gyr, dts, ba, bg)


def imu_factor_linearize(Gi_data, Gj_data, vi, vj, g, pre: Preint,
                         pose_jac: str = "analytic", eps: float = 5e-4):
    """Linealiza el residual IMU (9,) respecto a (pose_i, v_i, pose_j, v_j).

    Gi_data, Gj_data: (7,) datos lietorch SE3 world→cam de los dos keyframes.
    vi, vj: (3,) velocidades body-en-mundo. g: (3,) gravedad mundo. pre: Preint.

    Pose (`pose_jac="analytic"`, default): jacobiano ANALÍTICO (`pose_jacobians`)
    en la MISMA base que la BA — la perturbación `Exp(d)·G` de `pose_retr`. Elimina
    el forward-diff de 12 reevaluaciones del residual por arista (~4× fps, finding
    2026-06-20-imu-tight-analytical-pose-jacobians). El modo `"numeric"` conserva
    el forward-diff vía `SE3.retr` como referencia/fallback (lo usa el self-test).
    Velocidad: jacobiano ANALÍTICO (`vel_jacobians`). Devuelve
    (r (9,), Jpi (9,6), Jvi (9,3), Jpj (9,6), Jvj (9,3)).
    """
    from lietorch import SE3
    dev = Gi_data.device
    dt_type = vi.dtype
    Ri, pi = pose_to_body(Gi_data.view(1, 7))
    Rj, pj = pose_to_body(Gj_data.view(1, 7))
    Ri = Ri[0].to(dt_type); pi = pi[0].to(dt_type)
    Rj = Rj[0].to(dt_type); pj = pj[0].to(dt_type)
    r0 = imu_residual(Ri, pi, vi, Rj, pj, vj, g, pre)
    Jvi, Jvj = vel_jacobians(Ri, pre.DT)

    if pose_jac == "analytic":
        RiT_av = r0[3:6] + pre.dv.to(dt_type)                 # = Riᵀ·a_v
        RiT_ap = r0[6:9] + pre.dp.to(dt_type)                 # = Riᵀ·a_p
        Jpi, Jpj = pose_jacobians(Ri, Rj, RiT_av, RiT_ap, r0[0:3], pre.dR)
        return r0, Jpi, Jvi, Jpj, Jvj

    # ---- fallback NUMÉRICO (forward diff vía lietorch retr) ----
    Gi = SE3(Gi_data.view(1, 7))
    Gj = SE3(Gj_data.view(1, 7))
    Jpi = torch.zeros(9, 6, device=dev, dtype=dt_type)
    Jpj = torch.zeros(9, 6, device=dev, dtype=dt_type)
    eye6 = torch.eye(6, device=dev, dtype=Gi_data.dtype) * eps
    for k in range(6):
        d = eye6[k].view(1, 6)
        Rip, pip = pose_to_body(Gi.retr(d).data)
        Jpi[:, k] = (imu_residual(Rip[0].to(dt_type), pip[0].to(dt_type),
                                  vi, Rj, pj, vj, g, pre) - r0) / eps
        Rjp, pjp = pose_to_body(Gj.retr(d).data)
        Jpj[:, k] = (imu_residual(Ri, pi, vi, Rjp[0].to(dt_type),
                                  pjp[0].to(dt_type), vj, g, pre) - r0) / eps
    return r0, Jpi, Jvi, Jpj, Jvj
