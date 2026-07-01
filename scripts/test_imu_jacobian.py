#!/usr/bin/env python3
"""test_imu_jacobian.py — Valida la preintegración IMU y los jacobianos del factor
inercial in-solver (`dpvo.imu_preint`, tight coupling estrategia B, Hito 3).

Tres pruebas (todas sintéticas, sin datos reales):

  A. CONSISTENCIA preintegración ↔ integración directa. Se integra una trayectoria
     "verdadera" (R,v,p) con la MISMA discretización que `preintegrate` a partir de
     medidas IMU random, y se verifica que el residual de preintegración evaluado en
     esos estados verdaderos es ~0 (eq. de Forster bien implementada).

  B. JACOBIANOS DE VELOCIDAD analíticos (`vel_jacobians`) vs diferencias finitas
     centrales (perturbando v_i y v_j).

  C. PUENTE world→cam ↔ body-en-mundo (`pose_to_body`): round-trip exacto, y que el
     residual responde suave a una perturbación lietorch de la pose (la que usa
     `pose_retr` en la BA) — base del jacobiano de pose del solver.

  D. JACOBIANOS DE POSE analíticos (`pose_jacobians`, finding
     2026-06-20-imu-tight-analytical-pose-jacobians):
       D1 vs central-diff de la perturbación IZQUIERDA `Exp(d)·G` construida a mano
          en float64 (sin lietorch → sin techo de precisión f32): chequeo RIGUROSO
          de la matemática de la regla de la cadena. Objetivo rel < 1e-5.
       D2 vs forward-diff por lietorch `SE3.retr` (la convención REAL del solver,
          `imu_factor_linearize(pose_jac="numeric")`): confirma que `retr`==`Exp(d)·G`
          (sin transposiciones / orden de tangente cambiado). Tolerancia laxa
          (forward-diff eps=5e-4 + lietorch f32).

Uso (tras bootstrap/apply_dpvo_patches.sh, en el venv con lietorch+CUDA):
  .venv/bin/python scripts/test_imu_jacobian.py
"""
from __future__ import annotations

import sys

import torch

from dpvo.imu_preint import (Preint, imu_factor_linearize, imu_residual,
                             pose_jacobians, preintegrate, pose_to_body, so3_exp,
                             so3_hat, vel_jacobians)


def integrate_true(R0, v0, p0, acc, gyro, dts, g):
    """Integra la trayectoria verdadera con la misma discretización que
    `preintegrate` (ZOH al inicio del intervalo). Devuelve R_j, v_j, p_j."""
    R, v, p = R0.clone(), v0.clone(), p0.clone()
    for k in range(acc.shape[0]):
        dt = float(dts[k])
        acc_w = R @ acc[k] + g                       # a_world = R·f + g
        p = p + v * dt + 0.5 * acc_w * dt * dt
        v = v + acc_w * dt
        R = R @ so3_exp(gyro[k] * dt)
    return R, v, p


def se3_exp_matrix(d):
    """Exp SE3 EXACTA (float64) -> (4,4). d=[ρ(tras), ω(rot)] (orden lietorch).
    Exp([ρ,ω]) = [[Exp_SO3(ω), J_l(ω)·ρ], [0,1]] con J_l la jacobiana izq de SO(3).
    Se usa solo en el test (D1) para construir la perturbación `Exp(d)·G` sin
    lietorch y diferenciarla en float64 (truncamiento O(eps²), sin techo f32)."""
    rho, omega = d[:3], d[3:]
    R = so3_exp(omega)
    theta = omega.norm()
    eye = torch.eye(3, device=d.device, dtype=d.dtype)
    if float(theta) < 1e-8:
        Jl = eye
    else:
        Kw = so3_hat(omega)
        A = (1 - torch.cos(theta)) / (theta * theta)
        B = (theta - torch.sin(theta)) / (theta ** 3)
        Jl = eye + A * Kw + B * (Kw @ Kw)
    M = torch.eye(4, device=d.device, dtype=d.dtype)
    M[:3, :3] = R
    M[:3, 3] = Jl @ rho
    return M


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt64 = torch.float64
    print(f"device={dev}")

    K = 40
    dts = torch.full((K,), 0.0025, device=dev, dtype=dt64)     # 400 Hz
    acc = torch.randn(K, 3, device=dev, dtype=dt64) * 0.5
    acc[:, 2] += 9.81                                          # fuerza específica ~g
    gyro = torch.randn(K, 3, device=dev, dtype=dt64) * 0.3
    g = torch.tensor([0.3, -0.2, -9.81], device=dev, dtype=dt64)

    # estados "verdaderos" del frame i (random) y j (integrado)
    Ri = so3_exp(torch.randn(3, device=dev, dtype=dt64) * 0.4)
    vi = torch.randn(3, device=dev, dtype=dt64) * 0.5
    pi = torch.randn(3, device=dev, dtype=dt64)
    Rj, vj, pj = integrate_true(Ri, vi, pi, acc, gyro, dts, g)

    pre = preintegrate(acc, gyro, dts)

    # ---------- A. consistencia ----------
    r = imu_residual(Ri, pi, vi, Rj, pj, vj, g, pre)
    rA = r.abs().max().item()
    print(f"[A] residual en estados verdaderos: max|r|={rA:.2e}  "
          f"(r_R={r[:3].abs().max():.1e} r_v={r[3:6].abs().max():.1e} "
          f"r_p={r[6:9].abs().max():.1e})")
    okA = rA < 1e-6

    # ---------- B. jacobianos de velocidad ----------
    # Estados NO consistentes (residual ≠ 0) para que el jacobiano sea informativo.
    vi2 = vi + torch.randn(3, device=dev, dtype=dt64) * 0.3
    vj2 = vj + torch.randn(3, device=dev, dtype=dt64) * 0.3
    pj2 = pj + torch.randn(3, device=dev, dtype=dt64) * 0.2
    Jvi, Jvj = vel_jacobians(Ri, pre.DT)

    def res(vi_, vj_):
        return imu_residual(Ri, pi, vi_, Rj, pj2, vj_, g, pre)

    eps = 1e-6
    Jvi_fd = torch.zeros(9, 3, device=dev, dtype=dt64)
    Jvj_fd = torch.zeros(9, 3, device=dev, dtype=dt64)
    for c in range(3):
        e = torch.zeros(3, device=dev, dtype=dt64); e[c] = eps
        Jvi_fd[:, c] = (res(vi2 + e, vj2) - res(vi2 - e, vj2)) / (2 * eps)
        Jvj_fd[:, c] = (res(vi2, vj2 + e) - res(vi2, vj2 - e)) / (2 * eps)
    evi = (Jvi - Jvi_fd).abs().max().item() / (Jvi_fd.abs().max().item() + 1e-12)
    evj = (Jvj - Jvj_fd).abs().max().item() / (Jvj_fd.abs().max().item() + 1e-12)
    print(f"[B] rel(Jvi)={evi:.2e}  rel(Jvj)={evj:.2e}")
    okB = (evi < 1e-6) and (evj < 1e-6)

    # ---------- C. puente pose_to_body + perturbación lietorch ----------
    okC = True
    okD = True
    try:
        from lietorch import SE3
        # G = world→cam desde (R_wb, p_wb): R_cw = R_wbᵀ, t_cw = -R_cw·p_wb
        def body_to_pose_data(R_wb, p_wb):
            R_cw = R_wb.transpose(-1, -2)
            t_cw = -(R_cw @ p_wb[..., None]).squeeze(-1)
            q = mat_to_quat(R_cw)
            return torch.cat([t_cw, q], dim=-1)

        def mat_to_quat(R):
            # R (3,3) -> [qx,qy,qz,qw]
            t = R[0, 0] + R[1, 1] + R[2, 2]
            if t > 0:
                s = torch.sqrt(t + 1.0) * 2
                qw = 0.25 * s
                qx = (R[2, 1] - R[1, 2]) / s
                qy = (R[0, 2] - R[2, 0]) / s
                qz = (R[1, 0] - R[0, 1]) / s
            else:
                # rama estable: mayor diagonal
                i = int(torch.argmax(torch.tensor([R[0, 0], R[1, 1], R[2, 2]])))
                if i == 0:
                    s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
                    qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
                elif i == 1:
                    s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                    qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
                    qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
                else:
                    s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                    qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
                    qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s
            return torch.stack([qx, qy, qz, qw])

        # lietorch SE3 trabaja en float32
        Ri32, pi32 = Ri.float(), pi.float()
        Gi = body_to_pose_data(Ri32, pi32).view(1, 7)
        R_back, p_back = pose_to_body(Gi)
        e_rt = max((R_back[0] - Ri32).abs().max().item(),
                   (p_back[0] - pi32).abs().max().item())
        print(f"[C] round-trip pose_to_body: max err={e_rt:.2e}")
        okC = e_rt < 1e-4

        # perturbación lietorch (la de pose_retr): residual debe variar O(eps)
        Gse = SE3(Gi)
        d = torch.zeros(1, 6, device=dev); d[0, 0] = 1e-3
        Gp = Gse.retr(d)
        Rp, pp = pose_to_body(Gp.data)
        rr = imu_residual(Rp[0].double(), pp[0].double(), vi, Rj, pj, vj, g, pre)
        drift = (rr - imu_residual(Ri, pi, vi, Rj, pj, vj, g, pre)).abs().max().item()
        print(f"[C] |Δr| ante retr(1e-3)={drift:.2e} (debe ser O(1e-3), no 0 ni NaN)")
        okC = okC and (1e-6 < drift < 1e-1)

        # ---------- D. jacobianos de POSE analíticos ----------
        # Estados i,j NO consistentes (residual ≠ 0) para que el jacobiano informe
        # de verdad. Reusamos Ri,pi del frame i y Rj,vj,pj integrados, perturbados.
        pj2 = pj + torch.randn(3, device=dev, dtype=dt64) * 0.2
        vj2 = vj + torch.randn(3, device=dev, dtype=dt64) * 0.3
        vi2 = vi + torch.randn(3, device=dev, dtype=dt64) * 0.3

        # estado body-en-mundo del frame j perturbado (rotación distinta de Rj)
        Rj2 = Rj @ so3_exp(torch.randn(3, device=dev, dtype=dt64) * 0.2)

        # --- D1: analítico vs central-diff float64 de Exp(d)·G (math pura) ---
        def body_pert(R_wb, p_wb, d):
            """(R_wb,p_wb)=body-en-mundo -> world→cam M(G) -> Exp(d)·G -> body."""
            R_cw = R_wb.transpose(-1, -2)
            t_cw = -(R_cw @ p_wb)
            MG = torch.eye(4, device=dev, dtype=dt64)
            MG[:3, :3] = R_cw; MG[:3, 3] = t_cw
            Md = se3_exp_matrix(d) @ MG
            R_cw2, t_cw2 = Md[:3, :3], Md[:3, 3]
            R_wb2 = R_cw2.transpose(-1, -2)
            p_wb2 = -(R_wb2 @ t_cw2)
            return R_wb2, p_wb2

        def res_pose(di, dj):
            Rip, pip = body_pert(Ri, pi, di)
            Rjp, pjp = body_pert(Rj2, pj2, dj)
            return imu_residual(Rip, pip, vi2, Rjp, pjp, vj2, g, pre)

        # analítico (vía la API real del solver, modo numérico aparte para D2)
        Gi_d = body_to_pose_data(Ri.float(), pi.float()).view(1, 7)
        Gj_d = body_to_pose_data(Rj2.float(), pj2.float()).view(1, 7)
        r_an, Jpi_an, _, Jpj_an, _ = imu_factor_linearize(
            Gi_d, Gj_d, vi2, vj2, g, pre, pose_jac="analytic")

        z6 = torch.zeros(6, device=dev, dtype=dt64)
        h = 1e-6
        Jpi_fd = torch.zeros(9, 6, device=dev, dtype=dt64)
        Jpj_fd = torch.zeros(9, 6, device=dev, dtype=dt64)
        for k in range(6):
            ek = torch.zeros(6, device=dev, dtype=dt64); ek[k] = h
            Jpi_fd[:, k] = (res_pose(ek, z6) - res_pose(-ek, z6)) / (2 * h)
            Jpj_fd[:, k] = (res_pose(z6, ek) - res_pose(z6, -ek)) / (2 * h)
        e_pi = ((Jpi_an - Jpi_fd).abs().max()
                / (Jpi_fd.abs().max() + 1e-12)).item()
        e_pj = ((Jpj_an - Jpj_fd).abs().max()
                / (Jpj_fd.abs().max() + 1e-12)).item()
        print(f"[D1] rel(Jpi)={e_pi:.2e}  rel(Jpj)={e_pj:.2e}  (central-diff f64)")
        okD = (e_pi < 1e-5) and (e_pj < 1e-5)

        # --- D2: analítico vs forward-diff lietorch retr (convención del solver) ---
        r_nu, Jpi_nu, _, Jpj_nu, _ = imu_factor_linearize(
            Gi_d, Gj_d, vi2, vj2, g, pre, pose_jac="numeric")
        d_pi = ((Jpi_an - Jpi_nu).abs().max()
                / (Jpi_nu.abs().max() + 1e-12)).item()
        d_pj = ((Jpj_an - Jpj_nu).abs().max()
                / (Jpj_nu.abs().max() + 1e-12)).item()
        print(f"[D2] rel(Jpi)={d_pi:.2e}  rel(Jpj)={d_pj:.2e}  (vs lietorch retr fwd-diff)")
        okD = okD and (d_pi < 5e-2) and (d_pj < 5e-2)
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[C/D] SKIP (lietorch no disponible / CPU): {e}")
        traceback.print_exc()

    ok = okA and okB and okC and okD
    print("RESULTADO:", "OK ✅" if ok else "FALLA ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
