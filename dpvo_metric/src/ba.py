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

def BA(poses, patches, intrinsics, targets, weights, lmbda, ii, jj, kk, bounds, ep=100.0, PRINT=False, fixedp=1, structure_only=False, prior_d=None, prior_w=None, prior_strength=0.0, quality_boost=1.0):
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
    """

    b = 1
    n = max(ii.max().item(), jj.max().item()) + 1

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

    # --- variante C in-solver: prior unario de profundidad (curvatura -> Schur) ---
    if (prior_w is not None) and (prior_d is not None) and (prior_strength > 0):
        pw = prior_w[kx]                                   # (m,) validez 0/1
        if torch.any(pw > 0):
            Cdiag = C.view(b, m)[0]                        # (m,) Hessian de datos
            cref = Cdiag[pw > 0].median() if (pw > 0).any() else C.new_tensor(1.0)
            wp = (prior_strength * cref * pw).view(b, m, 1, 1)
            d_cur = patches[:, kx, 2, p // 2, p // 2].view(b, m, 1, 1)
            d_zed = prior_d[kx].view(b, m, 1, 1)
            C = C + wp                                     # +curvatura en profundidad
            w = w + wp * (d_zed - d_cur)                   # +gradiente del prior

    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)
        
    Q = 1.0 / (C + lmbda)
    
    ### solve w/ schur complement ###
    EQ = E * Q[:,None]

    if structure_only or n == 0:
        dZ = (Q * w).view(b, -1, 1, 1)

    else:
        S = B - block_matmul(EQ, E.permute(0,2,1,4,3))
        y = v - block_matmul(EQ, w.unsqueeze(dim=2))
        dX = block_solve(S, y, ep=ep, lm=1e-4)

        dZ = Q * (w - block_matmul(E.permute(0,2,1,4,3), dX).squeeze(dim=-1))
        dX = dX.view(b, -1, 6)
        dZ = dZ.view(b, -1, 1, 1)

    x, y, disps = patches.unbind(dim=2)
    disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    patches = torch.stack([x, y, disps], dim=2)

    if not structure_only and n > 0:
        poses = pose_retr(poses, dX, fixedp + torch.arange(n))

    return poses, patches
