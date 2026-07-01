import torch
import numpy as np
import torch.nn.functional as F

from . import fastba
from . import altcorr
import lietorch
from lietorch import SE3

from .net import VONet
from .utils import *
from . import projective_ops as pops

autocast = torch.autocast
Id = SE3.Identity(1, device="cuda")


class DPVO:
    def __init__(self, cfg, network, ht=480, wd=640):
        self.cfg = cfg
        self.load_weights(network)
        self.is_initialized = False
        self.enable_timing = False
        
        self.n = 0      # number of frames
        self.m = 0      # number of patches
        self.M = self.cfg.PATCHES_PER_FRAME
        self.N = self.cfg.BUFFER_SIZE

        self.ht = ht    # image height
        self.wd = wd    # image width

        DIM = self.DIM
        RES = self.RES

        ### state attributes ###
        self.tlist = []
        self.counter = 0

        self.tstamps_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.poses_ = torch.zeros(self.N, 7, dtype=torch.float, device="cuda")
        self.patches_ = torch.zeros(self.N, self.M, 3, self.P, self.P, dtype=torch.float, device="cuda")
        self.intrinsics_ = torch.zeros(self.N, 4, dtype=torch.float, device="cuda")

        self.points_ = torch.zeros(self.N * self.M, 3, dtype=torch.float, device="cuda")
        self.colors_ = torch.zeros(self.N, self.M, 3, dtype=torch.uint8, device="cuda")

        self.index_ = torch.zeros(self.N, self.M, dtype=torch.long, device="cuda")
        self.index_map_ = torch.zeros(self.N, dtype=torch.long, device="cuda")

        # variante C (prior blando de profundidad): profundidad inversa medida
        # del ZED (1/Z) por parche + peso de validez, alineados por slot de
        # frame con patches_ (se desplazan junto a el en keyframe()). Solo se
        # llenan/usan en depth_inject_mode == "prior". Ver finding
        # 2026-06-05-dpvo-metric-depth-injection-prototype (variante C).
        self.disps_prior_ = torch.zeros(self.N, self.M, device="cuda")
        self.disps_prior_w_ = torch.zeros(self.N, self.M, device="cuda")
        self.depth_prior_alpha = 0.0      # blend post-BA (modo "prior", gauge-libre)
        self.depth_prior_strength = 0.0   # factor unario in-solver (modo "prior_insolver")

        # factor de plano in-solver (P2 Hito 3): plano cam-local `[n,d]` por frame
        # (estimado fuera de la BA desde la nube estéreo validada — RANSAC en el
        # runner) + validez. Alineado por slot con patches_ (se desplaza en
        # keyframe()). En la BA se transforma a coords del MUNDO con la pose del
        # frame de referencia y entra como restricción de coplanaridad BLANDA
        # JUNTO al prior de depth (no en vez de él). Ver handoff 2026-06-18.
        self.plane_n_ = torch.zeros(self.N, 3, device="cuda")    # normal cam-local (|n|=1)
        self.plane_d_ = torch.zeros(self.N, device="cuda")       # offset cam-local
        self.plane_w_ = torch.zeros(self.N, device="cuda")       # 1 si el frame tiene plano
        self.plane_strength = 0.0         # fuerza del factor de plano (0 = off)
        self.plane_trim = 0.0             # trim por distancia euclídea al plano (m)
        self.plane_set_counter = 0        # ++ cada vez que el runner setea un plano
        self._plane_world_cache = None    # π_world congelado entre refits
        self._plane_world_cache_id = -1
        # B1 (planos LOCALES por ventana): "frozen" = P2 (plano per-frame
        # congelado, global); "window" = RANSAC fresco sobre los puntos 3-D de
        # los parches de la ventana activa cada BA. Ver handoff B1 Hito 3.
        self.plane_mode = "frozen"
        self.plane_inlier_thresh = 0.08   # umbral RANSAC del plano de ventana (m)
        # Gap 2 Hito 3 — plano VERTICAL por gravedad: si plane_vertical y plane_grav
        # (3,) están seteados, el RANSAC del plano de ventana restringe las hipótesis
        # a planos VERTICALES (normal ⊥ gravedad). La red de la jaula es vertical →
        # quita 1 DOF a la normal ruidosa (que era la varianza que hundió a B1). Usa
        # SOLO el acelerómetro (la gravedad mundo) → NO necesita gyro → corre en
        # video_4 (gyro muerto). Ver finding 2026-06-20-underwater-svo-no-gyro.
        self.plane_vertical = False
        self.plane_grav = None            # (3,) gravedad en el mundo (frame 0 = identidad)

        # tight coupling IMU (estrategia B Hito 3): velocidad body-en-mundo por
        # keyframe (estado EXTRA de la BA aumentada) + timestamp real (ns) por slot
        # para alinear con el stream IMU. Alineados con poses_; se desplazan en
        # keyframe(). El stream IMU (frame cámara) y la gravedad mundo los setea el
        # runner. El factor de preintegración vive en imu_preint.py + ba.py.
        # Ver finding 2026-06-19-ekf-imu-loose-coupling + handoff tight coupling.
        self.vels_ = torch.zeros(self.N, 3, device="cuda")
        self.frame_ts_ = torch.zeros(self.N, dtype=torch.long, device="cuda")
        self.imu_tight = False
        self.imu_ts = None        # (Msamp,) long ns ORDENADO
        self.imu_acc = None       # (Msamp,3) m/s² frame cámara
        self.imu_gyr = None       # (Msamp,3) rad/s frame cámara
        self.imu_g = None         # (3,) gravedad en el mundo (frame 0 = identidad)
        self.imu_sig_g = 0.01     # ruido gyro (peso relativo del residual r_ΔR)
        self.imu_sig_a = 0.2      # ruido acc  (peso de r_Δv / r_Δp)
        self.imu_strength = 1.0   # fuerza global del factor IMU (como prior_strength)
        # guard de divergencia (robustez, paso 1 Hito 3): cap físico de ‖v‖ que
        # mata la dirección degenerada que infla la escala, + cap de paso de pose
        # que cae a BA visual. 0=off. Ver finding tight-coupling-robustness.
        self.imu_v_max = 0.0      # m/s; máx ‖v‖ de keyframe (proyección post-solve; REFUTADO)
        self.imu_max_step = 0.0   # m; máx traslación de pose por paso (si no, fallback)
        self.imu_v_reg = 0.0      # Tikhonov in-solver sobre ‖v‖ (anti-drift de escala)
        self.imu_n_clamp = 0      # contador: velocidades acotadas
        self.imu_n_reject = 0     # contador: pasos IMU rechazados (fallback visual)
        self._preint_cache = {}   # (ts_i,ts_j) -> Preint (medidas fijas: se cachea)

        ### network attributes ###
        self.mem = 32

        if self.cfg.MIXED_PRECISION:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.half}
        else:
            self.kwargs = kwargs = {"device": "cuda", "dtype": torch.float}
        
        self.imap_ = torch.zeros(self.mem, self.M, DIM, **kwargs)
        self.gmap_ = torch.zeros(self.mem, self.M, 128, self.P, self.P, **kwargs)

        ht = ht // RES
        wd = wd // RES

        self.fmap1_ = torch.zeros(1, self.mem, 128, ht // 1, wd // 1, **kwargs)
        self.fmap2_ = torch.zeros(1, self.mem, 128, ht // 4, wd // 4, **kwargs)

        # feature pyramid
        self.pyramid = (self.fmap1_, self.fmap2_)

        self.net = torch.zeros(1, 0, DIM, **kwargs)
        self.ii = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.jj = torch.as_tensor([], dtype=torch.long, device="cuda")
        self.kk = torch.as_tensor([], dtype=torch.long, device="cuda")
        
        # initialize poses to identity matrix
        self.poses_[:,6] = 1.0

        # store relative poses for removed frames
        self.delta = {}

    def load_weights(self, network):
        # load network from checkpoint file
        if isinstance(network, str):
            from collections import OrderedDict
            state_dict = torch.load(network)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "update.lmbda" not in k:
                    new_state_dict[k.replace('module.', '')] = v
            
            self.network = VONet()
            self.network.load_state_dict(new_state_dict)

        else:
            self.network = network

        # steal network attributes
        self.DIM = self.network.DIM
        self.RES = self.network.RES
        self.P = self.network.P

        self.network.cuda()
        self.network.eval()

        # if self.cfg.MIXED_PRECISION:
        #     self.network.half()


    @property
    def poses(self):
        return self.poses_.view(1, self.N, 7)

    @property
    def patches(self):
        return self.patches_.view(1, self.N*self.M, 3, 3, 3)

    @property
    def intrinsics(self):
        return self.intrinsics_.view(1, self.N, 4)

    @property
    def ix(self):
        return self.index_.view(-1)

    @property
    def imap(self):
        return self.imap_.view(1, self.mem * self.M, self.DIM)

    @property
    def gmap(self):
        return self.gmap_.view(1, self.mem * self.M, 128, 3, 3)

    def get_pose(self, t):
        if t in self.traj:
            return SE3(self.traj[t])

        t0, dP = self.delta[t]
        return dP * self.get_pose(t0)

    def terminate(self):
        """ interpolate missing poses """
        self.traj = {}
        for i in range(self.n):
            self.traj[self.tstamps_[i].item()] = self.poses_[i]

        poses = [self.get_pose(t) for t in range(self.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(self.tlist, dtype=float)

        return poses, tstamps

    def corr(self, coords, indicies=None):
        """ local correlation volume """
        ii, jj = indicies if indicies is not None else (self.kk, self.jj)
        ii1 = ii % (self.M * self.mem)
        jj1 = jj % (self.mem)
        corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
        corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
        return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)

    def reproject(self, indicies=None):
        """ reproject patch k from i -> j """
        (ii, jj, kk) = indicies if indicies is not None else (self.ii, self.jj, self.kk)
        coords = pops.transform(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk)
        return coords.permute(0, 1, 4, 2, 3).contiguous()

    def append_factors(self, ii, jj):
        self.jj = torch.cat([self.jj, jj])
        self.kk = torch.cat([self.kk, ii])
        self.ii = torch.cat([self.ii, self.ix[ii]])

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        self.net = torch.cat([self.net, net], dim=1)

    def remove_factors(self, m):
        self.ii = self.ii[~m]
        self.jj = self.jj[~m]
        self.kk = self.kk[~m]
        self.net = self.net[:,~m]

    def motion_probe(self):
        """ kinda hacky way to ensure enough motion for initialization """
        kk = torch.arange(self.m-self.M, self.m, device="cuda")
        jj = self.n * torch.ones_like(kk)
        ii = self.ix[kk]

        net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
        coords = self.reproject(indicies=(ii, jj, kk))

        with autocast(enabled=self.cfg.MIXED_PRECISION, device_type="cuda"):
            corr = self.corr(coords, indicies=(kk, jj))
            ctx = self.imap[:,kk % (self.M * self.mem)]
            net, (delta, weight, _) = \
                self.network.update(net, ctx, corr, None, ii, jj, kk)

        return torch.quantile(delta.norm(dim=-1).float(), 0.5)

    def motionmag(self, i, j):
        k = (self.ii == i) & (self.jj == j)
        ii = self.ii[k]
        jj = self.jj[k]
        kk = self.kk[k]

        flow = pops.flow_mag(SE3(self.poses), self.patches, self.intrinsics, ii, jj, kk, beta=0.5)
        return flow.mean().item()

    def keyframe(self):

        i = self.n - self.cfg.KEYFRAME_INDEX - 1
        j = self.n - self.cfg.KEYFRAME_INDEX + 1
        m = self.motionmag(i, j) + self.motionmag(j, i)
 
        if m / 2 < self.cfg.KEYFRAME_THRESH:
            k = self.n - self.cfg.KEYFRAME_INDEX
            t0 = self.tstamps_[k-1].item()
            t1 = self.tstamps_[k].item()

            dP = SE3(self.poses_[k]) * SE3(self.poses_[k-1]).inv()
            self.delta[t1] = (t0, dP)

            to_remove = (self.ii == k) | (self.jj == k)
            self.remove_factors(to_remove)

            self.kk[self.ii > k] -= self.M
            self.ii[self.ii > k] -= 1
            self.jj[self.jj > k] -= 1

            for i in range(k, self.n-1):
                self.tstamps_[i] = self.tstamps_[i+1]
                self.colors_[i] = self.colors_[i+1]
                self.poses_[i] = self.poses_[i+1]
                self.patches_[i] = self.patches_[i+1]
                self.disps_prior_[i] = self.disps_prior_[i+1]
                self.disps_prior_w_[i] = self.disps_prior_w_[i+1]
                self.plane_n_[i] = self.plane_n_[i+1]
                self.plane_d_[i] = self.plane_d_[i+1]
                self.plane_w_[i] = self.plane_w_[i+1]
                self.vels_[i] = self.vels_[i+1]
                self.frame_ts_[i] = self.frame_ts_[i+1]
                self.intrinsics_[i] = self.intrinsics_[i+1]

                self.imap_[i%self.mem] = self.imap_[(i+1) % self.mem]
                self.gmap_[i%self.mem] = self.gmap_[(i+1) % self.mem]
                self.fmap1_[0,i%self.mem] = self.fmap1_[0,(i+1)%self.mem]
                self.fmap2_[0,i%self.mem] = self.fmap2_[0,(i+1)%self.mem]

            self.n -= 1
            self.m-= self.M

        to_remove = self.ix[self.kk] < self.n - self.cfg.REMOVAL_WINDOW
        self.remove_factors(to_remove)

    def update(self):
        with Timer("other", enabled=self.enable_timing):
            coords = self.reproject()

            with autocast(enabled=True, device_type="cuda"):
                corr = self.corr(coords)
                ctx = self.imap[:,self.kk % (self.M * self.mem)]
                self.net, (delta, weight, _) = \
                    self.network.update(self.net, ctx, corr, None, self.ii, self.jj, self.kk)

            lmbda = torch.as_tensor([1e-4], device="cuda")
            weight = weight.float()
            target = coords[...,self.P//2,self.P//2] + delta.float()

        with Timer("BA", enabled=self.enable_timing):
            t0 = self.n - self.cfg.OPTIMIZATION_WINDOW if self.is_initialized else 1
            t0 = max(t0, 1)

            mode = getattr(self, "depth_inject_mode", "off")
            if mode == "prior_insolver":
                # variante C in-solver: BA de Python con factor unario de depth
                self._ba_python_prior(target, weight, t0)
            else:
                try:
                    fastba.BA(self.poses, self.patches, self.intrinsics,
                        target, weight, lmbda, self.ii, self.jj, self.kk, t0, self.n, 2)
                except:
                    print("Warning BA failed...")

                # variante C (post-paso): ancla blanda de la profundidad (gauge-libre)
                if mode == "prior":
                    self._apply_depth_prior(t0)

            points = pops.point_cloud(SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m])
            points = (points[...,1,1,:3] / points[...,1,1,3:]).reshape(-1, 3)
            self.points_[:len(points)] = points[:]
                
    def __edges_all(self):
        return flatmeshgrid(
            torch.arange(0, self.m, device="cuda"),
            torch.arange(0, self.n, device="cuda"), indexing='ij')

    def __edges_forw(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - r), 0)
        t1 = self.M * max((self.n - 1), 0)
        return flatmeshgrid(
            torch.arange(t0, t1, device="cuda"),
            torch.arange(self.n-1, self.n, device="cuda"), indexing='ij')

    def __edges_back(self):
        r=self.cfg.PATCH_LIFETIME
        t0 = self.M * max((self.n - 1), 0)
        t1 = self.M * max((self.n - 0), 0)
        return flatmeshgrid(torch.arange(t0, t1, device="cuda"),
            torch.arange(max(self.n-r, 0), self.n, device="cuda"), indexing='ij')

    def _seed_inverse_depth(self, patches, depth):
        """Siembra la profundidad inversa de los parches desde un mapa de
        profundidad métrica (metros).

        `patches`: (1, M, 3, P, P) con coords de imagen (x, y) a resolución de
        feature en los canales 0,1. `depth`: (H, W) float en metros a la
        resolución de la imagen de entrada (pre-RES). La profundidad inválida
        (<=0, nan, inf) se rellena con la mediana de los parches válidos.
        Retorna (1, M, 1, 1) profundidad inversa clampeada al rango del BA
        [1e-3, 10]. Punto de inyección: ver finding
        2026-06-05-dpvo-sigma-vs-macvo-covariance-audit.
        """
        P = patches.shape[-1]
        cx = patches[0, :, 0, P // 2, P // 2]   # x a feature-res
        cy = patches[0, :, 1, P // 2, P // 2]   # y a feature-res
        H, W = depth.shape[-2], depth.shape[-1]
        u = (self.RES * cx).round().long().clamp(0, W - 1)
        v = (self.RES * cy).round().long().clamp(0, H - 1)
        Z = depth[v, u].float()
        valid = torch.isfinite(Z) & (Z > 1e-3)
        d = torch.zeros_like(Z)
        d[valid] = 1.0 / Z[valid]
        if valid.any():
            d[~valid] = d[valid].median()
        else:
            d[:] = 1.0
        d = d.clamp(min=1e-3, max=10.0)
        return d[None, :, None, None], valid

    def _apply_depth_prior(self, t0):
        """Variante C — prior blando de profundidad (post-paso BA).

        Tras un paso de `fastba.BA`, empuja la profundidad inversa de los
        parches con prior valido hacia `1/Z_zed` con peso `alpha`:
            disps <- (1 - alpha*w) * disps_BA + (alpha*w) * d_zed
        `alpha=1` fija (hard) la depth sembrada -> rompe la libertad de gauge
        que deja la siembra-como-init (fase 2: el BA respeta el seed pero el
        neto sigue gauge-libre); `alpha<1` la ancla de forma blanda. Opera solo
        sobre la ventana de optimizacion [t0, n). Python puro (sin tocar CUDA).
        Ver finding 2026-06-05-dpvo-metric-depth-injection-prototype (variante C).
        """
        a = float(getattr(self, "depth_prior_alpha", 0.0))
        if a <= 0.0:
            return
        lo, hi = max(int(t0), 0), self.n
        if hi <= lo:
            return
        w = self.disps_prior_w_[lo:hi]                 # (F, M)
        if not bool(torch.any(w > 0)):
            return
        aw = (a * w)[..., None, None]                  # (F, M, 1, 1)
        cur = self.patches_[lo:hi, :, 2]               # (F, M, P, P)
        d_prior = self.disps_prior_[lo:hi][..., None, None]
        blended = (1.0 - aw) * cur + aw * d_prior
        self.patches_[lo:hi, :, 2] = blended.clamp(min=1e-3, max=10.0)

    def _body_pos(self, i):
        """Posición de la cámara (body) en el mundo del slot i: p_wb = -R(G)ᵀ t(G),
        con G = poses_[i] (world→cam)."""
        M = SE3(self.poses_[i][None]).matrix()[0]
        R = M[:3, :3]
        t = M[:3, 3]
        return -(R.transpose(-1, -2) @ t)

    def _build_imu_factors(self, t0):
        """Arma el dict de factores IMU para la BA aumentada (tight coupling):
        preintegración entre cada par de keyframes consecutivos de la ventana
        [t0-1, n) (incluye el borde fijo→libre). Devuelve None si no hay pares o
        falta el stream IMU. Las medidas son fijas → la preintegración se cachea."""
        from .imu_preint import segment_preint
        if self.imu_ts is None or self.imu_g is None:
            return None
        n = self.n
        lo = max(int(t0) - 1, 0)
        if n - lo < 2:
            return None
        preints = []
        cache = self._preint_cache
        for i in range(lo, n - 1):
            ti = int(self.frame_ts_[i]); tj = int(self.frame_ts_[i + 1])
            if tj <= ti:
                continue
            key = (ti, tj)
            pre = cache.get(key)
            if pre is None:
                pre = segment_preint(self.imu_ts, self.imu_acc, self.imu_gyr, ti, tj)
                cache[key] = pre
            preints.append((i, i + 1, pre))
        if not preints:
            return None
        return {"preints": preints, "vels": self.vels_[:n], "g": self.imu_g,
                "sig_g": self.imu_sig_g, "sig_a": self.imu_sig_a,
                "strength": self.imu_strength,
                "v_max": self.imu_v_max, "max_step": self.imu_max_step,
                "v_reg": self.imu_v_reg}

    def _ba_python_prior(self, target, weight, t0):
        """Variante C in-solver: corre la BA de Python (`ba.py`) con un factor
        unario de profundidad DENTRO del solver (en vez de `fastba` + blend
        post-paso). El prior agrega curvatura a la dimension de profundidad y, via
        Schur, fuerza poses metricas -> rompe el modo plano del gauge mono que el
        blend post-BA no podia tocar. Mas lento que `fastba` (x86 research). Ver
        finding 2026-06-05-dpvo-variant-c-scale-nondeterministic.
        """
        from .ba import BA as _pyBA
        bounds = [-64, -64, self.wd // self.RES + 64, self.ht // self.RES + 64]
        Gs = SE3(self.poses)
        patches = self.patches
        prior_d = self.disps_prior_.view(-1)
        prior_w = self.disps_prior_w_.view(-1)
        strength = float(getattr(self, "depth_prior_strength", 0.0))
        qboost = float(getattr(self, "depth_quality_boost", 1.0))
        # factor de plano in-solver (P2): transforma el plano cam-local del frame
        # de referencia a coords del MUNDO con su pose actual y lo mantiene FIJO
        # durante el solve (alternancia a cadencia de frame). Ver handoff 2026-06-18.
        plane_strength = float(getattr(self, "plane_strength", 0.0))
        if plane_strength > 0:
            if getattr(self, "plane_mode", "frozen") == "window":
                plane = self._fit_window_plane(Gs, t0)   # B1: plano local fresco
            else:
                plane = self._world_plane(Gs)            # P2: plano per-frame congelado
        else:
            plane = None
        plane_trim = float(getattr(self, "plane_trim", 0.0))
        patch_anchor = self.ix if plane is not None else None
        use_imu = bool(getattr(self, "imu_tight", False)) and self.is_initialized
        for _ in range(2):
            imu = self._build_imu_factors(t0) if use_imu else None
            out = _pyBA(
                Gs, patches, self.intrinsics, target, weight, 1e-4,
                self.ii, self.jj, self.kk, bounds, ep=10.0, fixedp=t0,
                structure_only=False, prior_d=prior_d, prior_w=prior_w,
                prior_strength=strength, quality_boost=qboost,
                plane=plane, plane_strength=(plane_strength if plane is not None else 0.0),
                plane_trim=plane_trim, patch_anchor=patch_anchor, imu=imu)
            if imu is not None:
                Gs, patches, vels_new = out
                if vels_new is not None:
                    self.vels_[:vels_new.shape[0]] = vels_new
                self.imu_n_clamp += int(imu.get("_nclamp", 0))
                self.imu_n_reject += int(imu.get("_reject", 0))
            else:
                Gs, patches = out
        self.poses_[:] = Gs.data[0]
        self.patches_[:] = patches.view(self.N, self.M, 3, self.P, self.P)

    def _world_plane(self, Gs):
        """Plano (4,) `[n,d]` en coords del MUNDO desde el plano cam-local del frame
        de referencia (el más reciente con plano válido). El plano cam π_c cumple
        π_c·X_cam=0 con X_cam=G·X_world (G mundo→cam) → π_world=Gᵀ·π_c. Devuelve None
        si no hay plano válido.

        **Plano FIJO + alternancia** (spec del handoff 2026-06-18): π_world se
        CONGELA al setear un plano nuevo (contador `plane_set_counter` en __call__)
        y se reusa entre refits, en vez de re-derivarlo cada update() desde la pose
        (que evoluciona) del frame de referencia. Eso evita el "chase" de un plano
        que salta frame a frame (que desestabilizaba y acotaba el strength útil).
        La cadencia de refit la fija --plane-refit-every del runner.
        """
        cnt = int(getattr(self, "plane_set_counter", 0))
        if (getattr(self, "_plane_world_cache_id", -1) == cnt
                and getattr(self, "_plane_world_cache", None) is not None):
            return self._plane_world_cache                 # congelado entre refits
        valid = (self.plane_w_[:self.n] > 0).nonzero(as_tuple=False)
        if valid.numel() == 0:
            return None
        f = int(valid[-1].item())                          # frame de referencia más reciente
        pi_cam = torch.cat([self.plane_n_[f], self.plane_d_[f].view(1)])   # (4,)
        M = Gs[:, f].matrix()[0]                            # (4,4) mundo→cam del frame f
        pi_world = M.transpose(-1, -2) @ pi_cam            # (4,) plano en el mundo
        self._plane_world_cache = pi_world
        self._plane_world_cache_id = cnt
        return pi_world

    def _fit_window_plane(self, Gs, t0):
        """B1 (planos LOCALES): ajusta un plano FRESCO por RANSAC a los puntos 3-D
        de los parches de la VENTANA activa [t0, n) en coords del MUNDO, cada BA.
        A diferencia de `_world_plane` (P2: un plano per-frame congelado/global), se
        re-estima desde los puntos YA optimizados de la ventana → nunca obsoleto,
        robusto al ruido per-frame, y respeta la geometría orbital (cada ventana de
        la órbita ≈ un plano local). NO usa los buffers `plane_*` ni depth del SDK:
        el plano sale de los propios parches anclados (selectos+texturados+con prior
        de depth válido). Devuelve π=[n,d] (4,), |n|=1, o None. Ver handoff B1 Hito 3.
        """
        m = self.m
        if m < 30:
            return None
        P = self.P
        pc = pops.point_cloud(Gs, self.patches[:, :m], self.intrinsics, self.ix[:m])
        pts4 = pc[0, :, P // 2, P // 2, :]                 # (m, 4) homogéneo [X,Y,Z,W=1/Z]
        w = pts4[:, 3]
        xyz = pts4[:, :3] / w.clamp(min=1e-6)[:, None]     # (m, 3) euclídeo en el mundo
        anchor = self.ix[:m]
        pw = self.disps_prior_w_.view(-1)[:m]
        sel = ((anchor >= t0) & (anchor < self.n) & (pw > 0) & (w > 1e-6)
               & torch.isfinite(xyz).all(-1))
        Q = xyz[sel]
        if Q.shape[0] < 30:
            return None
        grav = (self.plane_grav if bool(getattr(self, "plane_vertical", False))
                and getattr(self, "plane_grav", None) is not None else None)
        return self._ransac_plane(
            Q, iters=64, thresh=float(getattr(self, "plane_inlier_thresh", 0.08)),
            grav=grav)

    @staticmethod
    def _ransac_plane(pts, iters=64, thresh=0.08, grav=None):
        """RANSAC de plano sobre (K,3) torch → π=[n,d] (4,), |n|=1, n·X+d=0; refit
        SVD sobre inliers. None si no converge (<20 inliers).

        Si `grav` (3,) no es None, restringe las hipótesis a planos VERTICALES
        (normal ⊥ gravedad): la red de la jaula cuelga vertical → quita 1 DOF a la
        normal y elimina la libertad del RANSAC de inclinarse a ajustar el "queso
        suizo"/alias (que era la varianza que hundió a B1). Gap 2 Hito 3 — usa solo
        el acelerómetro (no necesita gyro). El hipótesis vertical desde 2 puntos a,b:
        n = (b−a) × ĝ (⊥ gravedad y contiene a,b); el refit fuerza la normal al
        subespacio horizontal ⊥ ĝ (PCA 2-D de menor varianza)."""
        K = pts.shape[0]
        ghat = None
        if grav is not None:
            gn = grav.norm()
            if gn < 1e-9:
                grav = None
            else:
                ghat = (grav / gn).to(pts)
        best_inl, best_c = None, -1
        for _ in range(iters):
            if ghat is not None:
                idx = torch.randint(0, K, (2,), device=pts.device)
                a, b = pts[idx[0]], pts[idx[1]]
                nrm = torch.cross(b - a, ghat, dim=-1)    # ⊥ gravedad y contiene a,b
                anchor = a
            else:
                idx = torch.randint(0, K, (3,), device=pts.device)
                a, b, c = pts[idx[0]], pts[idx[1]], pts[idx[2]]
                nrm = torch.cross(b - a, c - a, dim=-1)
                anchor = a
            nn = nrm.norm()
            if nn < 1e-9:
                continue
            nrm = nrm / nn
            d = -(nrm * anchor).sum()
            inl = ((pts * nrm).sum(-1) + d).abs() < thresh
            cnt = int(inl.sum())
            if cnt > best_c:
                best_c, best_inl = cnt, inl
        if best_inl is None or best_c < 20:
            return None
        Qc = pts[best_inl]
        cen = Qc.mean(0)
        if ghat is not None:
            # refit VERTICAL: normal en el subespacio horizontal ⊥ gravedad (PCA 2-D)
            ref = torch.tensor([1.0, 0.0, 0.0], device=pts.device, dtype=pts.dtype)
            e1 = torch.cross(ghat, ref, dim=-1)
            if e1.norm() < 1e-6:
                ref = torch.tensor([0.0, 1.0, 0.0], device=pts.device, dtype=pts.dtype)
                e1 = torch.cross(ghat, ref, dim=-1)
            e1 = e1 / e1.norm()
            e2 = torch.cross(ghat, e1, dim=-1); e2 = e2 / e2.norm()
            B = torch.stack([e1, e2], 0)                   # (2,3) base horizontal
            XY = (Qc - cen) @ B.T                           # (K,2) proyección horizontal
            _, _, Vh = torch.linalg.svd(XY, full_matrices=False)
            n2 = Vh[-1]                                     # (2,) dir. de menor varianza
            nrm = n2[0] * e1 + n2[1] * e2
            nrm = nrm / nrm.norm()
        else:
            _, _, Vh = torch.linalg.svd(Qc - cen, full_matrices=False)
            nrm = Vh[-1]
            nrm = nrm / nrm.norm()
        d = -(nrm * cen).sum()
        return torch.cat([nrm, d.view(1)]).to(pts)

    def __call__(self, tstamp, image, intrinsics, depth=None, plane=None, frame_ts=None):
        """ track new frame """

        if (self.n+1) >= self.N:
            raise Exception(f'The buffer size is too small. You can increase it using "--buffer {self.N*2}"')

        image = 2 * (image[None,None] / 255.0) - 0.5
        
        with autocast(enabled=self.cfg.MIXED_PRECISION, device_type="cuda"):
            fmap, gmap, imap, patches, _, clr = \
                self.network.patchify(image,
                    patches_per_image=self.cfg.PATCHES_PER_FRAME,
                    gradient_bias=self.cfg.GRADIENT_BIAS,
                    selection=getattr(self.cfg, "PATCH_SELECTION", None),
                    oversample=getattr(self.cfg, "PATCH_OVERSAMPLE", 8),
                    depth=depth,
                    return_color=True)

        ### update state attributes ###
        self.tlist.append(tstamp)
        self.tstamps_[self.n] = self.counter
        if frame_ts is not None:
            self.frame_ts_[self.n] = int(frame_ts)   # tiempo real (ns) p/ IMU
        self.intrinsics_[self.n] = intrinsics / self.RES

        # color info for visualization
        clr = (clr[0,:,[2,1,0]] + 0.5) * (255.0 / 2)
        self.colors_[self.n] = clr.to(torch.uint8)

        self.index_[self.n + 1] = self.n + 1
        self.index_map_[self.n + 1] = self.m + self.M

        if self.n > 1:
            if self.cfg.MOTION_MODEL == 'DAMPED_LINEAR':
                P1 = SE3(self.poses_[self.n-1])
                P2 = SE3(self.poses_[self.n-2])                
                
                xi = self.cfg.MOTION_DAMPING * (P1 * P2.inv()).log()
                tvec_qvec = (SE3.exp(xi) * P1).data
                self.poses_[self.n] = tvec_qvec
            else:
                tvec_qvec = self.poses[self.n-1]
                self.poses_[self.n] = tvec_qvec

        # --- init de velocidad body-en-mundo (tight coupling): diferencia finita
        # de la posición VO del frame anterior. Es solo semilla; la BA aumentada
        # la refina como variable libre. ---
        if self.imu_tight and self.n >= 1 and frame_ts is not None:
            dt = (int(self.frame_ts_[self.n]) - int(self.frame_ts_[self.n-1])) / 1e9
            if dt > 1e-4:
                self.vels_[self.n] = (self._body_pos(self.n) - self._body_pos(self.n-1)) / dt
            else:
                self.vels_[self.n] = self.vels_[self.n-1]

        # --- siembra de profundidad inversa de los parches (init de escala) ---
        # Default DPVO: profundidad inversa aleatoria -> escala-ambiguo (mono).
        # Inyección métrica opcional: si se entrega `depth` (metros, misma
        # resolución que `image`) y `depth_inject_mode` está activo, sembrar
        # d = 1/Z -> recupera escala métrica. Modos: off | init | always.
        inject_mode = getattr(self, "depth_inject_mode", "off")
        do_inject = (depth is not None) and (
            inject_mode in ("always", "prior", "prior_insolver")
            or (inject_mode == "init" and not self.is_initialized)
        )
        if do_inject:
            d_seed, valid_seed = self._seed_inverse_depth(patches, depth)
            patches[:,:,2] = d_seed
            if inject_mode in ("prior", "prior_insolver"):
                # guardar el prior por parche (variante C). "prior": se mezcla
                # post-paso (gauge-libre). "prior_insolver": entra como factor
                # unario DENTRO de la BA de Python (rompe el gauge).
                self.disps_prior_[self.n] = d_seed.view(self.M)
                self.disps_prior_w_[self.n] = valid_seed.float()
            # factor de plano in-solver (P2): el runner entrega el plano cam-local
            # `[n0,n1,n2,d]` (RANSAC sobre la nube estéreo validada). Se guarda por
            # frame; la BA lo lleva a coords del mundo con la pose del frame.
            if (inject_mode == "prior_insolver") and (plane is not None):
                pl = torch.as_tensor(plane, dtype=torch.float, device="cuda").view(4)
                self.plane_n_[self.n] = pl[:3]
                self.plane_d_[self.n] = pl[3]
                self.plane_w_[self.n] = 1.0
                self.plane_set_counter += 1     # invalida el caché de π_world (refit)
        else:
            patches[:,:,2] = torch.rand_like(patches[:,:,2,0,0,None,None])
            if self.is_initialized:
                s = torch.median(self.patches_[self.n-3:self.n,:,2])
                patches[:,:,2] = s

        self.patches_[self.n] = patches

        ### update network attributes ###
        self.imap_[self.n % self.mem] = imap.squeeze()
        self.gmap_[self.n % self.mem] = gmap.squeeze()
        self.fmap1_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
        self.fmap2_[:, self.n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

        self.counter += 1        
        if self.n > 0 and not self.is_initialized:
            if self.motion_probe() < 2.0:
                self.delta[self.counter - 1] = (self.counter - 2, Id[0])
                return

        self.n += 1
        self.m += self.M

        # relative pose
        self.append_factors(*self.__edges_forw())
        self.append_factors(*self.__edges_back())

        if self.n == 8 and not self.is_initialized:
            self.is_initialized = True            

            for itr in range(12):
                self.update()
        
        elif self.is_initialized:
            self.update()
            self.keyframe()

