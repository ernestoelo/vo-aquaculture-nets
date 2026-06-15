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
        for _ in range(2):
            Gs, patches = _pyBA(
                Gs, patches, self.intrinsics, target, weight, 1e-4,
                self.ii, self.jj, self.kk, bounds, ep=10.0, fixedp=t0,
                structure_only=False, prior_d=prior_d, prior_w=prior_w,
                prior_strength=strength, quality_boost=qboost)
        self.poses_[:] = Gs.data[0]
        self.patches_[:] = patches.view(self.N, self.M, 3, self.P, self.P)

    def __call__(self, tstamp, image, intrinsics, depth=None):
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

