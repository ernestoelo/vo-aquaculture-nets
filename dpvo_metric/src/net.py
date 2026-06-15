import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from . import fastba
from . import altcorr
import lietorch
from lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.autocast
import matplotlib.pyplot as plt

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def __bucket_select(self, images, n, h, w, P, oversample):
        """Selección guiada con bucketing espacial (campaña Hito 3).

        Divide el plano de feature (h×w) en una grilla de ~P celdas con aspect
        ≈ w/h; en cada celda muestrea `oversample` candidatos al azar y se queda
        con el de mayor gradiente. Garantiza COBERTURA uniforme (vs el top-K
        global de `gradient`, que clusteriza en una región muy texturada y deja
        zonas sin parches → BA mal condicionado → jitter). Devuelve x, y long
        (n, P). Idea validada por MAC-VO §III-B (muestreo estructurado).
        """
        gx = max(1, int(round((P * w / max(h, 1)) ** 0.5)))
        gy = max(1, int(np.ceil(P / gx)))
        ncell = gx * gy
        K = max(1, int(oversample))
        cw = (w - 2.0) / gx
        ch = (h - 2.0) / gy
        cidx = torch.arange(ncell, device="cuda")
        x0 = 1.0 + (cidx % gx).float() * cw          # (ncell,) esquina de celda
        y0 = 1.0 + (cidx // gx).float() * ch
        rx = torch.rand(n, ncell, K, device="cuda")
        ry = torch.rand(n, ncell, K, device="cuda")
        xc = (x0.view(1, ncell, 1) + rx * cw).clamp(1, w - 2)   # (n, ncell, K)
        yc = (y0.view(1, ncell, 1) + ry * ch).clamp(1, h - 2)
        coords = torch.stack([xc.reshape(n, -1), yc.reshape(n, -1)], dim=-1)
        g = self.__image_gradient(images)
        gv = altcorr.patchify(g[0, :, None], coords, 0).view(n, ncell, K)
        best = gv.argmax(dim=2, keepdim=True)         # mejor candidato por celda
        xw = torch.gather(xc, 2, best).squeeze(-1)    # (n, ncell)
        yw = torch.gather(yc, 2, best).squeeze(-1)
        gw = torch.gather(gv, 2, best).squeeze(-1)
        if ncell > P:                                  # grilla excede P -> top-P por gradiente
            keep = torch.argsort(gw, dim=1)[:, -P:]
            xw = torch.gather(xw, 1, keep)
            yw = torch.gather(yw, 1, keep)
        return xw.round().long(), yw.round().long()

    def forward(self, images, patches_per_image=80, disps=None, gradient_bias=False,
                selection=None, oversample=8, depth=None, return_color=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size

        # estrategia de selección de parches (Hito 3). 'auto'/None deriva de
        # gradient_bias (compat): True ⇒ 'gradient', False ⇒ 'random'. Un valor
        # explícito ('random'|'gradient'|'bucket'|'depth_valid') tiene precedencia.
        if selection in (None, 'auto'):
            selection = 'gradient' if gradient_bias else 'random'
        if selection in ('depth_valid', 'depth_valid_random') and depth is None:
            # sin mapa de depth la validez es indecidible; caer a 'random'
            # (el control del benchmark), NO a 'gradient' (top-grad global
            # empeora el jitter — finding 2026-06-07).
            selection = 'random'

        if selection == 'bucket':
            x, y = self.__bucket_select(images, n, h, w, patches_per_image, oversample)

        elif selection in ('depth_valid', 'depth_valid_random'):
            # Mejora #2 Hito 3: validez de depth como FILTRO de candidatos +
            # ranking por gradiente ENTRE los válidos (híbrido). Mantiene el
            # mapa de depth completo (filtrar el mapa colapsa la escala uw:
            # finding 2026-06-12-depth-confidence-sweep-5m-cap) y NO impone
            # cobertura espacial (el bucket ciego dispersó parches a zonas
            # sin depth y rompió la escala — finding 2026-06-07). Fallback
            # obligatorio: si hay <P candidatos válidos (uw: frames con
            # 7-11% de depth válida) rellenan los inválidos, también
            # rankeados por gradiente.
            x = torch.randint(1, w-1, size=[n, oversample*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, oversample*patches_per_image], device="cuda")

            # los parches viven a res/4: mismo mapeo feature→imagen que
            # DPVO._seed_inverse_depth (u = RES·x), para que "candidato
            # válido" coincida 1:1 con "parche con ancla" en la siembra.
            H, W = depth.shape[-2], depth.shape[-1]
            u = (4 * x).clamp(0, W - 1)
            v = (4 * y).clamp(0, H - 1)
            Z = depth.view(H, W)[v, u]
            valid = torch.isfinite(Z) & (Z > 1e-3)
            self.last_valid_candidate_frac = float(valid.float().mean())

            if selection == 'depth_valid':
                # híbrido: válidos SIEMPRE por sobre inválidos; el gradiente
                # desempata dentro de cada grupo (offset > rango del frame).
                coords = torch.stack([x, y], dim=-1).float()
                g = self.__image_gradient(images)
                gv = altcorr.patchify(g[0,:,None], coords, 0).view(n, oversample*patches_per_image)
                span = (gv.max() - gv.min()).clamp(min=1.0)
                score = gv + valid.float() * 2.0 * span
            else:
                # filtro puro ('depth_valid_random'): aleatorio ENTRE los
                # válidos — aísla el filtro del ranking por gradiente, que
                # clusteriza y sesga (resultado sweep 2026-06-12: el híbrido
                # colapsa la escala aire a ~0.65-0.70 con ATE 10×).
                score = torch.rand(x.shape, device="cuda") + valid.float() * 2.0
            ix = torch.argsort(score, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif selection == 'gradient':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, oversample*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, oversample*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, oversample * patches_per_image)

            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False, device_type="cuda")
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

