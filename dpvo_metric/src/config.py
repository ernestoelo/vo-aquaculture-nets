from yacs.config import CfgNode as CN

_C = CN()

# max number of keyframes
_C.BUFFER_SIZE = 2048

# bias patch selection towards high gradient regions?
_C.GRADIENT_BIAS = True

# patch selection strategy (campaña Hito 3, reducción de jitter):
#   'random'      -> muestreo uniforme (DPVO original; GRADIENT_BIAS lo overridea)
#   'gradient'    -> top-gradiente GLOBAL (equiv. a GRADIENT_BIAS=True)
#   'bucket'      -> grilla espacial + mejor parche por celda: cobertura uniforme
#                    Y alto gradiente -> mejor condicionamiento geométrico del BA.
#   'depth_valid' -> validez de depth ZED como FILTRO de candidatos + ranking
#                    por gradiente entre los válidos (mejora #2 Hito 3; requiere
#                    pasar depth= al patchify — handoff 2026-06-12). Sin depth
#                    cae a 'random'.
#   'depth_valid_random' -> filtro de validez puro: aleatorio ENTRE los
#                    válidos (sin ranking por gradiente — aísla el filtro del
#                    clustering del híbrido). Sin depth cae a 'random'.
#   'auto'        -> deriva de GRADIENT_BIAS (compat: True->'gradient', False->'random').
_C.PATCH_SELECTION = 'auto'
# oversample por celda (modo bucket) / global (modo gradient): candidatos = K * P.
_C.PATCH_OVERSAMPLE = 8

# VO config (increase for better accuracy)
_C.PATCHES_PER_FRAME = 80
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.PATCH_LIFETIME = 12

# threshold for keyframe removal
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5

# camera motion model
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5

_C.MIXED_PRECISION = True

cfg = _C
