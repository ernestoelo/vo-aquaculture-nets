# DPVO métrico + fusión inercial: nuestra modificación

Modificación de DPVO en dos etapas sobre el mismo solver:

1. **Escala métrica (Hito 2):** siembra la profundidad inversa de los
   parches con la profundidad estéreo del SDK de ZED, como **factor
   unario dentro del bundle adjustment** (no como reescalado posterior).
2. **Fusión inercial tight-coupling (Hito 3):** factor de
   **preintegración IMU** (método de Forster et al., TRO 2017) también
   **dentro del bundle adjustment**, que ancla el rumbo (heading) al
   giroscopio y estabiliza la escala.

- **Base:** [`MAC-VO/S_DPVO`](https://github.com/MAC-VO/S_DPVO) commit
  **`f7266f7`** (fork de [DPVO](https://github.com/princeton-vl/DPVO),
  Teed et al. 2023, MIT).
- **Nuestra contribución:** `dpvo_metric.patch` (1360 líneas; 7 archivos).

## El factor unario de profundidad (escala métrica)

DPVO inicializa la profundidad inversa de cada parche **al azar**
(`patches[:,:,2] = torch.rand_like(...)`) → escala ambigua. Añadimos al
bundle adjustment un término que atrae cada profundidad al valor estéreo
medido:

```
E_prior = Σ_p  w_p · ( d_p − 1/Z_zed,p )²
```

sumado al residuo de reproyección. Anclar la profundidad **dentro** de la
optimización (no después) cambia el condicionamiento de la triangulación
y recupera una escala estable y reproducible (~1.0 en aire).

## El factor inercial (tight coupling)

Las muestras IMU entre keyframes (~406 Hz en la ZED 2i) se
**preintegran** (`dpvo/imu_preint.py`) en tres pseudo-mediciones por
arista de keyframes consecutivos: rotación relativa `ΔR` (giroscopio),
velocidad `Δv` y posición `Δp` (acelerómetro + gravedad). El sistema
reducido de poses del BA se **aumenta con una velocidad de 3 estados por
keyframe** (`ba._solve_augmented_imu`) y se agregan los residuales de
Forster con bias ≈ 0 y gravedad fija:

```
r_ΔR = Log( ΔR_imuᵀ · R_iᵀ R_j )          (rumbo: giroscopio)
r_Δv = R_iᵀ (v_j − v_i − g·Δt) − Δv_imu    (traslación: acelerómetro)
r_Δp = R_iᵀ (p_j − p_i − v_i·Δt − ½g·Δt²) − Δp_imu
```

Detalles de implementación que importan:

- **Jacobiano de pose analítico** (convención retracción izquierda
  `Exp(δ)·G`): 6.8 veces más rápido por factor que el forward-diff
  numérico, +40% de fps end-to-end, sin cambio de resultados.
  Autoverificable con `scripts/test_imu_jacobian.py`.
- **Regularización de Tikhonov sobre la velocidad** (`--imu-v-reg`,
  default 10): remueve una dirección casi nula del estado aumentado
  pose ⊕ velocidad que podía inflar la escala (1 divergencia en 9 runs
  sin ella; 0 con ella).
- **Desacople heading/escala** (`--imu-sig-a`): pondera a la baja SOLO
  los residuales de traslación (`r_Δv`, `r_Δp`, del acelerómetro)
  dejando plena la restricción de rotación (`r_ΔR`, del giroscopio).
  Con `--imu-sig-a 15` el rumbo queda anclado al giroscopio (error de
  pico ≤ 1°) y la escala del plano queda near-metric (0.87 a 0.99 vs el
  ground truth de cintas), con el prior de profundidad mandando en la
  traslación.

El punto de operación validado (N=3, secuencias con giro de ~130°):
`--imu-strength 10 --imu-sig-a 15 --imu-v-reg 10`.

## Dónde está cada cambio

| Archivo | Cambio |
|---|---|
| `dpvo/imu_preint.py` | **nuevo**: preintegración de Forster + jacobianos analíticos del factor IMU |
| `dpvo/ba.py` | factor unario de profundidad (`E_prior`), factor IMU (`_solve_augmented_imu`), factor de plano (opt-in, evaluado negativo) |
| `dpvo/dpvo.py` | ruteo del modo métrico (`_ba_python_prior`), inyección del prior por-parche, enganche del stream IMU |
| `dpvo/net.py` | estrategias de selección de parches (`PATCH_SELECTION`: random / depth_valid) |
| `dpvo/config.py` | nuevas claves de configuración (modo de inyección, fuerza del prior) |
| `dpvo/scatter_ops.py` | `from __future__ import annotations` (compat. Python 3.8) |
| `dpvo/fastba/ba_cuda.cu` | ajuste menor de la firma del BA CUDA |

Los archivos modificados completos están en `src/` (para lectura
directa); `dpvo_metric.patch` es la fuente canónica de los cambios.

**Nota de honestidad experimental:** el patch incluye también los
factores geométricos de **plano** (coplanaridad in-solver, RANSAC por
ventana, restricción vertical por gravedad) y el front-end de
**triangulación estéreo dispersa**, evaluados sistemáticamente con
resultado **negativo** para esta geometría (malla repetitiva, órbita no
coplanar). Quedan con default OFF (`--plane-*`, `--stereo-*`) y
documentados en el informe como parte del avance del equipo.

## Reproducir

```bash
# 1. clonar el fork base en el commit exacto
git clone https://github.com/MAC-VO/S_DPVO.git
cd S_DPVO && git checkout f7266f7

# 2. aplicar nuestra modificación
git apply /ruta/a/dpvo_metric/dpvo_metric.patch

# 3. instalar (ver el README del fork) + copiar nuestros configs
cp /ruta/a/dpvo_metric/config/*.yaml config/

# 4a. correr el modo métrico (sin IMU)
python run_sdpvo_metric.py --svo <video.svo2> --calib <calib.txt> \
    --config-sdpvo config/sweep_p24_ow3_lt6.yaml \
    --inject prior_insolver --prior-strength 1000

# 4b. correr el modo métrico + IMU (tight coupling; el dir IMU se
#     genera antes con scripts/extract_imu_svo.py)
python run_sdpvo_metric.py --svo <video.svo2> --calib <calib.txt> \
    --config-sdpvo config/sweep_p24_ow3_lt6.yaml \
    --inject prior_insolver --prior-strength 1000 \
    --imu results/imu/<secuencia> --imu-strength 10 --imu-sig-a 15
```

## Configs incluidos

| Config | Uso |
|---|---|
| `sweep_p24_ow3_lt6.yaml` | **recomendado** del modo métrico (24 parches, OW3/LT6) |
| `x86_smoke.yaml` | receta x86 para gym |
| `zedbox_lowmem.yaml` | receta embarcada (Jetson Orin NX) |

## Licencia

Los archivos de `src/` y el patch son derivados de DPVO (MIT, Princeton
Vision & Learning Lab 2022); ver `LICENSE-DPVO`. Nuestras adiciones se
liberan bajo la misma licencia. Ver `../NOTICE.md`.
