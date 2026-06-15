# DPVO métrico — nuestra modificación

Modificación de DPVO para recuperar **escala métrica** sembrando la
profundidad inversa de los parches con la profundidad estéreo del SDK de
ZED, como **factor unario dentro del bundle adjustment** (no como
reescalado posterior).

- **Base:** [`MAC-VO/S_DPVO`](https://github.com/MAC-VO/S_DPVO) commit
  **`f7266f7`** (fork de [DPVO](https://github.com/princeton-vl/DPVO),
  Teed et al. 2023, MIT).
- **Nuestra contribución:** `dpvo_metric.patch` (484 líneas; 6 archivos).

## El factor unario

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

## Dónde está cada cambio

| Archivo | Cambio |
|---|---|
| `dpvo/ba.py` | factor unario de profundidad en el BA de Python (`E_prior`) |
| `dpvo/dpvo.py` | ruteo del modo métrico (`_ba_python_prior`), inyección del prior por-parche |
| `dpvo/net.py` | estrategias de selección de parches (`PATCH_SELECTION`: random / depth_valid) |
| `dpvo/config.py` | nuevas claves de configuración (modo de inyección, fuerza del prior) |
| `dpvo/scatter_ops.py` | `from __future__ import annotations` (compat. Python 3.8) |
| `dpvo/fastba/ba_cuda.cu` | ajuste menor de la firma del BA CUDA |

Los archivos modificados completos están en `src/` (para lectura
directa); `dpvo_metric.patch` es la fuente canónica de los cambios.

## Reproducir

```bash
# 1. clonar el fork base en el commit exacto
git clone https://github.com/MAC-VO/S_DPVO.git
cd S_DPVO && git checkout f7266f7

# 2. aplicar nuestra modificación
git apply /ruta/a/dpvo_metric/dpvo_metric.patch

# 3. instalar (ver el README del fork) + copiar nuestros configs
cp /ruta/a/dpvo_metric/config/*.yaml config/

# 4. correr el modo métrico (desde scripts/ de este repo)
python run_sdpvo_metric.py --config config/sweep_p24_ow3_lt6.yaml \
    --inject prior_insolver --prior-strength 1000
```

## Configs incluidos

| Config | Uso |
|---|---|
| `sweep_p24_ow3_lt6.yaml` | **recomendado** del modo métrico (24 parches, OW3/LT6) |
| `x86_smoke.yaml` | receta x86 para gym |
| `zedbox_lowmem.yaml` | receta embarcada (Jetson Orin NX) |

## Licencia

Los archivos de `src/` y el patch son derivados de DPVO (MIT, Princeton
Vision & Learning Lab 2022) — ver `LICENSE-DPVO`. Nuestras adiciones se
liberan bajo la misma licencia. Ver `../NOTICE.md`.
