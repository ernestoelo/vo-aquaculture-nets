# MANIFEST — contenido de la entrega

Export curado del repositorio de trabajo. Incluye **el algoritmo y la
implementación** (modificación DPVO métrico), los scripts de corrida, el
marco de evaluación y los configs. Las dependencias upstream (DPVO base,
MAC-VO, ZED SDK) no se redistribuyen; se obtienen como se indica abajo.

## La implementación — `dpvo_metric/`

| Archivo | Rol |
|---|---|
| `dpvo_metric.patch` | **Nuestra contribución**: 484 líneas sobre `MAC-VO/S_DPVO@f7266f7` |
| `src/ba.py` | factor unario de profundidad en el BA (`E_prior`) |
| `src/dpvo.py` | ruteo del modo métrico, inyección del prior por-parche |
| `src/net.py` | estrategias de selección de parches (`PATCH_SELECTION`) |
| `src/config.py`, `src/scatter_ops.py`, `src/fastba/ba_cuda.cu` | claves de config, compat. Py3.8, firma BA CUDA |
| `config/*.yaml` | configs canónicos (x86, embarcado, recomendado métrico) |
| `LICENSE-DPVO` | MIT upstream (retenida) |
| `README.md` | dónde está cada cambio + cómo reproducir |

## Scripts de corrida — `scripts/`

| Script | Rol |
|---|---|
| `run_sdpvo_metric.py` | DPVO métrico (`--inject prior_insolver`) |
| `run_sdpvo_offline.py` | DPVO mono (baseline) |
| `run_zed_pt.py` | *positional tracking* del SDK de ZED |
| `svo_to_stereo_pngs.py` | extrae pares estéreo left/right para MAC-VO |
| `smooth_trajectory.py` | filtro savgol post-proceso (longitud de arco) |

## Evaluación — `eval/`

| Script | Rol |
|---|---|
| `eval_ate_tape_gt.py` | métricas along-track + cierre de lazo |
| `build_gt_tum.py` | GT TUM desde los timecodes de cintas |
| `make_slide_resultados_benchmark.py` | tabla maestra del benchmark |

## Configs — `configs/`

`tape_timecodes.yaml` (GT de cintas, 4 videos) + `runs/*.yaml`
(config por secuencia).

## Upstream (no redistribuido — ver `NOTICE.md`)

- **DPVO base** — `MAC-VO/S_DPVO@f7266f7` (clonar + aplicar el patch).
- **MAC-VO** — `MAC-VO/MAC-VO` (paper Qiu et al. 2025).
- **ZED SDK** 5.x + `pyzed` (Stereolabs).
