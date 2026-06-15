# MANIFEST — archivos de la entrega y dónde vive cada pieza

Este directorio es un **export curado** del repositorio de trabajo. Incluye
el marco de evaluación y los configs (autocontenidos); los *scripts* del
pipeline completo dependen de los submódulos upstream y se listan abajo con
su ruta en el repo de trabajo para quien quiera reproducir end-to-end.

## Incluido aquí (autocontenido)

| Archivo | Rol |
|---|---|
| `eval/eval_ate_tape_gt.py` | Métricas along-track + cierre de lazo (ATE/RPE/deriva/escala/loop) |
| `eval/build_gt_tum.py` | Construye el GT TUM desde los timecodes de cintas |
| `eval/make_slide_resultados_benchmark.py` | Genera la tabla maestra del benchmark |
| `configs/tape_timecodes.yaml` | Ground truth de cintas (los 4 videos) |
| `configs/runs/*.yaml` | Configs de corrida por secuencia |
| `README.md` | Portada: métodos, modificación, reproducción, resultados |
| `requirements.txt` | Dependencias del marco de evaluación |

## Pipeline completo (en el repo de trabajo, depende de upstream)

| Script | Rol |
|---|---|
| `run_sdpvo_metric.py` | Corre DPVO métrico (modificación: `--inject prior_insolver`) |
| `run_sdpvo_offline.py` | Corre DPVO mono (baseline) |
| `run_zed_pt.py` | Corre el *positional tracking* del SDK de ZED |
| `svo_to_stereo_pngs.py` | Extrae pares estéreo left/right para MAC-VO |
| `smooth_trajectory.py` | Filtro savgol post-proceso (longitud de arco) |

## La modificación DPVO métrico

El factor unario de profundidad (`E_prior`, ver README §2) se inyecta en el
**bundle adjustment de Python** del fork DPVO (`dpvo/ba.py`, ruteado desde
`dpvo/dpvo.py`). No recompila CUDA. Se activa con
`--inject prior_insolver --prior-strength 1000`. El fork base es
`princeton-vl/DPVO` adaptado a torch reciente.
