# MANIFEST: contenido de la entrega

Export curado del repositorio de trabajo. Incluye **el algoritmo y la
implementación** (modificación DPVO métrico + fusión inercial
tight-coupling del Hito 3), los scripts de corrida, el demo IMU on/off,
el marco de evaluación y los configs. Las dependencias upstream (DPVO
base, MAC-VO, ZED SDK) no se redistribuyen; se obtienen como se indica
abajo.

## La implementación (`dpvo_metric/`)

| Archivo | Rol |
|---|---|
| `dpvo_metric.patch` | **Nuestra contribución**: 1360 líneas sobre `MAC-VO/S_DPVO@f7266f7` |
| `src/imu_preint.py` | **nuevo (Hito 3)**: preintegración de Forster + jacobianos analíticos del factor IMU |
| `src/ba.py` | factor unario de profundidad (`E_prior`) + factor IMU (`_solve_augmented_imu`) en el BA |
| `src/dpvo.py` | ruteo del modo métrico, inyección del prior por-parche, enganche del stream IMU |
| `src/net.py` | estrategias de selección de parches (`PATCH_SELECTION`) |
| `src/config.py`, `src/scatter_ops.py`, `src/fastba/ba_cuda.cu` | claves de config, compat. Py3.8, firma BA CUDA |
| `config/*.yaml` | configs canónicos (x86, embarcado, recomendado métrico) |
| `LICENSE-DPVO` | MIT upstream (retenida) |
| `README.md` | dónde está cada cambio, la matemática de ambos factores, cómo reproducir |

## Scripts de corrida y demo (`scripts/`)

| Script | Rol |
|---|---|
| `run_sdpvo_metric.py` | DPVO métrico (`--inject prior_insolver`) + IMU (`--imu*`), planos/estéreo opt-in |
| `run_sdpvo_offline.py` | DPVO mono (baseline) |
| `run_zed_pt.py` | *positional tracking* del SDK de ZED |
| `extract_imu_svo.py` | **Hito 3**: extrae la IMU (~406 Hz) del SVO2 con gate de validez |
| `run_imu_siga_sweep.sh` | **Hito 3**: barrido de `--imu-sig-a` (desacople rumbo/escala) |
| `test_imu_jacobian.py` | **Hito 3**: self-test del factor IMU (jacobianos vs diferencias finitas) |
| `onoff_to_rerun.py` | **Hito 3**: demo on/off como `.rrd` de Rerun (video + planta + rumbo) |
| `onoff_demo_video.py` | **Hito 3**: demo on/off como `.mp4` (matplotlib + ffmpeg) |
| `svo_to_mp4_gt.py` | convierte SVO a mp4 con frame# quemado (insumo del demo) |
| `svo_to_stereo_pngs.py` | extrae pares estéreo left/right para MAC-VO |
| `smooth_trajectory.py` | filtro savgol post-proceso (longitud de arco) |
| `plane_anchor.py`, `stereo_triangulate.py` | anclajes geométricos evaluados (resultado negativo, opt-in) |

## Evaluación (`eval/`)

| Script | Rol |
|---|---|
| `eval_ate_tape_gt.py` | métricas along-track + cierre de lazo |
| `eval_heading_vs_gyro_gt.py` | **Hito 3, métrica líder**: rumbo de la cámara vs giroscopio |
| `eval_imu_siga_sweep.py` | **Hito 3**: escala por brazo + pico de rumbo del barrido `sig_a` |
| `eval_scale_arms_gt.py` | **Hito 3**: escala por brazo del recorrido vs GT de cintas |
| `build_gt_tum.py` | GT TUM desde los timecodes de cintas |
| `make_slide_resultados_benchmark.py` | tabla maestra del benchmark |

## Configs (`configs/`)

- `gt/tape_timecodes.yaml`: GT de cintas del Hito 2 (4 videos).
- `gt/gym_2026-06-30_giro_timecodes.yaml`: GT del dataset con giros del
  Hito 3 (recorte, cintas y giros por video).
- `calib/zed2i_gym_2026-06-30_hd720.txt`: calibración rectificada usada
  por el demo on/off.
- `runs/*.yaml`: config por secuencia (Hito 2).

## Upstream (no redistribuido, ver `NOTICE.md`)

- **DPVO base**: `MAC-VO/S_DPVO@f7266f7` (clonar + aplicar el patch).
- **MAC-VO**: `MAC-VO/MAC-VO` (paper Qiu et al. 2025).
- **ZED SDK** 5.x + `pyzed` (Stereolabs).
