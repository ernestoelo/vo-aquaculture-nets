# NOTICE: atribución de terceros

Este repositorio contiene código propio (marco de evaluación, scripts de
corrida y la modificación DPVO métrico) y código derivado de proyectos de
terceros, citados abajo. El informe escrito cita los artículos originales.

## DPVO (base de nuestra modificación)

- **Deep Patch Visual Odometry**, Z. Teed, L. Lipson, J. Deng. NeurIPS 2023.
- Repos: [`princeton-vl/DPVO`](https://github.com/princeton-vl/DPVO) ·
  fork usado [`MAC-VO/S_DPVO`](https://github.com/MAC-VO/S_DPVO) `@ f7266f7`.
- Licencia: **MIT** (Princeton Vision & Learning Lab, 2022); ver
  `dpvo_metric/LICENSE-DPVO`.
- Los archivos en `dpvo_metric/src/` y `dpvo_metric/dpvo_metric.patch` son
  versiones modificadas de archivos de ese repositorio.

## MAC-VO (método del estado del arte evaluado)

- **MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual
  Odometry**, Y. Qiu et al. ICRA 2025 (Best Conference Paper on Robot
  Perception). Repo: [`MAC-VO/MAC-VO`](https://github.com/MAC-VO/MAC-VO).
- Se ejecutó como baseline; no se redistribuye su código aquí.

## ZED SDK (referencia industrial)

- *Positional Tracking* del SDK de ZED (Stereolabs). Software propietario,
  no redistribuido; se usa vía `pyzed`.

## evo

- M. Grupp, *evo: Python package for the evaluation of odometry and SLAM*
  (2017). Usado como referencia de evaluación.
