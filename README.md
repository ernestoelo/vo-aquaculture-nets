# Benchmarking de Odometría Visual para Mallas de Acuicultura — Hito 2 (IPD441)

Entrega de código del Hito 2: implementación y evaluación comparativa de
algoritmos de odometría visual (VO) sobre mallas de acuicultura con una
cámara estéreo ZED 2i, en aire y bajo el agua.

**Autores:** Ernesto Gamero, Fernanda Quintana — UTFSM.
**Informe:** el PDF escrito (formato IEEE-conf) se entrega por separado; este repositorio contiene el código de reproducción y evaluación.

---

## 1. Qué se evalúa

Cuatro configuraciones sobre la misma cámara y los mismos datos:

| Modelo | Qué es | Escala | fps (x86 RTX 3060) |
|---|---|---|---|
| **DPVO mono** | DPVO monocular [Teed 2023], punto de partida | arbitraria | ~27 |
| **DPVO métrico** | **nuestra modificación**: prior de profundidad estéreo *in-solver* | métrica | ~19 (6.1 embarcado) |
| **MAC-VO** | estado del arte estéreo + covarianza [Qiu 2025, ICRA Best Paper] | métrica | ~1.2 |
| **ZED SDK PT** | *positional tracking* del SDK (visión + IMU), referencia industrial | métrica | tiempo real (Jetson) |

## 2. La modificación: DPVO métrico

DPVO inicializa la profundidad inversa de cada parche **al azar** → escala
ambigua. Nuestra modificación inyecta la profundidad estéreo del SDK de ZED
como **factor unario dentro del bundle adjustment**:

```
E_prior = Σ_p  w_p · ( d_p − 1/Z_zed,p )²
```

sumado al residuo de reproyección. Anclar la profundidad *dentro* de la
optimización (no como reescalado posterior) recupera una escala estable.
Implementación: factor unario en el BA de Python del fork DPVO, ruteado con
`run_sdpvo_metric.py --inject prior_insolver --prior-strength 1000`
(no recompila CUDA). Ver `MANIFEST.md` para los archivos exactos.

## 3. Métricas

Calculadas contra el *ground truth* de cintas (cada 1 m) con
`eval/eval_ate_tape_gt.py`. Como el GT es colineal, el alineamiento de
Umeyama de `evo` degenera → se usa una variante **along-track** (proyección
sobre el eje principal). Métricas: **ATE**, **RPE traslacional (1 m)**,
**deriva %**, **escala GT/est**, y para secuencias de **ida y vuelta**
(loop) las métricas honestas de **cierre de lazo** (`loop_closure_m`,
`max_excursion_m`): el ATE along-track es ciego al colapso de un loop.

## 4. Reproducir

```bash
# 1. Construir el GT TUM desde los timecodes de cintas
python eval/build_gt_tum.py --config configs/tape_timecodes.yaml --out results/gt/

# 2. Correr un modelo (requiere el pipeline completo — ver MANIFEST.md)
python run_sdpvo_metric.py --config configs/runs/zed2i_gym_video1.yaml \
    --inject prior_insolver --prior-strength 1000

# 3. Evaluar contra el GT
python eval/eval_ate_tape_gt.py \
    --gt results/gt/gym_video_1_gt_tum.txt \
    --est results/<run_dir>/trajectory.txt --json out.json
```

## 5. Resultados (resumen)

| Secuencia | DPVO métrico (ATE) | MAC-VO | ZED PT |
|---|---|---|---|
| gym_v1 (aire, ~2 m malla) | **0.114 m** | 0.258 | 1.161 |
| gym_v2 (aire, ~1 m malla) | **0.257 m** | 1.771 | 2.664 |
| gym_v3 (aire, ~0.5 m malla) | **0.274 m** | 1.449 | 1.991 |
| video_4 (agua, loop) | 2.382 m | 3.007 | 2.660 |

**En aire** nuestra modificación supera a los métodos del estado del arte
evaluados; la ventaja crece a corta distancia de la malla (aliasing del
patrón repetitivo). **Bajo el agua**, en el loop, los tres colapsan
(cierre de lazo 3.8–4.9 m sobre un recorrido que debía cerrar en 0).

## 6. Dependencias upstream (no incluidas aquí)

- **DPVO** — `princeton-vl/DPVO` (pesos `dpvo.pth`).
- **MAC-VO** — `MAC-VO/MAC-VO` (paper Qiu et al. 2025).
- **ZED SDK** 5.x + `pyzed` (Stereolabs).
- **evo** — evaluación de trayectorias.

Ver `requirements.txt` y `MANIFEST.md`.

## 7. Trabajo futuro (Hito 3)

Fusión inercial con filtro de Kalman (recuperar el giro del loop) y
restricciones de plano como anclaje geométrico frente al patrón repetitivo
de la malla. Detalle en la sección final del informe.
